"""
Async LLM Runtime.

- LLM: MiniMaxAI/MiniMax-M2 via vLLM / NVIDIA NIM
  using the OpenAI-compatible Chat Completions API.
- Async client: aiohttp

- Resilience: ResilienceManager for circuit breaker + token bucket.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, Union, cast

import aiohttp
import numpy as np
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger("cortex.llm.async_runtime")

# =============================================================================
# Config & exception integration
# =============================================================================
try:
    import jsonschema
except ImportError:
    jsonschema = None


try:
    # If we are inside the Cortex stack, use its config & exceptions.
    from cortex.common.exceptions import (
        CircuitBreakerOpenError,
        ConfigurationError,
        LLMOutputSchemaError,
        ProviderError,
        RateLimitError,
        ValidationError,
    )
    from cortex.config.loader import get_config

    _config = get_config()
    _retry_cfg = getattr(_config, "retry", None)
    if _retry_cfg is None:
        raise ImportError("retry config missing; fall back to local RuntimeConfig")

except ImportError:
    # Standalone / fallback configuration
    class RetryConfig:
        def __init__(self) -> None:
            self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "5"))
            self.circuit_failure_threshold = int(
                os.getenv("LLM_CIRCUIT_FAILURE_THRESHOLD", "5")
            )
            self.circuit_reset_seconds = float(
                os.getenv("LLM_CIRCUIT_RESET_SECONDS", "60.0")
            )
            self.rate_limit_per_sec = float(os.getenv("LLM_RATE_LIMIT_PER_SEC", "50.0"))
            self.rate_limit_capacity = float(
                os.getenv("LLM_RATE_LIMIT_CAPACITY", "50.0")
            )

    class RuntimeConfig:
        def __init__(self) -> None:
            # LLM: Default to DO Inference API
            self.llm_model = os.getenv(
                "LLM_MODEL", os.getenv("OUTLOOKCORTEX_LLM_MODEL", "openai-gpt-oss-120b")
            )
            self.llm_base_url = os.getenv(
                "LLM_ENDPOINT",
                os.getenv(
                    "OUTLOOKCORTEX_DO_LLM_BASE_URL",
                    os.getenv("DO_LLM_BASE_URL", "https://inference.do-ai.run/v1"),
                ),
            )
            self.llm_api_key = os.getenv(
                "LLM_API_KEY",
                os.getenv("DO_LLM_API_KEY", os.getenv("KIMI_API_KEY", "EMPTY")),
            )
            self.retry = RetryConfig()

    _config = RuntimeConfig()
    _retry_cfg = _config.retry

    # Minimal standalone exceptions (use private classes, then alias)
    class LLMRuntimeError(Exception):
        pass

    class _ProviderError(LLMRuntimeError):
        pass

    class _RateLimitError(LLMRuntimeError):
        pass

    class _CircuitBreakerOpenError(LLMRuntimeError):
        pass

    class _ValidationError(LLMRuntimeError):
        pass

    class _LLMOutputSchemaError(LLMRuntimeError):
        def __init__(
            self,
            message: str,
            schema_name: str | None = None,
            raw_output: str | None = None,
            repair_attempts: int | None = None,
            error_code: str | None = None,
        ) -> None:
            super().__init__(message)
            self.schema_name = schema_name
            self.raw_output = raw_output
            self.repair_attempts = repair_attempts
            self.error_code = error_code

    class _ConfigurationError(LLMRuntimeError):
        pass

    # Public aliases (names used across module)
    ProviderError = _ProviderError
    RateLimitError = _RateLimitError
    CircuitBreakerOpenError = _CircuitBreakerOpenError
    ValidationError = _ValidationError
    LLMOutputSchemaError = _LLMOutputSchemaError
    ConfigurationError = _ConfigurationError


# Help the type checker: ensure _retry_cfg has required attributes
class RetryLike(Protocol):
    max_retries: int
    circuit_failure_threshold: int
    circuit_reset_seconds: float
    rate_limit_per_sec: float
    rate_limit_capacity: float


assert _retry_cfg is not None
_retry_cfg = cast(RetryLike, _retry_cfg)


# =============================================================================
# Resilience Manager (thread-safe circuit breaker + rate limiter)
# =============================================================================
class ResilienceManager:
    """
    Manages operational stability: circuit breaking and rate limiting.

    - Circuit breaker: closed / open / half-open with reset timeout.
    - Rate limiter: token bucket (tokens/s, capacity).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        # Circuit breaker state
        self.failures = 0
        self.last_failure_time = 0.0
        self.circuit_state = "closed"  # closed | open | half-open

        # Rate limiter state (token bucket)
        self.tokens = cast(RetryLike, _retry_cfg).rate_limit_capacity
        self.last_refill = time.time()

    # ---------------- Circuit breaker ----------------
    def check_circuit(self) -> None:
        """Raises CircuitBreakerOpenError if the circuit is open and not ready to probe."""
        with self._lock:
            if self.circuit_state == "open":
                elapsed = time.time() - self.last_failure_time
                if elapsed > cast(RetryLike, _retry_cfg).circuit_reset_seconds:
                    self.circuit_state = "half-open"
                    logger.info("Circuit breaker probing (half-open).")
                else:
                    remaining = (
                        cast(RetryLike, _retry_cfg).circuit_reset_seconds - elapsed
                    )
                    raise CircuitBreakerOpenError(
                        f"Circuit open. Retry in {remaining:.1f}s"
                    )

    def record_outcome(self, success: bool) -> None:
        """Update circuit breaker state based on outcome."""
        with self._lock:
            if success:
                if self.circuit_state != "closed":
                    logger.info("Circuit breaker recovered (closed).")
                self.circuit_state = "closed"
                self.failures = 0
                return

            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= cast(RetryLike, _retry_cfg).circuit_failure_threshold:
                if self.circuit_state != "open":
                    logger.warning(
                        "Circuit breaker tripped after %d failures.",
                        self.failures,
                    )
                self.circuit_state = "open"

    # ---------------- Rate limiter (token bucket) ----------------
    async def acquire_token(self) -> None:
        """
        Blocking token acquisition using a token bucket.

        Tokens refill at rate_limit_per_sec up to rate_limit_capacity.
        """
        while True:
            with self._lock:
                now = time.time()
                elapsed = now - self.last_refill
                refill = elapsed * cast(RetryLike, _retry_cfg).rate_limit_per_sec
                self.tokens = min(
                    cast(RetryLike, _retry_cfg).rate_limit_capacity,
                    self.tokens + refill,
                )
                self.last_refill = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return

                missing = 1 - self.tokens
                wait_time = max(
                    missing / max(cast(RetryLike, _retry_cfg).rate_limit_per_sec, 1e-6),
                    0.001,
                )

            await asyncio.sleep(wait_time)


# =============================================================================
# Provider abstractions
# =============================================================================
class AsyncBaseProvider(ABC):
    """Abstract base class for providers."""

    @abstractmethod
    async def complete(self, prompt: str, **kwargs: Any) -> str: ...

    @staticmethod
    def normalize_l2(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize rows (if not already) for Blueprint §7.2.1."""
        if vectors.size == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class AioHTTPVLLMProvider(AsyncBaseProvider):
    """
    OpenAI-compatible vLLM provider using aiohttp.
    """

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is not None:
            return self._session
        async with self._lock:
            if self._session is not None:
                return self._session
            self._session = aiohttp.ClientSession()
            return self._session

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Chat completion via MiniMax-M2 (OpenAI Chat Completions API).
        """
        try:
            session = await self._get_session()
            base_url = getattr(_config, "llm_base_url", None) or os.getenv(
                "LLM_ENDPOINT",
                os.getenv(
                    "OUTLOOKCORTEX_DO_LLM_BASE_URL",
                    "https://inference.do-ai.run/v1",
                ),
            )
            api_key = (
                getattr(_config, "llm_api_key", None)
                or os.getenv("LLM_API_KEY")
                or os.getenv("DO_LLM_API_KEY", os.getenv("KIMI_API_KEY", "EMPTY"))
            )
            model_name = (
                kwargs.get("model")
                or getattr(_config, "llm_model", None)
                or os.getenv(
                    "LLM_MODEL",
                    os.getenv("OUTLOOKCORTEX_LLM_MODEL", "openai-gpt-oss-120b"),
                )
            )

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            messages = kwargs.get("messages") or [{"role": "user", "content": prompt}]

            req: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.3),
                "max_tokens": int(kwargs.get("max_tokens", 2048)),
                "response_format": kwargs.get("response_format"),
            }

            url = f"{base_url.rstrip('/')}/chat/completions"

            async with session.post(url, headers=headers, json=req) as response:
                if response.status != 200:
                    raise ProviderError(
                        f"LLM API request failed with status {response.status}: {await response.text()}"
                    )

                resp_json = await response.json()

                if not resp_json.get("choices"):
                    raise ProviderError("No choices in completion response")

                msg = resp_json["choices"][0]["message"]
                content = msg.get("content", "")
                return content
        except Exception as e:
            raise ProviderError(f"LLM failed: {e}") from e


# =============================================================================
# Runtime orchestrator
# =============================================================================
class AsyncLLMRuntime:
    """
    Orchestrates provider + resilience.
    """

    def __init__(self) -> None:
        self.resilience = ResilienceManager()
        self.primary = AioHTTPVLLMProvider()
        self.retry_config = cast(RetryLike, _retry_cfg)
        self._max_retries = self.retry_config.max_retries

    async def _execute(self, func, *args, **kwargs):
        """
        Execute `func` with resilience.
        """

        @retry(
            retry=retry_if_exception_type(
                (ProviderError, RateLimitError, TimeoutError, ConnectionError)
            ),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            stop=stop_after_attempt(self._max_retries),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _attempt():
            await self.resilience.acquire_token()
            self.resilience.check_circuit()
            try:
                result = await func(*args, **kwargs)
                self.resilience.record_outcome(success=True)
                return result
            except Exception as e:
                if isinstance(
                    e,
                    (
                        ProviderError,
                        RateLimitError,
                        TimeoutError,
                        ConnectionError,
                    ),
                ):
                    self.resilience.record_outcome(success=False)
                raise

        return await _attempt()

    async def async_complete_text(self, prompt: str, **kwargs: Any) -> str:
        if "messages" not in kwargs and (
            not isinstance(prompt, str) or not prompt.strip()
        ):
            raise ValidationError(
                "prompt must be a non-empty string if 'messages' is not provided"
            )

        result = await self._execute(self.primary.complete, prompt=prompt, **kwargs)

        if not isinstance(result, str) or not result.strip():
            raise LLMOutputSchemaError(
                "Completion output empty or invalid",
                schema_name="text_completion",
                raw_output=str(result),
            )

        return result

    def _create_json_repair_prompt(
        self, last_error: str, raw_output: str, schema_json: str
    ) -> list[dict[str, str]]:
        """Creates a prompt to ask the model to repair a JSON output."""
        return [
            {"role": "system", "content": "You are a JSON correction expert."},
            {
                "role": "user",
                "content": (
                    "The following JSON output is invalid. "
                    "Fix it so it becomes valid JSON matching the schema.\n\n"
                    f"Original error:\n{last_error}\n\n"
                    f"Invalid JSON:\n{raw_output}\n\n"
                    f"Required JSON Schema:\n{schema_json}\n\n"
                    "Respond with ONLY the corrected JSON object."
                ),
            },
        ]

    async def async_complete_json(
        self,
        prompt: str | list[dict[str, Any]],
        schema: dict[str, Any],
        max_repair_attempts: int = 2,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(prompt, str):
            if not prompt.strip():
                raise ValidationError("prompt string must not be empty")
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and prompt:
            messages = [dict(m) for m in prompt]
        else:
            raise ValidationError(
                "prompt must be a non-empty string or list of messages"
            )

        schema_json = json.dumps(schema, indent=2)
        json_instructions = (
            f"\n\nRespond with a single valid JSON object that conforms to this JSON Schema:\n"
            f"{schema_json}\n\nDo not include markdown. Return ONLY the JSON object."
        )
        messages[-1]["content"] += json_instructions

        async def _call_model_for_json(current_messages: list[dict[str, Any]]) -> str:
            base_kwargs = {
                "temperature": 0.0,
                "max_tokens": 2048,
                "response_format": {"type": "json_object"},
                **kwargs,
                "messages": current_messages,
            }
            try:
                return await self.async_complete_text("", **base_kwargs)
            except ProviderError as e:
                if (
                    "response_format" in str(e).lower()
                    or "json_object" in str(e).lower()
                ):
                    base_kwargs.pop("response_format", None)
                    return await self.async_complete_text("", **base_kwargs)
                raise

        raw_output: str | None = None
        last_error: str | None = None

        for attempt in range(max_repair_attempts + 1):
            current_prompt_messages = (
                messages
                if attempt == 0
                else self._create_json_repair_prompt(
                    str(last_error), str(raw_output), schema_json
                )
            )
            raw_output = await _call_model_for_json(current_prompt_messages)

            try:
                parsed = _try_load_json(raw_output or "")
                validation_error = _validate_json_schema(parsed, schema)
                if validation_error:
                    last_error = validation_error
                    logger.warning(
                        "JSON schema validation failed (attempt %d): %s",
                        attempt + 1,
                        validation_error,
                    )
                    continue
                return parsed
            except (ValueError, Exception) as e:
                last_error = str(e)
                logger.warning(
                    "JSON processing failed (attempt %d): %s", attempt + 1, last_error
                )

        raise LLMOutputSchemaError(
            message=f"Failed to generate valid JSON after {max_repair_attempts + 1} attempts: {last_error}",
            schema_name=schema.get("title", "unknown"),
            raw_output=(raw_output[:1000] if raw_output else None),
            repair_attempts=max_repair_attempts,
            error_code="JSON_SCHEMA_VALIDATION_FAILED",
        )


# =============================================================================
# JSON helpers
# =============================================================================
def _validate_json_schema(data: dict[str, Any], schema: dict[str, Any]) -> str | None:
    if not jsonschema:
        return None
    try:
        jsonschema.validate(data, schema)
        return None
    except Exception as e:
        return str(e)


def _extract_first_balanced_json_object(s: object) -> str | None:
    if not isinstance(s, str) or "{" not in s:
        return None

    first_brace = s.find("{")
    if first_brace == -1:
        return None

    balance = 0
    in_string = False
    escape_next = False
    for i, char in enumerate(s[first_brace:]):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
        if not in_string:
            if char == "{":
                balance += 1
            elif char == "}":
                balance -= 1
            if balance == 0:
                return s[first_brace : first_brace + i + 1]
    return None


def _parse_json_from_fenced_block(s: str) -> dict[str, Any] | None:
    """
    Extracts and parses JSON from the first markdown fenced block.

    The regex is designed to be ReDoS-safe by capturing content first,
    then stripping whitespace in Python, avoiding adjacent unbounded quantifiers.
    """
    fenced_matches = re.finditer(
        r"```(?:json|json5|hjson)?(.*?)```", s, flags=re.DOTALL | re.IGNORECASE
    )
    for m in fenced_matches:
        content = m.group(1).strip()
        block = _extract_first_balanced_json_object(content)
        if block:
            try:
                obj = json.loads(block)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def _try_load_json(data: Any) -> dict[str, Any]:
    """Attempts to parse a JSON object from various formats in a string."""
    if isinstance(data, dict):
        return data
    if not data:
        raise ValueError("Empty data for JSON parsing")

    s = str(data).strip()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE).strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    parsed_from_block = _parse_json_from_fenced_block(s)
    if parsed_from_block:
        return parsed_from_block

    balanced_block = _extract_first_balanced_json_object(s)
    if balanced_block:
        try:
            obj = json.loads(balanced_block)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Failed to parse JSON from string: {s[:500]!r} (Total len: {len(s)})"
    )
