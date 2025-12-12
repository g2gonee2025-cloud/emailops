"""
LLM Runtime.

Implements §7.2.1 of the Canonical Blueprint with 2025-standard practices:

- LLM: nvidia/Kimi-K2-Thinking-NVFP4 via vLLM / NVIDIA NIM
  using the OpenAI-compatible Chat Completions API.

- Embedding: tencent/KaLM-Embedding-Gemma3-12B-2511 via SentenceTransformers
  with trust_remote_code, BF16, flash_attention_2, and L2-normalized outputs.

- Retrieval helpers: embed_queries() / embed_documents() call encode_query() /
  encode_document() (recommended for asymmetric retrieval with query/document split).

- Resilience: ResilienceManager for circuit breaker + token bucket.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, cast

import numpy as np
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger("cortex.llm.runtime")

# =============================================================================
# Config & exception integration
# =============================================================================

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
            self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
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
            # LLM: Kimi-K2 (served via vLLM OpenAI-compatible server)
            self.llm_model = os.getenv("KIMI_MODEL", "nvidia/Kimi-K2-Thinking-NVFP4")
            self.llm_base_url = os.getenv("KIMI_ENDPOINT", "http://localhost:8000/v1")
            self.llm_api_key = os.getenv("KIMI_API_KEY", "EMPTY")

            # Embedding: KaLM-Embedding-Gemma3-12B-2511 (dim=3840)
            self.embed_model = os.getenv(
                "KALM_EMBED_MODEL", "tencent/KaLM-Embedding-Gemma3-12B-2511"
            )
            self.embed_dim = int(os.getenv("KALM_EMBED_DIM", "3840"))
            self.embed_device = os.getenv(
                "KALM_EMBED_DEVICE",
                "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu",
            )
            self.embed_batch_size = int(os.getenv("KALM_EMBED_BATCH_SIZE", "32"))

            # Optional override; KaLM model card recommends 512
            self.embed_max_seq_length = int(os.getenv("KALM_MAX_SEQ_LENGTH", "512"))

            # Resilience
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
    def acquire_token(self) -> None:
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

            time.sleep(wait_time)


# =============================================================================
# Provider abstractions
# =============================================================================
class BaseProvider(ABC):
    """Abstract base class for providers."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        ...

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str:
        ...

    @staticmethod
    def normalize_l2(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize rows (if not already) for Blueprint §7.2.1."""
        if vectors.size == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class KimiKaLMProvider(BaseProvider):
    """
    Primary provider (ONLY provider in this file):

    - LLM: nvidia/Kimi-K2-Thinking-NVFP4 served via vLLM / NIM
      exposing the OpenAI-compatible ChatCompletions API.

    - Embedding: tencent/KaLM-Embedding-Gemma3-12B-2511 via SentenceTransformers.

    Retrieval helpers:
      - embed_queries()   -> SentenceTransformer.encode_query()
      - embed_documents() -> SentenceTransformer.encode_document()
    """

    def __init__(self) -> None:
        self._llm_client: Any | None = None
        self._embed_model: Any | None = None
        self._lock = threading.RLock()

    # ------------- LLM client (Kimi-K2 via OpenAI-compatible server) -------------
    @property
    def llm_client(self):
        if self._llm_client is not None:
            return self._llm_client
        with self._lock:
            if self._llm_client is not None:
                return self._llm_client
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as e:  # pragma: no cover
                raise ConfigurationError(
                    "Missing dependency 'openai'. Install with: pip install -U openai"
                ) from e

            base_url = getattr(_config, "llm_base_url", None) or os.getenv(
                "KIMI_ENDPOINT", "http://localhost:8000/v1"
            )
            api_key = getattr(_config, "llm_api_key", None) or os.getenv(
                "KIMI_API_KEY", "EMPTY"
            )

            logger.info("Initializing OpenAI client for Kimi-K2 at %s", base_url)
            self._llm_client = OpenAI(base_url=base_url, api_key=api_key)
            return self._llm_client

    # ------------- Embedding model (KaLM SentenceTransformer) -------------
    def _ensure_embed_model(self) -> None:
        if self._embed_model is not None:
            return
        with self._lock:
            if self._embed_model is not None:
                return
            try:
                import torch  # type: ignore
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as e:  # pragma: no cover
                raise ConfigurationError(
                    "Missing dependencies for embeddings. Install:\n"
                    "  pip install -U sentence-transformers torch"
                ) from e

            model_name = getattr(_config, "embed_model", None) or os.getenv(
                "KALM_EMBED_MODEL", "tencent/KaLM-Embedding-Gemma3-12B-2511"
            )
            device = getattr(_config, "embed_device", None) or os.getenv(
                "KALM_EMBED_DEVICE",
                "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu",
            )

            model_kwargs: Dict[str, Any] = {}
            if device.startswith("cuda"):
                model_kwargs["torch_dtype"] = getattr(torch, "bfloat16", torch.float16)
                model_kwargs["attn_implementation"] = "flash_attention_2"
            else:
                model_kwargs["torch_dtype"] = torch.float32

            logger.info("Loading embedding model %s on %s...", model_name, device)
            self._embed_model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device=device,
                model_kwargs=model_kwargs,
            )

            max_len = getattr(_config, "embed_max_seq_length", 512)
            cast(Any, self._embed_model).max_seq_length = int(max_len)

            # Hard check: we rely on these for retrieval
            if not hasattr(self._embed_model, "encode_query") or not hasattr(
                self._embed_model, "encode_document"
            ):
                raise ConfigurationError(
                    "SentenceTransformer model does not expose encode_query/encode_document. "
                    "Upgrade sentence-transformers and ensure trust_remote_code=True."
                )

    # ------------- Retrieval embeddings -------------
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        self._ensure_embed_model()
        batch_size = getattr(_config, "embed_batch_size", 32)
        try:
            embeddings = cast(Any, self._embed_model).encode_document(
                documents,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            arr = np.asarray(embeddings, dtype=np.float32)
            return self.normalize_l2(arr)
        except Exception as e:
            raise ProviderError(f"KaLM encode_document failed: {e}") from e

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        self._ensure_embed_model()
        batch_size = getattr(_config, "embed_batch_size", 32)
        try:
            embeddings = cast(Any, self._embed_model).encode_query(
                queries,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            arr = np.asarray(embeddings, dtype=np.float32)
            return self.normalize_l2(arr)
        except Exception as e:
            raise ProviderError(f"KaLM encode_query failed: {e}") from e

    # ------------- BaseProvider interface (defaults to document embeddings) -------------
    def embed(self, texts: List[str]) -> np.ndarray:
        # Back-compat: treat generic "texts" as documents for RAG ingestion
        return self.embed_documents(texts)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Chat completion via Kimi-K2 (OpenAI Chat Completions API).

        Supports vLLM extension params via extra_body (e.g. extra_body={"top_k": 50}).
        """
        try:
            client = self.llm_client
            model_name = (
                kwargs.get("model")
                or getattr(_config, "llm_model", None)
                or os.getenv("KIMI_MODEL", "nvidia/Kimi-K2-Thinking-NVFP4")
            )

            messages = kwargs.get("messages") or [{"role": "user", "content": prompt}]
            response_format = kwargs.get("response_format")
            extra_body = kwargs.get("extra_body")

            req: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.3),
                "max_tokens": kwargs.get("max_tokens", 2048),
                "response_format": response_format,
                "extra_body": extra_body,
            }

            # Pass-through optional common OpenAI params when provided
            for k in ("top_p", "stop", "presence_penalty", "frequency_penalty", "seed"):
                if k in kwargs and kwargs[k] is not None:
                    req[k] = kwargs[k]

            resp = client.chat.completions.create(**req)
            if not resp.choices:
                raise ProviderError("No choices in completion response")
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise ProviderError(f"Kimi LLM failed: {e}") from e


# =============================================================================
# Runtime orchestrator (Kimi+KaLM only)
# =============================================================================
class LLMRuntime:
    """
    Orchestrates provider + resilience (Kimi+KaLM only).

    - primary: KimiKaLMProvider
    """

    def __init__(self) -> None:
        self.resilience = ResilienceManager()
        self.primary = KimiKaLMProvider()
        self._max_retries = cast(RetryLike, _retry_cfg).max_retries

    # ------------- Core execution wrapper with resilience -------------
    def _execute(self, operation: str, func, *args, **kwargs):
        """
        Execute `func` with:

        - Token bucket rate limiting
        - Circuit breaker
        - Tenacity retries on ProviderError / RateLimitError / TimeoutError / ConnectionError
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
        def _attempt():
            self.resilience.acquire_token()
            self.resilience.check_circuit()
            try:
                result = func(*args, **kwargs)
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

        return _attempt()

    # ---------------- Public API: embeddings (document/query) ----------------
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        if not documents:
            raise ValidationError("documents must be non-empty")
        if any(not isinstance(t, str) or not t.strip() for t in documents):
            raise ValidationError("documents must be non-empty strings")

        expected_dim = getattr(_config, "embed_dim", 3840)

        vectors = self._execute(
            "embed_documents", self.primary.embed_documents, documents
        )

        if not isinstance(vectors, np.ndarray):
            vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.ndim != 2:
            raise ProviderError(
                f"Embedding output must be 2D, got shape {vectors.shape}"
            )

        vectors = BaseProvider.normalize_l2(vectors)

        if vectors.shape[1] != expected_dim:
            raise ProviderError(
                f"Embedding dimension mismatch: expected {expected_dim}, got {vectors.shape[1]}"
            )

        if not np.all(np.isfinite(vectors)):
            raise ProviderError("Embedding contains non-finite values")

        return vectors

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        if not queries:
            raise ValidationError("queries must be non-empty")
        if any(not isinstance(t, str) or not t.strip() for t in queries):
            raise ValidationError("queries must be non-empty strings")

        expected_dim = getattr(_config, "embed_dim", 3840)

        vectors = self._execute("embed_queries", self.primary.embed_queries, queries)

        if not isinstance(vectors, np.ndarray):
            vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.ndim != 2:
            raise ProviderError(
                f"Embedding output must be 2D, got shape {vectors.shape}"
            )

        vectors = BaseProvider.normalize_l2(vectors)

        if vectors.shape[1] != expected_dim:
            raise ProviderError(
                f"Embedding dimension mismatch: expected {expected_dim}, got {vectors.shape[1]}"
            )

        if not np.all(np.isfinite(vectors)):
            raise ProviderError("Embedding contains non-finite values")

        return vectors

    # Back-compat: embed_texts defaults to document embeddings (RAG ingestion)
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.embed_documents(texts)

    # ---------------- Public API: text completion ----------------
    def complete_text(self, prompt: str, **kwargs: Any) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValidationError("prompt must be a non-empty string")

        result = self._execute("completion", self.primary.complete, prompt, **kwargs)

        if not isinstance(result, str) or not result.strip():
            raise LLMOutputSchemaError(
                "Completion output empty or invalid",
                schema_name="text_completion",
                raw_output=str(result),
            )

        return result

    # ---------------- Public API: JSON completion ----------------
    def complete_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        max_repair_attempts: int = 2,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValidationError("prompt must be a non-empty string")

        schema_json = json.dumps(schema, indent=2)
        json_prompt = (
            f"{prompt}\n\n"
            f"Respond with a single valid JSON object that conforms to this JSON Schema:\n"
            f"{schema_json}\n\n"
            f"Do not include markdown. Return ONLY the JSON object."
        )

        def _call_model_for_json(p: str) -> str:
            base_kwargs = dict(kwargs)
            base_kwargs.setdefault("temperature", 0.0)
            base_kwargs.setdefault("max_tokens", 2048)
            base_kwargs.setdefault("response_format", {"type": "json_object"})
            try:
                return self.complete_text(p, **base_kwargs)
            except ProviderError as e:
                # Some OpenAI-compatible servers may not implement response_format.
                msg = str(e).lower()
                if "response_format" in msg or "json_object" in msg:
                    base_kwargs.pop("response_format", None)
                    return self.complete_text(p, **base_kwargs)
                raise

        raw_output: Optional[str] = None
        last_error: Optional[str] = None

        for attempt in range(max_repair_attempts + 1):
            if attempt == 0:
                raw_output = _call_model_for_json(json_prompt)
            else:
                repair_prompt = (
                    "The following JSON output is invalid. "
                    "Fix it so it becomes valid JSON matching the schema.\n\n"
                    f"Original error:\n{last_error}\n\n"
                    f"Invalid JSON:\n{raw_output}\n\n"
                    f"Required JSON Schema:\n{schema_json}\n\n"
                    "Respond with ONLY the corrected JSON object."
                )
                raw_output = _call_model_for_json(repair_prompt)

            try:
                clean = _strip_markdown_fences(raw_output or "")
                parsed = json.loads(clean)
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
            except json.JSONDecodeError as e:
                last_error = f"JSON decode error: {e}"
                logger.warning("JSON decode failed (attempt %d): %s", attempt + 1, e)

                try:
                    extracted = _extract_json_from_text(raw_output or "")
                    if extracted:
                        validation_error = _validate_json_schema(extracted, schema)
                        if not validation_error:
                            return extracted
                        last_error = validation_error
                except Exception:
                    pass
            except Exception as e:
                last_error = str(e)
                logger.warning("JSON completion error (attempt %d): %s", attempt + 1, e)

        raise LLMOutputSchemaError(
            message=(
                "Failed to generate valid JSON after "
                f"{max_repair_attempts + 1} attempts: {last_error}"
            ),
            schema_name=schema.get("title", "unknown"),
            raw_output=(raw_output[:1000] if raw_output else None),
            repair_attempts=max_repair_attempts,
            error_code="JSON_SCHEMA_VALIDATION_FAILED",
        )


# =============================================================================
# JSON helpers
# =============================================================================
def _validate_json_schema(
    data: Dict[str, Any], schema: Dict[str, Any]
) -> Optional[str]:
    try:
        import jsonschema  # type: ignore

        jsonschema.validate(data, schema)
        return None
    except ImportError:
        return None
    except Exception as e:
        return str(e)


def _strip_markdown_fences(text: str) -> str:
    t = (text or "").strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    import re

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    brace_count = 0
    start_idx = -1
    for i, c in enumerate(text):
        if c == "{":
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif c == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx >= 0:
                candidate = text[start_idx : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start_idx = -1
    return {}


# =============================================================================
# Module-level singleton and convenience functions
# =============================================================================
_runtime_instance: Optional[LLMRuntime] = None


def get_runtime() -> LLMRuntime:
    global _runtime_instance
    if _runtime_instance is None:
        _runtime_instance = LLMRuntime()
    return _runtime_instance


def embed_documents(documents: List[str]) -> np.ndarray:
    return get_runtime().embed_documents(documents)


def embed_queries(queries: List[str]) -> np.ndarray:
    return get_runtime().embed_queries(queries)


def embed_texts(texts: List[str]) -> np.ndarray:
    return get_runtime().embed_texts(texts)


def complete_text(prompt: str, **kwargs: Any) -> str:
    return get_runtime().complete_text(prompt, **kwargs)


def complete_json(prompt: str, schema: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    return get_runtime().complete_json(prompt, schema, **kwargs)
