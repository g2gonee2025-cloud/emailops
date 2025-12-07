"""
LLM Runtime.

Implements §7.2.1 of the Canonical Blueprint.
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import openai
import vertexai
from cortex.common.exceptions import (
    CircuitBreakerOpenError,
    ConfigurationError,
    LLMOutputSchemaError,
    ProviderError,
    RateLimitError,
)
from cortex.config.loader import get_config
from cortex.llm.doks_scaler import DigitalOceanLLMService
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from vertexai.generative_models import GenerationConfig, GenerativeModel
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

logger = logging.getLogger(__name__)
_config = get_config()

# -----------------------------------------------------------------------------
# Global State for Resilience (§7.2.1)
# -----------------------------------------------------------------------------
_vertex_initialized: bool = False
_PROJECT_ROTATION: Dict[str, Any] = {
    "projects": [],
    "current_index": 0,
    "consecutive_errors": 0,
    "_initialized": False,
}
_PROJECT_ROTATION_LOCK = threading.Lock()
_CIRCUIT_BREAKER: Dict[str, Any] = {
    "failures": 0,
    "last_failure_time": 0.0,
    "state": "closed",  # closed, open, half-open
}
_CIRCUIT_BREAKER_LOCK = threading.Lock()

# Retryable error substrings
RETRYABLE_SUBSTRINGS = (
    "quota exceeded",
    "resource_exhausted",
    "429",
    "temporarily unavailable",
    "rate limit",
    "deadline exceeded",
    "unavailable",
    "internal error",
    "503",
)


def _is_retryable_error(error: Exception) -> bool:
    """Check if an error should trigger retry."""
    error_str = str(error).lower()
    return any(sub in error_str for sub in RETRYABLE_SUBSTRINGS)


def _should_rotate_on(error: Exception) -> bool:
    """Check if error indicates quota exhaustion requiring project rotation."""
    error_str = str(error).lower()
    return "quota" in error_str or "resource_exhausted" in error_str


def _sleep_with_backoff(
    attempt: int, base_delay: float = 4.0, max_delay: float = 60.0
) -> None:
    """Sleep with exponential backoff and jitter."""
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    jitter = delay * random.uniform(-0.2, 0.2)
    time.sleep(delay + jitter)


# -----------------------------------------------------------------------------
# Circuit Breaker (§7.2.1)
# -----------------------------------------------------------------------------
def _check_circuit_breaker() -> bool:
    """
    Check if circuit breaker allows requests.
    Returns True if request should proceed, False if circuit is open.
    """
    with _CIRCUIT_BREAKER_LOCK:
        state = _CIRCUIT_BREAKER["state"]
        if state == "closed":
            return True
        elif state == "open":
            # Check if reset timeout has passed
            reset_seconds = _config.retry.circuit_reset_seconds
            if time.time() - _CIRCUIT_BREAKER["last_failure_time"] > reset_seconds:
                _CIRCUIT_BREAKER["state"] = "half-open"
                logger.info("Circuit breaker transitioning to half-open")
                return True
            return False
        else:  # half-open
            return True


def _record_circuit_success() -> None:
    """Record successful request for circuit breaker."""
    with _CIRCUIT_BREAKER_LOCK:
        if _CIRCUIT_BREAKER["state"] == "half-open":
            _CIRCUIT_BREAKER["state"] = "closed"
            _CIRCUIT_BREAKER["failures"] = 0
            logger.info("Circuit breaker closed after successful request")


def _record_circuit_failure() -> None:
    """Record failed request for circuit breaker."""
    with _CIRCUIT_BREAKER_LOCK:
        _CIRCUIT_BREAKER["failures"] += 1
        _CIRCUIT_BREAKER["last_failure_time"] = time.time()
        threshold = _config.retry.circuit_failure_threshold
        if _CIRCUIT_BREAKER["failures"] >= threshold:
            _CIRCUIT_BREAKER["state"] = "open"
            logger.warning(
                "Circuit breaker opened after %d consecutive failures", threshold
            )


# -----------------------------------------------------------------------------
# Project Rotation (§7.2.1)
# -----------------------------------------------------------------------------
def _ensure_projects_loaded() -> None:
    """Lazily load project configurations for rotation."""
    with _PROJECT_ROTATION_LOCK:
        if _PROJECT_ROTATION["_initialized"]:
            return

        # Load from validated_accounts.json if exists
        secrets_dir = Path(_config.directories.secrets_dir)
        accounts_file = secrets_dir / "validated_accounts.json"

        projects = []
        if accounts_file.exists():
            try:
                with accounts_file.open() as f:
                    accounts = json.load(f)
                for acc in accounts:
                    if acc.get("credentials_path") and acc.get("project_id"):
                        projects.append(
                            {
                                "project_id": acc["project_id"],
                                "credentials_path": acc["credentials_path"],
                            }
                        )
            except Exception as e:
                logger.warning("Failed to load validated_accounts.json: %s", e)

        # Fallback to current config
        if not projects:
            projects.append(
                {
                    "project_id": _config.gcp.gcp_project,
                    "credentials_path": os.environ.get(
                        "GOOGLE_APPLICATION_CREDENTIALS", ""
                    ),
                }
            )

        _PROJECT_ROTATION["projects"] = projects
        _PROJECT_ROTATION["_initialized"] = True
        logger.info("Loaded %d project(s) for rotation", len(projects))


def _rotate_to_next_project() -> bool:
    """
    Rotate to the next available project.
    Returns True if rotation succeeded, False if no more projects.
    """
    with _PROJECT_ROTATION_LOCK:
        _ensure_projects_loaded()
        projects = _PROJECT_ROTATION["projects"]
        if len(projects) <= 1:
            return False

        current_idx = _PROJECT_ROTATION["current_index"]
        next_idx = (current_idx + 1) % len(projects)
        _PROJECT_ROTATION["current_index"] = next_idx

        proj = projects[next_idx]
        os.environ["GCP_PROJECT"] = proj["project_id"]
        os.environ["GOOGLE_CLOUD_PROJECT"] = proj["project_id"]
        if proj.get("credentials_path"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = proj["credentials_path"]

        # Reset Vertex init so it reinitializes with new project
        global _vertex_initialized
        _vertex_initialized = False

        logger.info("Rotated to project: %s", proj["project_id"])
        return True


def reset_vertex_init() -> None:
    """Reset Vertex AI initialization state (useful for testing or re-init)."""
    global _vertex_initialized
    _vertex_initialized = False


# -----------------------------------------------------------------------------
# L2 Normalization (§7.2.1)
# -----------------------------------------------------------------------------
def _normalize_vectors(vectors: List[List[float]]) -> np.ndarray:
    """
    Normalize embedding vectors to unit length (L2 normalization).
    Required by Blueprint §7.2.1: Embeddings must be L2-normalized.
    """
    arr = np.array(vectors, dtype=np.float32)
    if arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Add small epsilon to prevent division by zero
    norms = np.where(norms > 0, norms, 1e-12)
    return arr / norms


class LLMRuntime:
    """
    LLM Runtime with resilience.

    Features per Blueprint §7.2.1:
    * Retry with exponential backoff
    * Circuit breaker (trips after N failures, resets after timeout)
    * Client-side rate limiting (token bucket)
    * Project/account rotation for quota management
    * L2-normalized embeddings
    """

    def __init__(self):
        self.retry_config = _config.retry
        self._rate_limit_tokens = _config.retry.rate_limit_capacity
        self._rate_limit_last_refill = time.time()
        self._rate_limit_lock = threading.Lock()
        self._doks_service: Optional[DigitalOceanLLMService] = None
        self._doks_lock = threading.Lock()
        self._init_vertex()

    def _init_vertex(self):
        """Initialize Vertex AI if configured."""
        global _vertex_initialized
        if _vertex_initialized:
            return

        if _config.core.provider == "vertex":
            try:
                _ensure_projects_loaded()
                vertexai.init(
                    project=_config.gcp.gcp_project,
                    location=_config.gcp.vertex_location,
                )
                _vertex_initialized = True
            except Exception as e:
                logger.warning(f"Failed to initialize Vertex AI: {e}")

    def _acquire_rate_limit(self) -> None:
        """
        Acquire a token from the rate limiter (token bucket algorithm).
        Blocks if no tokens available.
        """
        with self._rate_limit_lock:
            now = time.time()
            elapsed = now - self._rate_limit_last_refill

            # Refill tokens based on elapsed time
            refill = elapsed * self.retry_config.rate_limit_per_sec
            self._rate_limit_tokens = min(
                self.retry_config.rate_limit_capacity, self._rate_limit_tokens + refill
            )
            self._rate_limit_last_refill = now

            if self._rate_limit_tokens < 1:
                # Wait for token to become available
                wait_time = (
                    1 - self._rate_limit_tokens
                ) / self.retry_config.rate_limit_per_sec
                time.sleep(wait_time)
                self._rate_limit_tokens = 0
            else:
                self._rate_limit_tokens -= 1

    def _call_with_resilience(
        self, fn, *args, max_retries: Optional[int] = None, **kwargs
    ):
        """
        Call function with full resilience: circuit breaker, rate limiting, retry, rotation.
        """
        max_retries = max_retries or self.retry_config.max_retries

        for attempt in range(1, max_retries + 1):
            # Check circuit breaker
            if not _check_circuit_breaker():
                raise CircuitBreakerOpenError(
                    "Circuit breaker open - too many consecutive failures",
                    provider=_config.core.provider,
                    reset_at=_CIRCUIT_BREAKER["last_failure_time"]
                    + _config.retry.circuit_reset_seconds,
                )

            # Rate limiting
            self._acquire_rate_limit()

            try:
                result = fn(*args, **kwargs)
                _record_circuit_success()
                return result
            except Exception as e:
                _record_circuit_failure()

                # Check if it's a rate limit error
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str:
                    raise RateLimitError(
                        message=str(e),
                        provider=_config.core.provider,
                        retry_after=30.0,  # Default retry-after
                    ) from e

                if _should_rotate_on(e):
                    if _rotate_to_next_project():
                        self._init_vertex()
                        logger.info(
                            "Retrying after project rotation (attempt %d)", attempt
                        )
                        continue

                if _is_retryable_error(e) and attempt < max_retries:
                    _sleep_with_backoff(attempt)
                    logger.warning("Retrying after error (attempt %d): %s", attempt, e)
                    continue

                raise

        raise ProviderError(
            f"Max retries ({max_retries}) exceeded",
            provider=_config.core.provider,
            retryable=False,
        )

    def _get_doks_service(self) -> DigitalOceanLLMService:
        if self._doks_service is not None:
            return self._doks_service
        with self._doks_lock:
            if self._doks_service is None:
                if not hasattr(_config, "digitalocean"):
                    raise ConfigurationError(
                        "DigitalOcean configuration block is missing"
                    )
                self._doks_service = DigitalOceanLLMService(_config.digitalocean)
        return self._doks_service

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ProviderError),
    )
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of strings to embed

        Returns:
            numpy.ndarray of L2-normalized embedding vectors, shape (N, D)

        Note:
            Per Vertex AI API limits (official docs):
            - Max 250 texts per request
            - Max 20,000 tokens per request
            - Max 2,048 tokens per individual text
        """
        if not texts:
            return np.zeros(
                (0, _config.embedding.output_dimensionality), dtype=np.float32
            )

        provider = _config.core.provider
        output_dim = _config.embedding.output_dimensionality

        # Vertex AI API limit: max 250 texts per request
        VERTEX_MAX_BATCH_SIZE = 250

        def _do_embed():
            if provider == "vertex":
                self._init_vertex()
                model = TextEmbeddingModel.from_pretrained(_config.embedding.model_name)

                # Process in batches to respect API limits
                all_vectors = []
                for i in range(0, len(texts), VERTEX_MAX_BATCH_SIZE):
                    batch = texts[i : i + VERTEX_MAX_BATCH_SIZE]
                    # Use TextEmbeddingInput for proper task type and output_dimensionality
                    inputs = [
                        TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT")
                        for t in batch
                    ]
                    # Pass output_dimensionality to control embedding size (per official GCP docs)
                    embeddings = model.get_embeddings(
                        inputs,
                        output_dimensionality=output_dim if output_dim < 3072 else None,
                        auto_truncate=True,
                    )
                    all_vectors.extend([e.values for e in embeddings])
                return _normalize_vectors(all_vectors)

            elif provider == "openai":
                if not _config.sensitive.openai_api_key:
                    raise ConfigurationError("OpenAI API key not configured")

                # Use generic_embed_model if set, otherwise default to text-embedding-3-small
                embed_model = (
                    _config.embedding.generic_embed_model or "text-embedding-3-small"
                )

                client = openai.OpenAI(api_key=_config.sensitive.openai_api_key)
                response = client.embeddings.create(input=texts, model=embed_model)
                vectors = [d.embedding for d in response.data]
                return _normalize_vectors(vectors)

            elif provider == "azure_openai":
                if not _config.sensitive.azure_openai_api_key:
                    raise ConfigurationError("Azure OpenAI API key not configured")
                if not _config.sensitive.azure_openai_endpoint:
                    raise ConfigurationError("Azure OpenAI endpoint not configured")

                from openai import AzureOpenAI

                client = AzureOpenAI(
                    api_key=_config.sensitive.azure_openai_api_key,
                    api_version=_config.sensitive.azure_openai_api_version
                    or "2024-02-15-preview",
                    azure_endpoint=_config.sensitive.azure_openai_endpoint,
                )
                response = client.embeddings.create(
                    input=texts,
                    model=_config.sensitive.azure_openai_deployment
                    or "text-embedding-ada-002",
                )
                vectors = [d.embedding for d in response.data]
                return _normalize_vectors(vectors)

            elif provider == "digitalocean":
                service = self._get_doks_service()
                vectors = service.embed_texts(
                    texts,
                    expected_dim=_config.embedding.output_dimensionality,
                )
                return _normalize_vectors(vectors)

            else:
                raise ConfigurationError(f"Unsupported embedding provider: {provider}")

        try:
            result = self._call_with_resilience(_do_embed)

            # Validate embedding dimensions per Blueprint §7.2.1
            expected_dim = _config.embedding.output_dimensionality
            if result.shape[1] != expected_dim:
                logger.warning(
                    "Embedding dimension mismatch: got %d, expected %d",
                    result.shape[1],
                    expected_dim,
                )

            # Validate all values are finite
            if not np.all(np.isfinite(result)):
                logger.error("Embedding contains non-finite values")
                raise ProviderError(
                    "Invalid embedding values", provider=provider, retryable=False
                )

            return result

        except Exception as e:
            logger.error(f"Embedding failed ({provider}): {e}")
            raise ProviderError(
                f"Embedding failed: {e!s}", provider=provider, retryable=True
            ) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ProviderError),
    )
    def complete_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion.

        Args:
            prompt: Input prompt
            **kwargs: Additional provider args

        Returns:
            Generated text
        """
        provider = _config.core.provider

        def _do_complete():
            if provider == "vertex":
                self._init_vertex()
                model_name = _config.embedding.vertex_model
                model = GenerativeModel(model_name)

                config = GenerationConfig(
                    temperature=kwargs.get("temperature", 0.2),
                    max_output_tokens=kwargs.get("max_tokens", 1024),
                )

                response = model.generate_content(prompt, generation_config=config)
                return response.text

            elif provider == "openai":
                if not _config.sensitive.openai_api_key:
                    raise ConfigurationError("OpenAI API key not configured")

                client = openai.OpenAI(api_key=_config.sensitive.openai_api_key)
                response = client.chat.completions.create(
                    model=kwargs.get("model", "gpt-4-turbo-preview"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", 0.2),
                    max_tokens=kwargs.get("max_tokens", 1024),
                )
                return response.choices[0].message.content or ""

            elif provider == "azure_openai":
                if not _config.sensitive.azure_openai_api_key:
                    raise ConfigurationError("Azure OpenAI API key not configured")
                if not _config.sensitive.azure_openai_endpoint:
                    raise ConfigurationError("Azure OpenAI endpoint not configured")

                from openai import AzureOpenAI

                client = AzureOpenAI(
                    api_key=_config.sensitive.azure_openai_api_key,
                    api_version=_config.sensitive.azure_openai_api_version
                    or "2024-02-15-preview",
                    azure_endpoint=_config.sensitive.azure_openai_endpoint,
                )
                response = client.chat.completions.create(
                    model=_config.sensitive.azure_openai_deployment or "gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", 0.2),
                    max_tokens=kwargs.get("max_tokens", 1024),
                )
                return response.choices[0].message.content or ""

            elif provider == "digitalocean":
                service = self._get_doks_service()
                return service.generate_text(
                    prompt,
                    temperature=kwargs.get("temperature", 0.2),
                    max_tokens=kwargs.get("max_tokens", 1024),
                    model=kwargs.get("model"),
                )

            else:
                raise ConfigurationError(f"Unsupported completion provider: {provider}")

        try:
            return self._call_with_resilience(_do_complete)
        except Exception as e:
            logger.error(f"Completion failed ({provider}): {e}")
            raise ProviderError(
                f"Completion failed: {e!s}", provider=provider, retryable=True
            ) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ProviderError),
    )
    def complete_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        max_repair_attempts: int = 2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate JSON completion conforming to schema with repair fallback.

        Blueprint §7.2.1:
        - complete_json with schema + repair fallback
        - Raises LLMOutputSchemaError if validation fails after repair attempts

        Args:
            prompt: Input prompt
            schema: JSON schema for validation
            max_repair_attempts: Max attempts to repair invalid JSON
            **kwargs: Additional provider args

        Returns:
            Parsed and validated JSON dict

        Raises:
            LLMOutputSchemaError: If output doesn't match schema after repairs
        """
        provider = _config.core.provider

        # Append schema instruction to prompt if not implicit in provider
        json_prompt = f"{prompt}\n\nRespond with valid JSON conforming to:\n{json.dumps(schema, indent=2)}"

        def _do_complete_json():
            if provider == "vertex":
                self._init_vertex()
                model_name = _config.embedding.vertex_model
                model = GenerativeModel(model_name)

                # Vertex Gemini 1.5+ supports response_mime_type="application/json"
                config = GenerationConfig(
                    temperature=kwargs.get("temperature", 0.0),
                    max_output_tokens=kwargs.get("max_tokens", 2048),
                    response_mime_type="application/json",
                )

                response = model.generate_content(json_prompt, generation_config=config)
                return response.text

            elif provider == "openai":
                if not _config.sensitive.openai_api_key:
                    raise ConfigurationError("OpenAI API key not configured")

                client = openai.OpenAI(api_key=_config.sensitive.openai_api_key)
                response = client.chat.completions.create(
                    model=kwargs.get("model", "gpt-4-turbo-preview"),
                    messages=[{"role": "user", "content": json_prompt}],
                    temperature=kwargs.get("temperature", 0.0),
                    max_tokens=kwargs.get("max_tokens", 2048),
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content or "{}"

            elif provider == "azure_openai":
                if not _config.sensitive.azure_openai_api_key:
                    raise ConfigurationError("Azure OpenAI API key not configured")
                if not _config.sensitive.azure_openai_endpoint:
                    raise ConfigurationError("Azure OpenAI endpoint not configured")

                from openai import AzureOpenAI

                client = AzureOpenAI(
                    api_key=_config.sensitive.azure_openai_api_key,
                    api_version=_config.sensitive.azure_openai_api_version
                    or "2024-02-15-preview",
                    azure_endpoint=_config.sensitive.azure_openai_endpoint,
                )
                response = client.chat.completions.create(
                    model=_config.sensitive.azure_openai_deployment or "gpt-4",
                    messages=[{"role": "user", "content": json_prompt}],
                    temperature=kwargs.get("temperature", 0.0),
                    max_tokens=kwargs.get("max_tokens", 2048),
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content or "{}"

            elif provider == "digitalocean":
                service = self._get_doks_service()

                def _call_gateway() -> str:
                    return service.generate_text(
                        prompt=json_prompt,
                        temperature=kwargs.get("temperature", 0.0),
                        max_tokens=kwargs.get("max_tokens", 2048),
                        model=kwargs.get("model"),
                        response_format={"type": "json_object"},
                        extra_payload=kwargs.get("extra_payload"),
                    )

                return _call_gateway()

            else:
                raise ConfigurationError(f"Unsupported completion provider: {provider}")

        raw_output = ""
        last_error = None

        for repair_attempt in range(max_repair_attempts + 1):
            try:
                if repair_attempt == 0:
                    # Initial attempt
                    raw_output = self._call_with_resilience(_do_complete_json)
                else:
                    # Repair attempt using PROMPT_GUARDRAILS_REPAIR
                    repair_prompt = (
                        f"The following JSON output has an error. Fix it to match the schema.\n\n"
                        f"Original error: {last_error}\n\n"
                        f"Invalid JSON:\n{raw_output}\n\n"
                        f"Required schema:\n{json.dumps(schema, indent=2)}\n\n"
                        f"Respond with ONLY the corrected valid JSON."
                    )
                    raw_output = self._call_with_resilience(
                        lambda: self.complete_text(repair_prompt, temperature=0.0)
                    )

                # Parse JSON
                parsed = (
                    json.loads(raw_output)
                    if isinstance(raw_output, str)
                    else raw_output
                )

                # Validate against schema if jsonschema available
                validation_error = _validate_json_schema(parsed, schema)
                if validation_error:
                    last_error = validation_error
                    logger.warning(
                        "JSON schema validation failed (attempt %d): %s",
                        repair_attempt + 1,
                        validation_error,
                    )
                    continue

                return parsed

            except json.JSONDecodeError as e:
                last_error = f"JSON decode error: {e}"
                logger.warning(
                    "JSON decode failed (attempt %d): %s", repair_attempt + 1, e
                )

                # Try to extract JSON from text
                try:
                    extracted = _extract_json_from_text(raw_output)
                    if extracted:
                        validation_error = _validate_json_schema(extracted, schema)
                        if not validation_error:
                            return extracted
                        last_error = validation_error
                except Exception:
                    pass

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "JSON completion error (attempt %d): %s", repair_attempt + 1, e
                )

        # All attempts failed
        raise LLMOutputSchemaError(
            message=f"Failed to generate valid JSON after {max_repair_attempts + 1} attempts: {last_error}",
            schema_name=schema.get("title", "unknown"),
            raw_output=raw_output[:1000] if raw_output else None,
            repair_attempts=max_repair_attempts,
            error_code="JSON_SCHEMA_VALIDATION_FAILED",
        )


def _validate_json_schema(
    data: Dict[str, Any], schema: Dict[str, Any]
) -> Optional[str]:
    """
    Validate JSON data against a schema.
    Returns error message if invalid, None if valid.
    """
    try:
        import jsonschema

        jsonschema.validate(data, schema)
        return None
    except ImportError:
        # jsonschema not available, skip validation
        return None
    except jsonschema.ValidationError as e:
        return str(e.message)
    except Exception as e:
        return str(e)


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON from a text response that may contain markdown or other content.
    Falls back to empty dict if no valid JSON found.
    """
    import re

    # Try to find JSON in code blocks
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find first balanced JSON object
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
                try:
                    return json.loads(text[start_idx : i + 1])
                except json.JSONDecodeError:
                    start_idx = -1

    return {}


# Module-level singleton
_runtime: Optional[LLMRuntime] = None


def get_runtime() -> LLMRuntime:
    """Get or create the LLM runtime singleton."""
    global _runtime
    if _runtime is None:
        _runtime = LLMRuntime()
    return _runtime


# Convenience functions for module-level access
def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed texts using the runtime singleton."""
    return get_runtime().embed_texts(texts)


def complete_text(prompt: str, **kwargs) -> str:
    """Complete text using the runtime singleton."""
    return get_runtime().complete_text(prompt, **kwargs)


def complete_json(prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Complete JSON using the runtime singleton."""
    return get_runtime().complete_json(prompt, schema, **kwargs)
