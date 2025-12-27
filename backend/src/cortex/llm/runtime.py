"""
LLM Runtime.

Implements §7.2.1 of the Canonical Blueprint with 2025-standard practices:

- LLM: MiniMaxAI/MiniMax-M2 via vLLM / NVIDIA NIM
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

            # Embedding: KaLM-Embedding-Gemma3-12B-2511 (dim=3840)
            self.embed_model = os.getenv(
                "EMBED_MODEL",
                os.getenv("KALM_EMBED_MODEL", "tencent/KaLM-Embedding-Gemma3-12B-2511"),
            )
            self.embed_dim = int(
                os.getenv("EMBED_DIM", os.getenv("KALM_EMBED_DIM", "3840"))
            )
            self.embed_device = os.getenv(
                "EMBED_DEVICE",
                os.getenv(
                    "KALM_EMBED_DEVICE",
                    "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu",
                ),
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
    def embed(self, texts: List[str]) -> np.ndarray: ...

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str: ...

    @staticmethod
    def normalize_l2(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize rows (if not already) for Blueprint §7.2.1."""
        if vectors.size == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class VLLMProvider(BaseProvider):
    """
    OpenAI-compatible vLLM provider.

    Defaults to MiniMaxAI/MiniMax-M2 when LLM_MODEL is not set.
    """

    """
    Primary provider (ONLY provider in this file):

    - LLM: MiniMaxAI/MiniMax-M2 served via vLLM / NIM
      exposing the OpenAI-compatible ChatCompletions API.

    - Embedding: tencent/KaLM-Embedding-Gemma3-12B-2511 via vLLM API (OpenAI-compatible).

    Retrieval helpers:
      - embed_queries()   -> client.embeddings.create()
      - embed_documents() -> client.embeddings.create()
    """

    def __init__(self) -> None:
        self._llm_client: Any | None = None
        self._embed_client: Any | None = None
        self._lock = threading.RLock()

    # ------------- LLM client (MiniMax-M2 via OpenAI-compatible server) -------------
    @property
    def llm_client(self):
        if self._llm_client is not None:
            return self._llm_client
        with self._lock:
            if self._llm_client is not None:
                return self._llm_client
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as e:
                raise ConfigurationError(
                    "Missing dependency 'openai'. Install with: pip install -U openai"
                ) from e

            # Resolve Base URL
            base_url = None
            # 1. Try unified config nesting (EmailOpsConfig)
            if hasattr(_config, "digitalocean"):
                do_cfg = getattr(_config, "digitalocean")
                endpoint_cfg = getattr(do_cfg, "endpoint", None)
                if endpoint_cfg and endpoint_cfg.BASE_URL:
                    base_url = str(endpoint_cfg.BASE_URL)

            # 2. Try simple config / env fallback (prefer LLM_ENDPOINT for DO Inference)
            if not base_url:
                base_url = getattr(_config, "llm_base_url", None) or os.getenv(
                    "LLM_ENDPOINT",
                    os.getenv(
                        "OUTLOOKCORTEX_DO_LLM_BASE_URL",
                        "https://inference.do-ai.run/v1",
                    ),
                )

            # Resolve API Key
            api_key = None
            # 1. Try unified config nesting
            if hasattr(_config, "digitalocean"):
                do_cfg = getattr(_config, "digitalocean")
                endpoint_cfg = getattr(do_cfg, "endpoint", None)
                if endpoint_cfg and endpoint_cfg.api_key:
                    api_key = endpoint_cfg.api_key

            # 2. Try simple config / env fallback
            if not api_key:
                config_key = getattr(_config, "llm_api_key", None)
                if config_key == "EMPTY":
                    config_key = None

                env_key = os.getenv("LLM_API_KEY")
                if env_key == "EMPTY":
                    env_key = None

                api_key = (
                    config_key
                    or env_key
                    or os.getenv("DO_LLM_API_KEY", os.getenv("KIMI_API_KEY", "EMPTY"))
                )

            # Resolve actual model name for logging context (used in complete_text but good to know here)
            # Note: The client is generic, but we log the INTENDED model if known from env.
            intended_model = os.getenv("LLM_MODEL", "MiniMax-M2 (Default)")

            logger.info(
                "Initializing OpenAI client for LLM Model '%s' at %s",
                intended_model,
                base_url,
            )
            self._llm_client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0)
            return self._llm_client

    # ------------- Embedding client (KaLM via vLLM API) -------------
    @property
    def embed_client(self):
        if self._embed_client is not None:
            return self._embed_client
        with self._lock:
            if self._embed_client is not None:
                return self._embed_client

            # Dependencies check
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ConfigurationError(
                    "Missing dependency 'openai'. Install with: pip install -U openai"
                ) from e

            # Endpoint resolution:
            # 1. EMBED_ENDPOINT (standard)
            # 2. DO_LLM_BASE_URL (legacy DigitalOcean scaler config)
            # 3. K8s Service DNS (fallback)
            base_url = (
                os.getenv("EMBED_ENDPOINT")
                or os.getenv("DO_LLM_BASE_URL")
                or "http://embeddings-api.emailops.svc.cluster.local"
            )

            # Ensure /v1 suffix
            if not base_url.endswith("/v1"):
                base_url = f"{base_url.rstrip('/')}/v1"

            api_key = os.getenv("EMBED_API_KEY", "EMPTY")

            logger.info("Initializing OpenAI client for Embeddings at %s", base_url)
            self._embed_client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0)
            return self._embed_client

    def _ensure_embed_model(self) -> None:
        # No-op: client is lazy-loaded in embed_client property
        pass

    # ------------- Retrieval embeddings -------------
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        if not documents:
            return np.array([], dtype=np.float32)

        client = self.embed_client

        # Resolve model name
        model_name = None
        if hasattr(_config, "embedding"):
            embed_cfg = getattr(_config, "embedding")
            if embed_cfg and embed_cfg.model_name:
                model_name = embed_cfg.model_name

        if not model_name:
            model_name = getattr(_config, "embed_model", None) or os.getenv(
                "EMBED_MODEL",
                os.getenv("KALM_EMBED_MODEL", "tencent/KaLM-Embedding-Gemma3-12B-2511"),
            )

        # Batching
        embedding_cfg = getattr(_config, "embedding", None)
        # Default batch size 256 for H100/H200 efficiency
        batch_size = getattr(embedding_cfg, "batch_size", 256) if embedding_cfg else 256

        all_embeddings = []

        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]

                # Replace newlines if configured (common for retrieval)
                # For Gemma-3-Embedding, it's usually robust, but consistency helps.
                # batch = [d.replace("\n", " ") for d in batch]

                resp = client.embeddings.create(
                    input=batch, model=model_name, encoding_format="float"
                )

                # Extract embeddings and sort by index to ensure order
                batch_data = sorted(resp.data, key=lambda x: x.index)
                batch_embeddings = [d.embedding for d in batch_data]
                all_embeddings.extend(batch_embeddings)

            arr = np.asarray(all_embeddings, dtype=np.float32)
            # vLLM/OpenAI embeddings are usually already normalized, but we enforce it for safety
            return self.normalize_l2(arr)

        except Exception as e:
            # Fallback or error reporting
            raise ProviderError(f"Embedding API failed: {e}") from e

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        # For asymmetric models, queries often need a prefix (e.g. "params: query: ...")
        # But KaLM-Embedding-Gemma3 usually handles raw text or uses specific instructions.
        # We'll treat them matches documents for now unless specific instruction template is needed.
        # If instruction is needed, it should be prepended here.
        return self.embed_documents(queries)

    # ------------- BaseProvider interface (defaults to document embeddings) -------------
    def embed(self, texts: List[str]) -> np.ndarray:
        # Back-compat: treat generic "texts" as documents for RAG ingestion
        return self.embed_documents(texts)

    def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Chat completion via MiniMax-M2 (OpenAI Chat Completions API).

        Supports vLLM extension params via extra_body (e.g. extra_body={"top_k": 50}).
        """
        try:
            if not messages:
                raise ValueError("`messages` list cannot be empty.")

            client = self.llm_client
            model_name = kwargs.get("model")

            if not model_name:
                # 1. Try unified config nesting
                if hasattr(_config, "digitalocean"):
                    do_cfg = getattr(_config, "digitalocean")
                    endpoint_cfg = getattr(do_cfg, "endpoint", None)
                    if endpoint_cfg and endpoint_cfg.default_completion_model:
                        model_name = endpoint_cfg.default_completion_model

                # 2. Fallback to simple field / env
                if not model_name:
                    model_name = getattr(_config, "llm_model", None) or os.getenv(
                        "LLM_MODEL",
                        os.getenv("OUTLOOKCORTEX_LLM_MODEL", "openai-gpt-oss-120b"),
                    )
            response_format = kwargs.get("response_format")
            extra_body = kwargs.get("extra_body")

            # NOTE: Keep this request object small. For long-context models,
            # client-side overhead can become noticeable if you do heavy post-processing here.
            req: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.3),
                # Prefer max_completion_tokens when supplied (newer OpenAI-compatible servers),
                # but keep max_tokens for broad compatibility.
                "max_tokens": int(kwargs.get("max_tokens", 2048)),
                "response_format": response_format,
                "extra_body": extra_body,
            }

            # Common OpenAI Chat Completions params (pass-through if provided)
            for k in (
                "top_p",
                "stop",
                "presence_penalty",
                "frequency_penalty",
                "seed",
                "n",
                "logprobs",
                "top_logprobs",
                "tools",
                "tool_choice",
                "parallel_tool_calls",
            ):
                if k in kwargs and kwargs[k] is not None:
                    req[k] = kwargs[k]

            # max_completion_tokens is preferred by some OpenAI-compatible servers; include if set.
            if (
                "max_completion_tokens" in kwargs
                and kwargs["max_completion_tokens"] is not None
            ):
                req["max_completion_tokens"] = int(kwargs["max_completion_tokens"])

            resp = client.chat.completions.create(**req)
            if not resp.choices:
                raise ProviderError("No choices in completion response")

            msg = resp.choices[0].message

            # vLLM reasoning parsers may surface reasoning in additional fields.
            # For MiniMax-M2, the recommended setup is to keep <think> blocks in the
            # conversation history, so we try to preserve them when present.
            raw_content = getattr(msg, "content", None) or ""
            content = raw_content if isinstance(raw_content, str) else str(raw_content)
            raw_reasoning = (
                getattr(msg, "reasoning", None)
                or getattr(msg, "reasoning_content", None)
                or ""
            )
            reasoning = raw_reasoning.strip() if isinstance(raw_reasoning, str) else ""

            if reasoning and "<think>" not in content:
                content = f"<think>{reasoning}</think>\n{content}".strip()
            return content
        except Exception as e:
            raise ProviderError(f"LLM failed: {e}") from e


# Backward-compat alias (older code may import KimiKaLMProvider)
KimiKaLMProvider = VLLMProvider


# =============================================================================
# Runtime orchestrator (MiniMax+KaLM only)
# =============================================================================
class LLMRuntime:
    """
    Orchestrates provider + resilience (MiniMax+KaLM only).

    - primary: VLLMProvider (MiniMax-M2 by default)
    - scaler: DigitalOceanLLMService (optional, for GPU pool autoscaling)
    """

    def __init__(self) -> None:
        self.resilience = ResilienceManager()
        self.primary = VLLMProvider()
        self.retry_config = cast(RetryLike, _retry_cfg)
        self._max_retries = self.retry_config.max_retries
        self._scaler = self._init_scaler()
        self._inflight = 0
        self._inflight_lock = threading.Lock()

    def _init_scaler(self):
        """
        Initialize the DigitalOcean GPU scaler if configured.

        NOTE: Custom scaler disabled in favor of DO native autoscaler.
        DO native autoscaler is more robust (runs in DO infrastructure)
        and prevents GPU nodes from staying running if Python app crashes.

        To enable DO native autoscaler, use:
            doctl kubernetes cluster node-pool update <cluster-id> <gpu-pool-id> \
                --auto-scale --min-nodes 0 --max-nodes 4
        """
        logger.info("Custom GPU scaler disabled - using DO native autoscaler")
        return None

        # Original custom scaler initialization (disabled)
        # try:
        #     from cortex.config.loader import get_config
        #     from cortex.llm.doks_scaler import DigitalOceanLLMService
        #
        #     config = get_config()
        #     do_config = getattr(config, "digitalocean", None)
        #     if do_config is None:
        #         logger.debug("DigitalOcean config not found; GPU scaler disabled")
        #         return None
        #
        #     # Check if cluster/pool IDs are configured
        #     scaling = getattr(do_config, "scaling", None)
        #     if not scaling or not scaling.cluster_id or not scaling.node_pool_id:
        #         logger.debug(
        #             "DO_CLUSTER_ID or DO_NODE_POOL_ID not set; GPU scaler disabled"
        #         )
        #         return None
        #
        #     scaler = DigitalOceanLLMService(do_config)
        #     logger.info(
        #         "DigitalOcean GPU scaler initialized (cluster=%s, pool=%s)",
        #         scaling.cluster_id[:8],
        #         scaling.node_pool_id[:8],
        #     )
        #     return scaler
        # except ImportError:
        #     logger.debug("doks_scaler module not available; GPU scaler disabled")
        #     return None
        # except Exception as e:
        #     logger.warning("Failed to initialize GPU scaler: %s", e)
        #     return None

    def _maybe_scale(self) -> None:
        """Trigger GPU scaling based on current request load."""
        if self._scaler is None:
            return
        try:
            with self._inflight_lock:
                inflight = self._inflight
            # The scaler's _maybe_scale method handles the actual scaling logic
            self._scaler._maybe_scale(inflight)
        except Exception as e:
            logger.warning("GPU scaling check failed: %s", e)

    def _track_request_start(self) -> None:
        """Track start of a request for scaling decisions."""
        with self._inflight_lock:
            self._inflight += 1
        self._maybe_scale()

    def _track_request_end(self) -> None:
        """Track end of a request for scaling decisions."""
        with self._inflight_lock:
            self._inflight = max(0, self._inflight - 1)

    # ------------- Core execution wrapper with resilience -------------
    def _execute(self, operation: str, func, *args, **kwargs):
        """
        Execute `func` with:

        - Token bucket rate limiting
        - Circuit breaker
        - Tenacity retries on ProviderError / RateLimitError / TimeoutError / ConnectionError
        - GPU autoscaling trigger (if configured)
        """
        # Track request for GPU scaling
        self._track_request_start()

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

        try:
            return _attempt()
        finally:
            self._track_request_end()

    # ---------------- Public API: embeddings (document/query) ----------------
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        if not documents:
            raise ValidationError("documents must be non-empty")
        if any(not isinstance(t, str) or not t.strip() for t in documents):
            raise ValidationError("documents must be non-empty strings")

        expected_dim = 3840
        if hasattr(_config, "embedding"):
            embed_cfg = getattr(_config, "embedding")
            if embed_cfg and embed_cfg.output_dimensionality:
                expected_dim = embed_cfg.output_dimensionality
        else:
            expected_dim = getattr(_config, "embed_dim", 3840)

        # Determine embed mode from config
        embed_mode = "auto"  # default
        cpu_fallback_enabled = True
        if hasattr(_config, "embedding"):
            embed_cfg = getattr(_config, "embedding")
            embed_mode = getattr(embed_cfg, "embed_mode", "auto")
            cpu_fallback_enabled = getattr(embed_cfg, "cpu_fallback_enabled", True)

        vectors = None
        gpu_error = None

        # CPU-only mode: skip GPU entirely
        if embed_mode == "cpu":
            try:
                from cortex.llm.gguf_provider import GGUFProvider

                gguf = GGUFProvider()
                if gguf.is_available():
                    logger.info(
                        "Using CPU GGUF for document embedding (embed_mode=cpu)"
                    )
                    vectors = gguf.embed(documents)
                else:
                    raise ProviderError(
                        "GGUF model not available but embed_mode=cpu requires it"
                    )
            except ImportError as e:
                raise ProviderError(f"GGUF provider not available: {e}") from e

        # GPU-only mode: no fallback
        elif embed_mode == "gpu":
            vectors = self._execute(
                "embed_documents", self.primary.embed_documents, documents
            )

        # Auto mode: GPU first, CPU fallback if enabled
        else:  # embed_mode == "auto"
            try:
                vectors = self._execute(
                    "embed_documents", self.primary.embed_documents, documents
                )
            except (ProviderError, ConnectionError, TimeoutError) as e:
                gpu_error = e
                logger.warning("GPU embedding failed, checking CPU fallback: %s", e)

                if cpu_fallback_enabled:
                    try:
                        from cortex.llm.gguf_provider import GGUFProvider

                        gguf = GGUFProvider()
                        if gguf.is_available():
                            logger.info(
                                "Using CPU GGUF fallback for document embedding"
                            )
                            vectors = gguf.embed(documents)
                        else:
                            logger.warning("GGUF model not available for CPU fallback")
                    except Exception as fallback_error:
                        logger.error("CPU fallback also failed: %s", fallback_error)

        # If no vectors from either source, raise original error
        if vectors is None:
            if gpu_error:
                raise gpu_error
            raise ProviderError("No embedding provider available")

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

        expected_dim = 3840
        if hasattr(_config, "embedding"):
            embed_cfg = getattr(_config, "embedding")
            if embed_cfg and embed_cfg.output_dimensionality:
                expected_dim = embed_cfg.output_dimensionality
        else:
            expected_dim = getattr(_config, "embed_dim", 3840)

        # Determine embed mode from config
        embed_mode = "auto"  # default
        cpu_fallback_enabled = True
        if hasattr(_config, "embedding"):
            embed_cfg = getattr(_config, "embedding")
            embed_mode = getattr(embed_cfg, "embed_mode", "auto")
            cpu_fallback_enabled = getattr(embed_cfg, "cpu_fallback_enabled", True)

        vectors = None
        gpu_error = None

        # CPU-only mode: skip GPU entirely
        if embed_mode == "cpu":
            try:
                from cortex.llm.gguf_provider import GGUFProvider

                gguf = GGUFProvider()
                if gguf.is_available():
                    logger.info("Using CPU GGUF for query embedding (embed_mode=cpu)")
                    vectors = gguf.embed_queries(queries)
                else:
                    raise ProviderError(
                        "GGUF model not available but embed_mode=cpu requires it"
                    )
            except ImportError as e:
                raise ProviderError(f"GGUF provider not available: {e}") from e

        # GPU-only mode: no fallback
        elif embed_mode == "gpu":
            vectors = self._execute(
                "embed_queries", self.primary.embed_queries, queries
            )

        # Auto mode: GPU first, CPU fallback if enabled
        else:  # embed_mode == "auto"
            try:
                vectors = self._execute(
                    "embed_queries", self.primary.embed_queries, queries
                )
            except (ProviderError, ConnectionError, TimeoutError) as e:
                gpu_error = e
                logger.warning("GPU embedding failed, checking CPU fallback: %s", e)

                if cpu_fallback_enabled:
                    try:
                        from cortex.llm.gguf_provider import GGUFProvider

                        gguf = GGUFProvider()
                        if gguf.is_available():
                            logger.info("Using CPU GGUF fallback for query embedding")
                            vectors = gguf.embed_queries(queries)
                        else:
                            logger.warning("GGUF model not available for CPU fallback")
                    except Exception as fallback_error:
                        logger.error("CPU fallback also failed: %s", fallback_error)

        # If no vectors from either source, raise original error
        if vectors is None:
            if gpu_error:
                raise gpu_error
            raise ProviderError("No embedding provider available")

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

    # ---------------- Public API: text completion (secure) ----------------
    def complete_messages(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> str:
        """Secure way to get text completion using a structured messages list."""
        if not isinstance(messages, list) or not messages:
            raise ValidationError("`messages` must be a non-empty list of dicts")

        result = self._execute("completion", self.primary.complete, messages, **kwargs)

        if not isinstance(result, str) or not result.strip():
            raise LLMOutputSchemaError(
                "Completion output empty or invalid",
                schema_name="text_completion",
                raw_output=str(result),
            )
        return result


    # ---------------- Public API: text completion (UNSAFE) ----------------
    def complete_text(self, prompt: str, **kwargs: Any) -> str:
        """
        DEPRECATED: Unsafe method for text completion.
        Use `complete_messages` instead.
        """
        import warnings

        warnings.warn(
            "`complete_text` is deprecated and insecure. Use `complete_messages`.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValidationError("prompt must be a non-empty string")

        messages = [{"role": "user", "content": prompt}]
        # It's possible old code passed 'messages' in kwargs, which would be confusing.
        # We prioritize the new messages list.
        kwargs.pop("messages", None)
        return self.complete_messages(messages, **kwargs)

    # ---------------- Public API: JSON completion (UNSAFE) ----------------
    def complete_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        max_repair_attempts: int = 2,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Unsafe method for JSON completion.
        Construct messages with `construct_prompt_messages` and use `complete_messages`.
        """
        import warnings

        warnings.warn(
            "`complete_json` is deprecated and insecure. Use `complete_messages`.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValidationError("prompt must be a non-empty string")

        schema_json = json.dumps(schema, indent=2)
        system_prompt = (
            f"Respond with a single valid JSON object that conforms to this JSON Schema:\n"
            f"{schema_json}\n\n"
            f"Do not include markdown. Return ONLY the JSON object."
        )

        initial_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        def _call_model_for_json(msgs: List[Dict[str, str]]) -> str:
            base_kwargs = dict(kwargs)
            base_kwargs.setdefault("temperature", 0.0)
            base_kwargs.setdefault("max_tokens", 2048)
            base_kwargs.setdefault("response_format", {"type": "json_object"})
            try:
                # Use the new secure method
                return self.complete_messages(msgs, **base_kwargs)
            except ProviderError as e:
                # Some OpenAI-compatible servers may not implement response_format.
                msg = str(e).lower()
                if "response_format" in msg or "json_object" in msg:
                    base_kwargs.pop("response_format", None)
                    return self.complete_messages(msgs, **base_kwargs)
                raise

        raw_output: Optional[str] = None
        last_error: Optional[str] = None

        for attempt in range(max_repair_attempts + 1):
            if attempt == 0:
                raw_output = _call_model_for_json(initial_messages)
            else:
                from cortex.prompts import (
                    SYSTEM_GUARDRAILS_REPAIR,
                    USER_GUARDRAILS_REPAIR,
                    construct_prompt_messages,
                )

                # Use the proper repair prompt templates
                repair_messages = construct_prompt_messages(
                    system_prompt_template=SYSTEM_GUARDRAILS_REPAIR,
                    user_prompt_template=USER_GUARDRAILS_REPAIR,
                    error=last_error or "Unknown",
                    invalid_json=raw_output or "",
                    target_schema=schema_json,
                    validation_errors=last_error or "N/A",
                )
                raw_output = _call_model_for_json(repair_messages)

            try:
                # Robust extraction first
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
            except ValueError as e:
                # _try_load_json raises ValueError on parsing failure
                last_error = str(e)
                logger.warning("JSON decode failed (attempt %d): %s", attempt + 1, e)

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
        import jsonschema

        jsonschema.validate(data, schema)
        return None
    except ImportError:
        return None
    except Exception as e:
        return str(e)


def _extract_first_balanced_json_object(s: object) -> str | None:
    """
    Finds the first balanced JSON object (from '{' to '}') in a string.
    Properly handles nested braces, strings with braces, and escaped characters.
    """
    if not isinstance(s, str):
        return None

    if not s or "{" not in s:
        return None

    first_brace = s.find("{")
    if first_brace == -1:
        return None

    balance = 0
    in_string = False
    escape_next = False

    try:
        # We only care about finding the matching closing brace for the first opening brace
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

                if balance == 0 and i > 0:
                    return s[first_brace : first_brace + i + 1]
    except (TypeError, IndexError):
        # Handle cases where s is not a string or slicing fails
        return None
    return None


def _try_load_json(data: Any) -> Dict[str, Any]:
    """
    Robustly parse JSON from model output, accepting dict, bytes, or string.
    Handles fenced code blocks and partial strings.
    """
    if isinstance(data, dict):
        return data  # type: ignore

    if not data:
        raise ValueError("Empty data for JSON parsing")

    s = data
    if isinstance(data, bytes):
        s = data.decode("utf-8")

    s = str(s).strip()

    # 0. Pre-processing: Remove reasoning traces (e.g., <think>...</think>)
    import re

    # Remove <think>...</think> (DOTALL to span newlines)
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE).strip()

    # 1. Try direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 2. Try fenced block extraction
    import re

    # Match ```json ... ``` or just ``` ... ```
    fenced_matches = list(
        re.finditer(
            r"```(?:json|json5|hjson)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE
        )
    )
    for m in fenced_matches:
        block = _extract_first_balanced_json_object(m.group(1))
        if block:
            try:
                obj = json.loads(block)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

    # 3. Fallback: find first balanced object in entire string
    block = _extract_first_balanced_json_object(s)
    if block:
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Failed to parse JSON from string: {s[:500]!r} (Total len: {len(s)})"
    )


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


def complete_messages(messages: List[Dict[str, str]], **kwargs: Any) -> str:
    return get_runtime().complete_messages(messages, **kwargs)


def complete_text(prompt: str, **kwargs: Any) -> str:
    return get_runtime().complete_text(prompt, **kwargs)


def complete_json(prompt: str, schema: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    return get_runtime().complete_json(prompt, schema, **kwargs)
