"""
Configuration Models.

Implements §2.3 of the Canonical Blueprint.
All configuration models use Pydantic for validation benefits.
"""

from __future__ import annotations

import os
import socket
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    Field,
    PostgresDsn,
    RedisDsn,
    SecretStr,
    field_validator,
    model_validator,
)

# -----------------------------------------------------------------------------
# Environment Variable Helper
# -----------------------------------------------------------------------------


def _env(key: str, default: Any, value_type: type = str) -> Any:
    """
    Get environment variable with OUTLOOKCORTEX_ prefix fallback.

    Args:
        key: Environment variable name (without prefix)
        default: Default value if not set
        value_type: Type to convert value to

    Returns:
        Environment variable value or default
    """
    env_key_new = f"OUTLOOKCORTEX_{key}"
    env_key_legacy = f"EMAILOPS_{key}"

    value = os.getenv(env_key_new)
    source_key = env_key_new if value is not None else None
    if value is None:
        value = os.getenv(env_key_legacy)
        if value is not None:
            source_key = env_key_legacy
    if value is None:
        value = os.getenv(key)
        if value is not None:
            source_key = key

    if value is None:
        return default
    try:
        if value_type is bool:
            normalized = str(value).strip().lower()
            if normalized in ("true", "1", "yes", "on"):
                return True
            if normalized in ("false", "0", "no", "off"):
                return False
            raise ValueError("Invalid boolean value")
        if value_type is int:
            return int(value)
        if value_type is float:
            return float(value)
        return value
    except (ValueError, TypeError) as exc:
        key_name = source_key or key
        raise ValueError(
            f"Invalid value for {key_name}; expected {value_type.__name__}."
        ) from exc


def _env_list(key: str, default: str = "") -> list[str]:
    """Get a comma-separated env var as a list of trimmed strings."""
    raw = _env(key, default)
    if raw is None:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _is_local_host(host: str | None) -> bool:
    if not host:
        return False
    if host in _LOCAL_HOSTS:
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _unwrap_secret(value: SecretStr | str | None) -> str | None:
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value


# -----------------------------------------------------------------------------
# Directory Configuration
# -----------------------------------------------------------------------------


class DirectoryConfig(BaseModel):
    """Directory paths configuration."""

    index_dirname: str = Field(
        default_factory=lambda: _env("INDEX_DIRNAME", "_index"),
        description="Name of the index directory",
    )
    chunk_dirname: str = Field(
        default_factory=lambda: _env("CHUNK_DIRNAME", "_chunks"),
        description="Name of the chunks directory",
    )
    secrets_dir: Path = Field(
        default_factory=lambda: Path(_env("SECRETS_DIR", "secrets")),
        description="Path to secrets directory",
    )
    export_root: str = Field(
        default_factory=lambda: _env("EXPORT_ROOT", ""),
        description="Root directory for exports",
    )

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Storage Configuration (DigitalOcean Spaces / S3)
# -----------------------------------------------------------------------------


class StorageConfig(BaseModel):
    """Object storage configuration (DigitalOcean Spaces / S3)."""

    endpoint_url: AnyHttpUrl = Field(
        default_factory=lambda: _env(
            "S3_ENDPOINT", "https://nyc3.digitaloceanspaces.com"
        ),
        description="S3-compatible endpoint URL",
    )
    access_key: SecretStr | None = Field(
        default_factory=lambda: _env("S3_ACCESS_KEY", None), description="S3 access key"
    )
    secret_key: SecretStr | None = Field(
        default_factory=lambda: _env("S3_SECRET_KEY", None), description="S3 secret key"
    )
    bucket_raw: str = Field(
        default_factory=lambda: _env("S3_BUCKET_RAW", "emailops-bucket"),
        description="Bucket for raw email exports",
    )
    region: str = Field(
        default_factory=lambda: _env("S3_REGION", "nyc3"), description="S3 region"
    )

    @model_validator(mode="after")
    def validate_credentials(self) -> StorageConfig:
        """Require access/secret keys for non-local endpoints."""
        access_key = _unwrap_secret(self.access_key)
        secret_key = _unwrap_secret(self.secret_key)
        if _is_local_host(self.endpoint_url.host):
            return self
        if access_key and secret_key:
            return self
        if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
            return self
        raise ValueError("S3 access_key and secret_key are required.")

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Core Configuration
# -----------------------------------------------------------------------------


class CoreConfig(BaseModel):
    """Core application configuration."""

    env: Literal["dev", "staging", "prod"] = Field(
        default_factory=lambda: _env("ENV", "prod"),
        description="Environment name",
    )
    tenant_mode: Literal["single", "multi"] = Field(
        default_factory=lambda: _env("TENANT_MODE", "single"),
        description="Tenant isolation mode",
    )
    persona: str = Field(
        default_factory=lambda: _env("PERSONA", "GROUP INSURANCE SPECIALIST"),
        description="Default persona for LLM interactions",
    )
    provider: str = Field(
        default_factory=lambda: _env("EMBED_PROVIDER", "digitalocean"),
        description="Default LLM/embedding provider",
    )

    @field_validator("env", mode="before")
    def normalize_env(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized == "production":
            return "prod"
        return normalized

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Database Configuration
# -----------------------------------------------------------------------------


class DatabaseConfig(BaseModel):
    """Database connection configuration."""

    url: PostgresDsn | None = Field(
        default_factory=lambda: _env("DB_URL", None),
        description="Database connection URL (required - set via OUTLOOKCORTEX_DB_URL or DB_URL env var)",
    )
    pool_size: int = Field(
        default_factory=lambda: _env("DB_POOL_SIZE", 20, int),
        ge=1,
        le=100,
        description="Connection pool size",
    )
    max_overflow: int = Field(
        default_factory=lambda: _env("DB_MAX_OVERFLOW", 10, int),
        ge=0,
        le=50,
        description="Maximum pool overflow",
    )

    @model_validator(mode="after")
    def validate_db_url(self) -> DatabaseConfig:
        """Validate that required configuration is present."""
        if not self.url:
            raise ValueError(
                "Database URL is required. Set OUTLOOKCORTEX_DB_URL or DB_URL environment variable."
            )
        return self

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Redis Configuration
# -----------------------------------------------------------------------------


class RedisConfig(BaseModel):
    """Redis connection configuration."""

    url: RedisDsn = Field(
        default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379"),
        description="Redis connection URL",
    )
    host: str | None = Field(
        default_factory=lambda: _env("REDIS_HOST", None),
        description="Redis host",
    )
    port: int = Field(
        default_factory=lambda: _env("REDIS_PORT", 6379, int),
        description="Redis port",
    )
    password: SecretStr | None = Field(
        default_factory=lambda: _env("REDIS_PASSWORD", None),
        description="Redis password",
    )
    ssl: bool = Field(
        default_factory=lambda: _env("REDIS_SSL", False, bool),
        description="Use SSL for Redis connection",
    )

    @model_validator(mode="after")
    def validate_password(self) -> RedisConfig:
        """Require a password for non-local Redis endpoints."""
        if _is_local_host(self.url.host):
            return self
        url_password = self.url.password
        password_value = _unwrap_secret(self.password)
        if url_password or password_value:
            return self
        raise ValueError("Redis password is required for non-local endpoints.")

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Processing Configuration
# -----------------------------------------------------------------------------


class ProcessingConfig(BaseModel):
    """Text processing configuration."""

    chunk_size: int = Field(
        default_factory=lambda: _env("CHUNK_SIZE", 1600, int),
        ge=100,
        le=10000,
        description="Target chunk size in tokens",
    )
    chunk_overlap: int = Field(
        default_factory=lambda: _env("CHUNK_OVERLAP", 200, int),
        ge=0,
        le=500,
        description="Overlap between chunks in tokens",
    )
    batch_size: int = Field(
        default_factory=lambda: _env("EMBED_BATCH", 64, int),
        ge=1,
        le=1000,
        description="Batch size for embedding",
    )
    num_workers: int = Field(
        default_factory=lambda: _env("NUM_WORKERS", 8, int),
        ge=1,
        le=64,
        description="Number of parallel workers for GPU saturation",
    )

    @model_validator(mode="after")
    def validate_overlap(self) -> ProcessingConfig:
        """Ensure overlap is less than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Embedding Configuration (Blueprint §2.3)
# -----------------------------------------------------------------------------


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model_name: str = Field(
        default_factory=lambda: _env(
            "EMBED_MODEL", "tencent/KaLM-Embedding-Gemma3-12B-2511"
        ),
        description="Embedding model name (gateway-hosted)",
    )
    output_dimensionality: int = Field(
        default_factory=lambda: _env("EMBED_DIM", 3840, int),
        ge=64,
        le=4096,
        description="Output embedding dimension (must match DB vector column and gateway config)",
        validate_default=True,
    )
    batch_size: int = Field(
        default_factory=lambda: _env("KALM_EMBED_BATCH_SIZE", 256, int),
        ge=8,
        le=1024,
        description="Batch size for embedding inference (tune for GPU VRAM - 256 optimal for H200)",
    )
    max_seq_length: int = Field(
        default_factory=lambda: _env("KALM_MAX_SEQ_LENGTH", 512, int),
        ge=128,
        le=8192,
        description="Max sequence length for embedding model (KaLM recommends 512)",
    )

    # Vertex AI specific configuration removed
    # vertex_model: str = Field(...)

    # Generic option for other embedding models
    generic_embed_model: str | None = Field(
        default_factory=lambda: _env("GENERIC_EMBED_MODEL", None),
        description="Generic embedding model name for other providers (set via OUTLOOKCORTEX_GENERIC_EMBED_MODEL)",
    )

    # CPU fallback using GGUF quantized models
    gguf_model_path: str | None = Field(
        default_factory=lambda: _env("GGUF_MODEL_PATH", None),
        description="Path to GGUF quantized model for CPU fallback (e.g., models/kalm-12b-q4.gguf)",
    )
    cpu_fallback_enabled: bool = Field(
        default_factory=lambda: _env("CPU_FALLBACK_ENABLED", True, bool),
        description="Enable CPU fallback for query embedding when GPU endpoint is unavailable",
    )

    # Embedding mode selector (overrides cpu_fallback_enabled behavior)
    embed_mode: Literal["gpu", "cpu", "auto"] = Field(
        default_factory=lambda: _env("EMBED_MODE", "auto"),
        description="Embedding mode: 'gpu' (remote API only), 'cpu' (local GGUF only), 'auto' (GPU with CPU fallback)",
    )

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# GCP Configuration Removed
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# DigitalOcean LLM Configuration (DOKS scaling + inference)
# -----------------------------------------------------------------------------


class DigitalOceanScalerConfig(BaseModel):
    """Scaling controls for DigitalOcean Kubernetes GPU pools."""

    token: SecretStr | None = Field(
        default_factory=lambda: _env("DO_TOKEN", os.getenv("DIGITALOCEAN_TOKEN")),
        description="DigitalOcean API token (falls back to DIGITALOCEAN_TOKEN)",
    )
    cluster_id: str | None = Field(
        default_factory=lambda: _env("DO_CLUSTER_ID", None),
        description="Target DOKS cluster id (e.g., c-123)",
    )
    cluster_name: str | None = Field(
        default_factory=lambda: _env("DO_CLUSTER_NAME", None),
        description="Logical cluster name when provisioning",
    )
    node_pool_id: str | None = Field(
        default_factory=lambda: _env("DO_NODE_POOL_ID", None),
        description="GPU node pool id controlled by the scaler",
    )
    region: str | None = Field(
        default_factory=lambda: _env("DO_REGION", None),
        description="DigitalOcean region slug for provisioning",
    )
    kubernetes_version: str | None = Field(
        default_factory=lambda: _env("DO_K8S_VERSION", None),
        description="Desired DOKS Kubernetes version (e.g., 1.29.1-do.0)",
    )
    gpu_node_size: str | None = Field(
        default_factory=lambda: _env("DO_GPU_NODE_SIZE", None),
        description="Droplet size slug for GPU nodes (e.g., g-2vcpu-24gb)",
    )
    cluster_tags: list[str] = Field(
        default_factory=lambda: _env_list("DO_CLUSTER_TAGS"),
        description="Tags applied to the cluster when provisioning",
    )
    node_tags: list[str] = Field(
        default_factory=lambda: _env_list("DO_GPU_NODE_TAGS"),
        description="Tags applied to the GPU node pool when provisioning",
    )
    api_base_url: AnyHttpUrl = Field(
        default_factory=lambda: _env(
            "DO_API_BASE_URL", "https://api.digitalocean.com/v2"
        ),
        description="DigitalOcean API base URL",
    )
    memory_per_gpu_gb: float = Field(
        default_factory=lambda: _env("DO_GPU_MEMORY_GB", 141.0, float),
        gt=0.0,
        description="Usable GPU memory per accelerator (GB)",
    )
    gpus_per_node: int = Field(
        default_factory=lambda: _env("DO_GPUS_PER_NODE", 1, int),
        ge=1,
        description="GPUs available on each node in the pool",
    )
    headroom: float = Field(
        default_factory=lambda: _env("DO_GPU_HEADROOM", 0.2, float),
        ge=0.0,
        lt=1.0,
        description="Fractional VRAM headroom to reserve per GPU",
    )
    min_nodes: int = Field(
        default_factory=lambda: _env("DO_GPU_MIN_NODES", 0, int),
        ge=0,
        description="Minimum nodes to keep warm",
    )
    max_nodes: int = Field(
        default_factory=lambda: _env("DO_GPU_MAX_NODES", 4, int),
        ge=1,
        description="Maximum nodes allowed for the GPU pool",
    )
    max_scale_factor: float = Field(
        default_factory=lambda: _env("DO_GPU_MAX_SCALE_FACTOR", 2.0, float),
        gt=0.0,
        description="Upper bound multiplier when scaling up from warm state",
    )
    min_downscale_interval_s: int = Field(
        default_factory=lambda: _env("DO_GPU_DOWNSCALE_INTERVAL", 300, int),
        ge=0,
        description="Billing-aware hysteresis (seconds) before scaling down",
    )
    target_gpu_utilization: float = Field(
        default_factory=lambda: _env("DO_GPU_TARGET_UTIL", 0.7, float),
        ge=0.1,
        le=1.0,
        description="Target utilization used when sizing via tokens/sec",
    )
    dry_run: bool = Field(
        default_factory=lambda: _env("DO_GPU_DRY_RUN", False, bool),
        description="If true, scaler logs actions without mutating the API",
    )

    @field_validator("api_base_url")
    @classmethod
    def validate_api_base_url(cls, v: AnyHttpUrl) -> AnyHttpUrl:
        """Validate that the URL does not resolve to a private IP."""
        if not v:
            return v
        try:
            hostname = v.host
            if not hostname:
                raise ValueError("URL must have a valid hostname")

            ip = ip_address(socket.gethostbyname(hostname))
            if ip.is_private or ip.is_loopback or ip.is_reserved:
                raise ValueError(f"URL resolves to a non-public IP address: {ip}")
        except (socket.gaierror, ValueError) as e:
            raise ValueError(f"URL validation failed: {e}") from e
        return v

    @model_validator(mode="after")
    def validate_scaler_bounds(self) -> DigitalOceanScalerConfig:
        if self.min_nodes > self.max_nodes:
            raise ValueError("min_nodes must be less than or equal to max_nodes")
        if self.dry_run:
            return self
        token_value = _unwrap_secret(self.token)
        if token_value:
            return self
        if any(
            [
                self.cluster_id,
                self.cluster_name,
                self.node_pool_id,
                self.region,
                self.gpu_node_size,
            ]
        ):
            raise ValueError("DigitalOcean API token is required.")
        return self

    model_config = {"extra": "forbid"}


class DigitalOceanLLMModelConfig(BaseModel):
    """Describes the hosted LLM for sizing + throughput."""

    name: str = Field(
        default_factory=lambda: _env("DO_LLM_NAME", "minimax-m2"),
        description="Identifier for the hosted model",
    )
    params_total: float = Field(
        default_factory=lambda: _env("DO_LLM_PARAMS_TOTAL", 230.0, float),
        gt=0,
        description="Total parameters in billions (MoE aware)",
    )
    params_active: float | None = Field(
        default_factory=lambda: _env("DO_LLM_PARAMS_ACTIVE", 10.0, float),
        description="Active parameters per token in billions (MoE)",
    )
    context_length: int = Field(
        default_factory=lambda: _env("DO_LLM_CONTEXT", 204_800, int),
        ge=1_000,
        description="Max context window supported",
    )
    quantization: str = Field(
        default_factory=lambda: _env("DO_LLM_QUANT", "fp8"),
        description="Quantization key (fp16, int4, nf4, etc.)",
    )
    additional_memory_gb: float = Field(
        default_factory=lambda: _env("DO_LLM_OVERHEAD_GB", 8.0, float),
        ge=0.0,
        description="Fixed runtime VRAM overhead (GB)",
    )
    kv_bytes_per_token: float = Field(
        default_factory=lambda: _env("DO_LLM_KV_BYTES", 131_072.0, float),
        ge=32_768.0,
        description="KV cache bytes per token (default ~128KB/token)",
    )
    tps_per_gpu: float | None = Field(
        default_factory=lambda: _env("DO_LLM_TPS_PER_GPU", 60.0, float),
        ge=0.0,
        description="Measured tokens/sec per GPU",
    )
    max_concurrent_requests_per_gpu: int | None = Field(
        default_factory=lambda: _env("DO_LLM_CONCURRENCY", 6, int),
        ge=1,
        description="Concurrent requests per GPU that meet latency SLOs",
    )

    model_config = {"extra": "forbid"}


class DigitalOceanLLMEndpointConfig(BaseModel):
    """Inference endpoint exposed from the DOKS-hosted LLM."""

    BASE_URL: AnyHttpUrl | None = Field(
        default_factory=lambda: _env("DO_LLM_BASE_URL", None),
        description="Base URL for the OpenAI-compatible gateway running in DOKS",
    )
    completion_path: str = Field(
        default_factory=lambda: _env("DO_LLM_COMPLETIONS_PATH", "/v1/completions"),
        description="Relative path for completion requests",
    )
    embedding_path: str = Field(
        default_factory=lambda: _env("DO_LLM_EMBEDDINGS_PATH", "/v1/embeddings"),
        description="Relative path for embedding requests",
    )
    api_key: str | None = Field(
        default_factory=lambda: _env("DO_LLM_API_KEY", None),
        description="Bearer token forwarded to the gateway",
    )
    default_completion_model: str = Field(
        default_factory=lambda: _env("DO_LLM_COMPLETION_MODEL", "openai-gpt-oss-120b"),
        description="Model name to send to completion endpoint",
    )
    default_embedding_model: str = Field(
        default_factory=lambda: _env(
            "DO_LLM_EMBEDDING_MODEL", "tencent/KaLM-Embedding-Gemma3-12B-2511"
        ),
        description="Model name to send to embedding endpoint (serverless/gateway hosted)",
    )
    request_timeout_seconds: float = Field(
        default_factory=lambda: _env("DO_LLM_TIMEOUT", 45.0, float),
        ge=1.0,
        le=300.0,
        description="HTTP timeout when calling the inference gateway",
    )
    verify_tls: bool = Field(
        default_factory=lambda: _env("DO_LLM_VERIFY_TLS", True, bool),
        description="Toggle TLS verification for internal gateways",
    )
    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Optional static headers injected into every request",
    )

    model_config = {"extra": "forbid"}


class DigitalOceanLLMConfig(BaseModel):
    """Top-level DigitalOcean LLM configuration block."""

    scaling: DigitalOceanScalerConfig = Field(default_factory=DigitalOceanScalerConfig)
    model: DigitalOceanLLMModelConfig = Field(
        default_factory=DigitalOceanLLMModelConfig
    )
    endpoint: DigitalOceanLLMEndpointConfig = Field(
        default_factory=DigitalOceanLLMEndpointConfig
    )

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Retry Configuration (Blueprint §2.3)
# -----------------------------------------------------------------------------


class RetryConfig(BaseModel):
    """
    Retry and resilience configuration.

    Controls llm.runtime retry, backoff, circuit-breaker, and rate-limit behavior.
    """

    max_retries: int = Field(
        default_factory=lambda: _env("API_MAX_RETRIES", 3, int),
        ge=0,
        le=10,
        description="Maximum retry attempts",
    )
    initial_backoff_seconds: float = Field(
        default_factory=lambda: _env("API_BACKOFF_INITIAL", 1.0, float),
        ge=0.1,
        le=30.0,
        description="Initial backoff delay",
    )
    backoff_multiplier: float = Field(
        default_factory=lambda: _env("API_BACKOFF_MULTIPLIER", 2.0, float),
        ge=1.0,
        le=5.0,
        description="Backoff multiplier",
    )
    rate_limit_per_sec: float = Field(
        default_factory=lambda: _env("API_RATE_LIMIT", 5.0, float),
        ge=0.0,
        le=100.0,
        description="Rate limit (requests/sec)",
    )
    rate_limit_capacity: int = Field(
        default_factory=lambda: _env("API_RATE_LIMIT_CAPACITY", 10, int),
        ge=1,
        le=100,
        description="Rate limit bucket capacity",
    )
    circuit_failure_threshold: int = Field(
        default_factory=lambda: _env("CIRCUIT_FAILURE_THRESHOLD", 5, int),
        ge=1,
        le=50,
        description="Failures before circuit trips",
    )
    circuit_reset_seconds: int = Field(
        default_factory=lambda: _env("CIRCUIT_RESET_SECONDS", 60, int),
        ge=1,
        le=600,
        description="Circuit reset timeout",
    )

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Search Configuration (Blueprint §2.3)
# -----------------------------------------------------------------------------


class SearchConfig(BaseModel):
    """
    Search and retrieval configuration.

    Controls hybrid search, fusion, reranking, and recency boost behavior.
    """

    fusion_strategy: Literal["rrf", "weighted_sum"] = Field(
        default_factory=lambda: _env("FUSION_STRATEGY", "rrf"),
        description="Fusion strategy for hybrid search",
    )
    k: int = Field(
        default_factory=lambda: _env("SEARCH_K", 50, int),
        ge=1,
        le=500,
        description="Number of results to retrieve",
    )
    half_life_days: float = Field(
        default_factory=lambda: _env("HALF_LIFE_DAYS", 30.0, float),
        ge=1.0,
        le=365.0,
        description="Recency boost half-life",
    )
    recency_boost_strength: float = Field(
        default_factory=lambda: _env("RECENCY_BOOST_STRENGTH", 1.0, float),
        ge=0.0,
        le=5.0,
        description="Recency boost strength multiplier",
    )
    mmr_lambda: float = Field(
        default_factory=lambda: _env("MMR_LAMBDA", 0.5, float),
        ge=0.0,
        le=1.0,
        description="MMR diversity vs relevance tradeoff",
    )
    min_score: float = Field(
        default_factory=lambda: _env("MIN_SCORE", 0.0, float),
        ge=0.0,
        le=1.0,
        description="Minimum score threshold",
    )

    # Additional tuning parameters
    candidates_multiplier: int = Field(
        default_factory=lambda: _env("CANDIDATES_MULTIPLIER", 3, int),
        ge=1,
        le=10,
        description="Candidate pool multiplier",
    )
    sim_threshold: float = Field(
        default_factory=lambda: _env("SIM_THRESHOLD_DEFAULT", 0.30, float),
        ge=0.0,
        le=1.0,
        description="Similarity threshold",
    )
    reply_tokens: int = Field(
        default_factory=lambda: _env("REPLY_TOKENS_TARGET_DEFAULT", 20000, int),
        ge=1000,
        le=100000,
        description="Target tokens for reply context",
    )
    fresh_tokens: int = Field(
        default_factory=lambda: _env("FRESH_TOKENS_TARGET_DEFAULT", 10000, int),
        ge=1000,
        le=50000,
        description="Target tokens for fresh email context",
    )
    context_snippet_chars: int = Field(
        default_factory=lambda: _env("CONTEXT_SNIPPET_CHARS", 8000, int),
        ge=100,
        le=10000,
        description="Max chars per context snippet",
    )
    rerank_alpha: float = Field(
        default_factory=lambda: _env("RERANK_ALPHA", 0.35, float),
        ge=0.0,
        le=1.0,
        description="Reranking alpha blending factor",
    )
    reranker_endpoint: str | None = Field(
        default_factory=lambda: _env("RERANKER_ENDPOINT", None),
        description="External reranker endpoint (vLLM serving mxbai-rerank-large-v2)",
    )

    # Graph RAG Configuration (disabled by default for safe rollout)
    enable_graph_rag: bool = Field(
        default_factory=lambda: _env("SEARCH_ENABLE_GRAPH_RAG", False, bool),
        description="Enable knowledge graph retrieval in hybrid search",
    )
    graph_max_hops: int = Field(
        default_factory=lambda: _env("SEARCH_GRAPH_MAX_HOPS", 1, int),
        ge=1,
        le=3,
        description="Maximum graph traversal depth for entity expansion",
    )
    graph_entity_limit: int = Field(
        default_factory=lambda: _env("SEARCH_GRAPH_ENTITY_LIMIT", 10, int),
        ge=1,
        le=50,
        description="Maximum entities to extract from query",
    )
    graph_weight_in_fusion: float = Field(
        default_factory=lambda: _env("SEARCH_GRAPH_WEIGHT", 0.3, float),
        ge=0.0,
        le=1.0,
        description="Weight for graph signal in triple fusion (FTS + Vector + Graph)",
    )

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Email Configuration
# -----------------------------------------------------------------------------


class EmailConfig(BaseModel):
    """Email drafting and sending configuration."""

    reply_policy: str = Field(
        default_factory=lambda: _env("REPLY_POLICY", "reply_all"),
        description="Default reply policy",
    )
    sender_locked_name: str = Field(
        default_factory=lambda: _env("SENDER_LOCKED_NAME", ""),
        description="Locked sender display name",
    )
    sender_locked_email: str = Field(
        default_factory=lambda: _env("SENDER_LOCKED_EMAIL", ""),
        description="Locked sender email address",
    )
    message_id_domain: str = Field(
        default_factory=lambda: _env("MESSAGE_ID_DOMAIN", ""),
        description="Domain for Message-ID generation",
    )
    sender_reply_to: str = Field(
        default_factory=lambda: _env("SENDER_REPLY_TO", ""),
        description="Reply-To address",
    )
    allowed_senders: set[str] = Field(
        default_factory=lambda: {
            s.strip() for s in (_env("ALLOWED_SENDERS", "")).split(",") if s.strip()
        },
        description="Set of allowed sender addresses",
    )

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Summarizer Configuration
# -----------------------------------------------------------------------------


class SummarizerConfig(BaseModel):
    """Thread summarization configuration."""

    summarizer_version: str = Field(
        default_factory=lambda: _env("SUMMARIZER_VERSION", "2.2-facts-ledger"),
        description="Summarizer version",
    )
    summarizer_thread_max_chars: int = Field(
        default=16000, ge=1000, le=100000, description="Max thread chars"
    )
    summarizer_critic_max_chars: int = Field(
        default=5000, ge=500, le=20000, description="Max chars for critic pass"
    )
    summarizer_improve_max_chars: int = Field(
        default=8000, ge=1000, le=30000, description="Max chars for improve pass"
    )
    summarizer_max_participants: int = Field(
        default=25, ge=1, le=100, description="Max participants to track"
    )
    summarizer_max_summary_points: int = Field(
        default=25, ge=1, le=100, description="Max summary bullet points"
    )
    summarizer_max_next_actions: int = Field(
        default=50, ge=1, le=200, description="Max next actions"
    )
    summarizer_max_fact_items: int = Field(
        default=50, ge=1, le=200, description="Max fact items per category"
    )
    summarizer_subject_max_len: int = Field(
        default=100, ge=10, le=500, description="Max subject length"
    )
    audit_target_min_score: int = Field(
        default=8, ge=1, le=10, description="Minimum audit score target"
    )

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Limits Configuration
# -----------------------------------------------------------------------------


class LimitsConfig(BaseModel):
    """Resource limits configuration."""

    max_attachment_text_chars: int = Field(
        default=100000, ge=1000, le=10000000, description="Max chars per attachment"
    )
    excel_max_cells: int = Field(
        default=200000, ge=1000, le=10000000, description="Max Excel cells to process"
    )
    skip_attachment_over_mb: float = Field(
        default=25.0,
        ge=1.0,
        le=100.0,
        description="Skip attachments over this size (MB)",
    )
    max_total_attachments_mb: float = Field(
        default=500.0, ge=10.0, le=5000.0, description="Max total attachment size (MB)"
    )
    max_indexable_file_mb: float = Field(
        default=50.0, ge=1.0, le=500.0, description="Max file size to index (MB)"
    )
    max_indexable_chars: int = Field(
        default=5000000, ge=10000, le=100000000, description="Max chars to index"
    )
    max_chat_snippets: int = Field(
        default=50, ge=1, le=500, description="Max chat context snippets"
    )
    max_chat_context_mb: float = Field(
        default=10.0, ge=1.0, le=100.0, description="Max chat context size (MB)"
    )

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# System Configuration
# -----------------------------------------------------------------------------


class SystemConfig(BaseModel):
    """System-level configuration."""

    log_level: str = Field(
        default_factory=lambda: _env("LOG_LEVEL", "INFO"), description="Logging level"
    )
    command_timeout: int = Field(
        default=3600, ge=60, le=86400, description="Command timeout (seconds)"
    )
    active_window_seconds: int = Field(
        default=180, ge=10, le=3600, description="Active window duration"
    )
    file_encoding_cache_size: int = Field(
        default=128, ge=16, le=1024, description="File encoding cache size"
    )
    pip_timeout: int = Field(
        default=300, ge=30, le=1800, description="Pip command timeout"
    )

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Security Configuration
# -----------------------------------------------------------------------------


class SecurityConfig(BaseModel):
    """Security-related configuration."""

    allow_parent_traversal: bool = Field(default=False, description="Allow .. in paths")
    allow_provider_override: bool = Field(
        default=False, description="Allow provider override at runtime"
    )
    force_renorm: bool = Field(
        default=False, description="Force re-normalization of embeddings"
    )
    blocked_extensions: set[str] = Field(
        default_factory=lambda: {
            s.strip().lower()
            for s in _env(
                "BLOCKED_EXTENSIONS", ".exe,.bat,.cmd,.scr,.vbs,.js,.jar,.msi,.dll"
            ).split(",")
            if s.strip()
        },
        description="Blocked file extensions",
    )

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# PII Configuration
# -----------------------------------------------------------------------------


class PiiConfig(BaseModel):
    """PII processing configuration."""

    strict: bool = Field(
        default_factory=lambda: _env("PII_STRICT", False, bool),
        description="If true, require Presidio initialization; else fallback to regex without aborting.",
    )
    enabled: bool = Field(
        default_factory=lambda: _env("PII_ENABLED", False, bool),
        description="If false, disable all PII redaction.",
    )

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Sensitive Configuration (Credentials)
# -----------------------------------------------------------------------------


class SensitiveConfig(BaseModel):
    """
    Sensitive configuration (credentials).

    WARNING: Never log these values!
    """

    google_application_credentials: str | None = Field(
        default_factory=lambda: _env("GOOGLE_APPLICATION_CREDENTIALS", None),
        description="Path to GCP credentials JSON",
    )
    openai_api_key: str | None = Field(
        default_factory=lambda: _env("OPENAI_API_KEY", None),
        description="OpenAI API key",
    )
    cohere_api_key: str | None = Field(
        default_factory=lambda: _env("COHERE_API_KEY", None),
        description="Cohere API key",
    )
    huggingface_api_key: str | None = Field(
        default_factory=lambda: _env("HUGGINGFACE_API_KEY", None),
        description="HuggingFace API key",
    )
    qwen_api_key: str | None = Field(
        default_factory=lambda: _env("QWEN_API_KEY", None), description="Qwen API key"
    )
    qwen_base_url: str | None = Field(
        default_factory=lambda: _env("QWEN_BASE_URL", None),
        description="Qwen API base URL",
    )
    qwen_timeout: int | None = Field(
        default_factory=lambda: _env("QWEN_TIMEOUT", None, int),
        description="Qwen API timeout",
    )
    run_id: str | None = Field(
        default_factory=lambda: _env("RUN_ID", None),
        description="Unique run identifier",
    )

    def __repr__(self) -> str:
        """Return a PII-redacted representation of the sensitive config."""
        return (
            f"SensitiveConfig(google_application_credentials=REDACTED, "
            f"openai_api_key=REDACTED, cohere_api_key=REDACTED, "
            f"huggingface_api_key=REDACTED, qwen_api_key=REDACTED, "
            f"qwen_base_url='{self.qwen_base_url}', "
            f"qwen_timeout={self.qwen_timeout}, run_id='{self.run_id}')"
        )

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# File Patterns Configuration
# -----------------------------------------------------------------------------


class FilePatternsConfig(BaseModel):
    """File pattern matching configuration."""

    allowed_file_patterns: list[str] = Field(
        default_factory=lambda: [
            "*.pdf",
            "*.docx",
            "*.doc",
            "*.xlsx",
            "*.xls",
            "*.xlsm",
            "*.xlsb",
            "*.csv",
            "*.pptx",
            "*.ppt",
            "*.pptm",
            "*.txt",
            "*.md",
            "*.rtf",
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.gif",
            "*.tif",
            "*.tiff",
            "*.webp",
            "*.heic",
            "*.heif",
            "*.eml",
            "*.msg",
            "*.html",
            "*.htm",
            "*.json",
            "*.xml",
        ],
        description="Allowed file patterns for processing",
    )

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Unified Configuration
# -----------------------------------------------------------------------------


class UnifiedConfig(BaseModel):
    """Unified runtime configuration for sessions."""

    temperature: float = Field(
        default=0.2, ge=0.0, le=2.0, description="LLM temperature"
    )
    chat_session_id: str = Field(
        default="default", description="Chat session identifier"
    )
    max_chat_history: int = Field(
        default=5, ge=0, le=100, description="Max chat history entries"
    )
    last_to: str = Field(default="", description="Last To recipients")
    last_cc: str = Field(default="", description="Last CC recipients")
    last_subject: str = Field(default="", description="Last email subject")
    timeout_seconds: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Request timeout"
    )
    cache_ttl_seconds: float = Field(
        default=0.0, ge=0.0, le=86400.0, description="Cache TTL"
    )
    cache_max_entries: int = Field(
        default=1024, ge=0, le=100000, description="Max cache entries"
    )

    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Qdrant Configuration
# -----------------------------------------------------------------------------


class QdrantConfig(BaseModel):
    """Qdrant configuration."""

    enabled: bool = Field(
        default_factory=lambda: _env("QDRANT_ENABLED", False, bool),
        description="Whether to use Qdrant for vector search",
    )
    url: str = Field(
        default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333"),
        description="Qdrant API URL",
    )
    api_key: str | None = Field(
        default_factory=lambda: _env("QDRANT_API_KEY", None),
        description="Qdrant API Key",
    )
    collection_name: str = Field(
        default_factory=lambda: _env("QDRANT_COLLECTION", "chunks"),
        description="Qdrant collection name",
    )
