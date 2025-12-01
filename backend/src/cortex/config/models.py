"""
Configuration Models.

Implements §2.3 of the Canonical Blueprint.
All configuration models use Pydantic for validation benefits.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Literal, Optional, Set

from pydantic import BaseModel, Field, field_validator


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
    value = os.getenv(env_key_new) or os.getenv(key)
    if value is None:
        return default
    try:
        if value_type is bool:
            return str(value).lower() in ("true", "1", "yes", "on")
        if value_type is int:
            return int(value)
        if value_type is float:
            return float(value)
        return value
    except (ValueError, TypeError):
        return default


# -----------------------------------------------------------------------------
# Directory Configuration
# -----------------------------------------------------------------------------

class DirectoryConfig(BaseModel):
    """Directory paths configuration."""
    
    index_dirname: str = Field(
        default_factory=lambda: _env("INDEX_DIRNAME", "_index"),
        description="Name of the index directory"
    )
    chunk_dirname: str = Field(
        default_factory=lambda: _env("CHUNK_DIRNAME", "_chunks"),
        description="Name of the chunks directory"
    )
    secrets_dir: Path = Field(
        default_factory=lambda: Path(_env("SECRETS_DIR", "secrets")),
        description="Path to secrets directory"
    )
    export_root: str = Field(
        default_factory=lambda: _env("EXPORT_ROOT", ""),
        description="Root directory for exports"
    )
    
    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Core Configuration
# -----------------------------------------------------------------------------

class CoreConfig(BaseModel):
    """Core application configuration."""
    
    env: Literal["dev", "staging", "prod"] = Field(
        default_factory=lambda: _env("ENV", "dev"),
        description="Environment name"
    )
    tenant_mode: Literal["single", "multi"] = Field(
        default_factory=lambda: _env("TENANT_MODE", "single"),
        description="Tenant isolation mode"
    )
    persona: str = Field(
        default_factory=lambda: _env("PERSONA", "expert insurance CSR"),
        description="Default persona for LLM interactions"
    )
    provider: str = Field(
        default_factory=lambda: _env("EMBED_PROVIDER", "vertex"),
        description="Default LLM/embedding provider"
    )
    
    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Database Configuration
# -----------------------------------------------------------------------------

class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    
    url: str = Field(
        default_factory=lambda: _env("DB_URL", "postgresql://postgres:postgres@localhost:5432/cortex"),
        description="Database connection URL"
    )
    pool_size: int = Field(
        default_factory=lambda: _env("DB_POOL_SIZE", 20, int),
        ge=1,
        le=100,
        description="Connection pool size"
    )
    max_overflow: int = Field(
        default_factory=lambda: _env("DB_MAX_OVERFLOW", 10, int),
        ge=0,
        le=50,
        description="Maximum pool overflow"
    )
    
    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Processing Configuration
# -----------------------------------------------------------------------------

class ProcessingConfig(BaseModel):
    """Text processing configuration."""
    
    chunk_size: int = Field(
        default=1600, ge=100, le=10000, description="Target chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=200, ge=0, le=500, description="Overlap between chunks in tokens"
    )
    batch_size: int = Field(
        default=64, ge=1, le=1000, description="Batch size for embedding"
    )
    num_workers: int = Field(
        default=4, ge=1, le=32, description="Number of parallel workers"
    )
    
    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 1600)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v
    
    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Embedding Configuration (Blueprint §2.3)
# -----------------------------------------------------------------------------

class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    
    model_name: str = Field(
        default_factory=lambda: _env("EMBED_MODEL", "gemini-embedding-001"),
        description="Embedding model name"
    )
    output_dimensionality: int = Field(
        default_factory=lambda: _env("EMBED_DIM", 3072, int),
        ge=256,
        le=4096,
        description="Output embedding dimension (must match DB vector column)"
    )
    
    # Vertex AI specific
    vertex_model: str = Field(
        default_factory=lambda: _env("VERTEX_MODEL", "gemini-2.5-pro"),
        description="Vertex AI model for completion"
    )
    
    # Generic option for other embedding models
    generic_embed_model: Optional[str] = Field(
        default_factory=lambda: _env("GENERIC_EMBED_MODEL", None),
        description="Generic embedding model name for other providers (set via OUTLOOKCORTEX_GENERIC_EMBED_MODEL)"
    )
    
    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# GCP Configuration
# -----------------------------------------------------------------------------

class GcpConfig(BaseModel):
    """Google Cloud Platform configuration."""
    
    gcp_region: str = Field(
        default_factory=lambda: _env("GCP_REGION", "global"),
        description="GCP region"
    )
    vertex_location: str = Field(
        default_factory=lambda: _env("VERTEX_LOCATION", "us-central1"),
        description="Vertex AI location"
    )
    gcp_project: str = Field(
        default_factory=lambda: _env("GCP_PROJECT", ""),
        description="GCP project ID"
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
        default=3, ge=0, le=10, description="Maximum retry attempts"
    )
    initial_backoff_seconds: float = Field(
        default=1.0, ge=0.1, le=30.0, description="Initial backoff delay"
    )
    backoff_multiplier: float = Field(
        default=2.0, ge=1.0, le=5.0, description="Backoff multiplier"
    )
    rate_limit_per_sec: float = Field(
        default=5.0, ge=0.0, le=100.0, description="Rate limit (requests/sec)"
    )
    rate_limit_capacity: int = Field(
        default=10, ge=1, le=100, description="Rate limit bucket capacity"
    )
    circuit_failure_threshold: int = Field(
        default=5, ge=1, le=50, description="Failures before circuit trips"
    )
    circuit_reset_seconds: int = Field(
        default=60, ge=1, le=600, description="Circuit reset timeout"
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
        default="rrf", description="Fusion strategy for hybrid search"
    )
    k: int = Field(
        default=50, ge=1, le=500, description="Number of results to retrieve"
    )
    half_life_days: float = Field(
        default=30.0, ge=1.0, le=365.0, description="Recency boost half-life"
    )
    recency_boost_strength: float = Field(
        default=1.0, ge=0.0, le=5.0, description="Recency boost strength multiplier"
    )
    mmr_lambda: float = Field(
        default=0.5, ge=0.0, le=1.0, description="MMR diversity vs relevance tradeoff"
    )
    min_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum score threshold"
    )
    
    # Additional tuning parameters
    candidates_multiplier: int = Field(
        default=3, ge=1, le=10, description="Candidate pool multiplier"
    )
    sim_threshold: float = Field(
        default=0.30, ge=0.0, le=1.0, description="Similarity threshold"
    )
    reply_tokens: int = Field(
        default=20000, ge=1000, le=100000, description="Target tokens for reply context"
    )
    fresh_tokens: int = Field(
        default=10000, ge=1000, le=50000, description="Target tokens for fresh email context"
    )
    context_snippet_chars: int = Field(
        default=1600, ge=100, le=10000, description="Max chars per context snippet"
    )
    rerank_alpha: float = Field(
        default=0.35, ge=0.0, le=1.0, description="Reranking alpha blending factor"
    )
    
    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Email Configuration
# -----------------------------------------------------------------------------

class EmailConfig(BaseModel):
    """Email drafting and sending configuration."""
    
    reply_policy: str = Field(
        default_factory=lambda: _env("REPLY_POLICY", "reply_all"),
        description="Default reply policy"
    )
    sender_locked_name: str = Field(
        default_factory=lambda: _env("SENDER_LOCKED_NAME", ""),
        description="Locked sender display name"
    )
    sender_locked_email: str = Field(
        default_factory=lambda: _env("SENDER_LOCKED_EMAIL", ""),
        description="Locked sender email address"
    )
    message_id_domain: str = Field(
        default_factory=lambda: _env("MESSAGE_ID_DOMAIN", ""),
        description="Domain for Message-ID generation"
    )
    sender_reply_to: str = Field(
        default_factory=lambda: _env("SENDER_REPLY_TO", ""),
        description="Reply-To address"
    )
    allowed_senders: Set[str] = Field(
        default_factory=lambda: {
            s.strip() for s in (_env("ALLOWED_SENDERS", "")).split(",") if s.strip()
        },
        description="Set of allowed sender addresses"
    )
    
    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# Summarizer Configuration
# -----------------------------------------------------------------------------

class SummarizerConfig(BaseModel):
    """Thread summarization configuration."""
    
    summarizer_version: str = Field(
        default="2.2-facts-ledger", description="Summarizer version"
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
        default=25.0, ge=1.0, le=100.0, description="Skip attachments over this size (MB)"
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
        default_factory=lambda: _env("LOG_LEVEL", "INFO"),
        description="Logging level"
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
    
    allow_parent_traversal: bool = Field(
        default=False, description="Allow .. in paths"
    )
    allow_provider_override: bool = Field(
        default=False, description="Allow provider override at runtime"
    )
    force_renorm: bool = Field(
        default=False, description="Force re-normalization of embeddings"
    )
    blocked_extensions: Set[str] = Field(
        default_factory=lambda: {
            s.strip().lower() for s in _env(
                "BLOCKED_EXTENSIONS",
                ".exe,.bat,.cmd,.scr,.vbs,.js,.jar,.msi,.dll"
            ).split(",") if s.strip()
        },
        description="Blocked file extensions"
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
    
    google_application_credentials: Optional[str] = Field(
        default_factory=lambda: _env("GOOGLE_APPLICATION_CREDENTIALS", None),
        description="Path to GCP credentials JSON"
    )
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: _env("OPENAI_API_KEY", None),
        description="OpenAI API key"
    )
    azure_openai_api_key: Optional[str] = Field(
        default_factory=lambda: _env("AZURE_OPENAI_API_KEY", None),
        description="Azure OpenAI API key"
    )
    azure_openai_endpoint: Optional[str] = Field(
        default_factory=lambda: _env("AZURE_OPENAI_ENDPOINT", None),
        description="Azure OpenAI endpoint"
    )
    azure_openai_deployment: Optional[str] = Field(
        default_factory=lambda: _env("AZURE_OPENAI_DEPLOYMENT", None),
        description="Azure OpenAI deployment name"
    )
    azure_openai_api_version: Optional[str] = Field(
        default_factory=lambda: _env("AZURE_OPENAI_API_VERSION", None),
        description="Azure OpenAI API version"
    )
    cohere_api_key: Optional[str] = Field(
        default_factory=lambda: _env("COHERE_API_KEY", None),
        description="Cohere API key"
    )
    hf_api_key: Optional[str] = Field(
        default_factory=lambda: _env("HF_API_KEY", None),
        description="HuggingFace API key"
    )
    huggingface_api_key: Optional[str] = Field(
        default_factory=lambda: _env("HUGGINGFACE_API_KEY", None),
        description="HuggingFace API key (alias)"
    )
    qwen_api_key: Optional[str] = Field(
        default_factory=lambda: _env("QWEN_API_KEY", None),
        description="Qwen API key"
    )
    qwen_base_url: Optional[str] = Field(
        default_factory=lambda: _env("QWEN_BASE_URL", None),
        description="Qwen API base URL"
    )
    qwen_timeout: Optional[int] = Field(
        default_factory=lambda: _env("QWEN_TIMEOUT", None, int),
        description="Qwen API timeout"
    )
    run_id: Optional[str] = Field(
        default_factory=lambda: _env("RUN_ID", None),
        description="Unique run identifier"
    )
    
    model_config = {"extra": "forbid"}


# -----------------------------------------------------------------------------
# File Patterns Configuration
# -----------------------------------------------------------------------------

class FilePatternsConfig(BaseModel):
    """File pattern matching configuration."""
    
    allowed_file_patterns: List[str] = Field(
        default_factory=lambda: [
            "*.pdf", "*.docx", "*.doc", "*.xlsx", "*.xls", "*.xlsm", "*.xlsb", "*.csv",
            "*.pptx", "*.ppt", "*.pptm", "*.txt", "*.md", "*.rtf", "*.png", "*.jpg",
            "*.jpeg", "*.gif", "*.tif", "*.tiff", "*.webp", "*.heic", "*.heif", "*.eml",
            "*.msg", "*.html", "*.htm", "*.json", "*.xml",
        ],
        description="Allowed file patterns for processing"
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
    last_to: str = Field(
        default="", description="Last To recipients"
    )
    last_cc: str = Field(
        default="", description="Last CC recipients"
    )
    last_subject: str = Field(
        default="", description="Last email subject"
    )
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