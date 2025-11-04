from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _env(key: str, default: Any, value_type: type = str) -> Any:
    import os
    env_key_new = f"EMAILOPS_{key}"
    value = os.getenv(env_key_new) or os.getenv(key)
    if value is None:
        return default
    try:
        if value_type is bool:
            return value.lower() in ("true", "1", "yes", "on")
        if value_type is int:
            return int(value)
        if value_type is float:
            return float(value)
        return value
    except (ValueError, TypeError):
        return default

@dataclass
class DirectoryConfig:
    index_dirname: str = field(default_factory=lambda: _env("INDEX_DIRNAME", "_index"))
    chunk_dirname: str = field(default_factory=lambda: _env("CHUNK_DIRNAME", "_chunks"))
    secrets_dir: Path = field(default_factory=lambda: Path(_env("SECRETS_DIR", "secrets")))

@dataclass
class CoreConfig:
    export_root: str = field(default_factory=lambda: _env("EXPORT_ROOT", ""))
    provider: str = field(default_factory=lambda: _env("EMBED_PROVIDER", "vertex"))
    persona: str = field(default_factory=lambda: _env("PERSONA", "expert insurance CSR"))

@dataclass
class ProcessingConfig:
    chunk_size: int = 1600
    chunk_overlap: int = 200
    batch_size: int = 64
    num_workers: int = 4

@dataclass
class EmbeddingConfig:
    vertex_embed_model: str = field(default_factory=lambda: _env("VERTEX_EMBED_MODEL", "gemini-embedding-001"))
    vertex_model: str = field(default_factory=lambda: _env("VERTEX_MODEL", "gemini-2.5-pro"))
    vertex_embed_dim: int | None = field(default_factory=lambda: _env("VERTEX_EMBED_DIM", None, int))
    vertex_output_dim: int | None = field(default_factory=lambda: _env("VERTEX_OUTPUT_DIM", None, int))
    openai_embed_model: str = field(default_factory=lambda: _env("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    cohere_embed_model: str = field(default_factory=lambda: _env("COHERE_EMBED_MODEL", "embed-english-v3.0"))
    cohere_input_type: str = field(default_factory=lambda: _env("COHERE_INPUT_TYPE", "search_document"))
    hf_embed_model: str = field(default_factory=lambda: _env("HF_EMBED_MODEL", "BAAI/bge-large-en-v1.5"))
    qwen_embed_model: str = field(default_factory=lambda: _env("QWEN_EMBED_MODEL", "Qwen/Qwen2-57B-A14B-Instruct"))
    local_embed_model: str = field(default_factory=lambda: _env("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

@dataclass
class GcpConfig:
    gcp_region: str = field(default_factory=lambda: _env("GCP_REGION", "global"))
    vertex_location: str = field(default_factory=lambda: _env("VERTEX_LOCATION", "us-central1"))
    gcp_project: str = field(default_factory=lambda: _env("GCP_PROJECT", ""))

@dataclass
class RetryConfig:
    max_retries: int = 4
    initial_backoff_seconds: float = 0.25
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 8.0
    jitter: bool = True
    rate_limit_per_sec: float = 0.0
    rate_limit_capacity: int = 0
    circuit_failure_threshold: int = 5
    circuit_reset_seconds: float = 30.0

@dataclass
class SearchConfig:
    half_life_days: int = field(default_factory=lambda: _env("HALF_LIFE_DAYS", 30, int))
    recency_boost_strength: float = field(default_factory=lambda: _env("RECENCY_BOOST_STRENGTH", 1.0, float))
    candidates_multiplier: int = field(default_factory=lambda: _env("CANDIDATES_MULTIPLIER", 3, int))
    sim_threshold: float = field(default_factory=lambda: _env("SIM_THRESHOLD_DEFAULT", 0.30, float))
    reply_tokens: int = field(default_factory=lambda: _env("REPLY_TOKENS_TARGET_DEFAULT", 20000, int))
    fresh_tokens: int = field(default_factory=lambda: _env("FRESH_TOKENS_TARGET_DEFAULT", 10000, int))
    context_snippet_chars: int = field(default_factory=lambda: _env("CONTEXT_SNIPPET_CHARS", 1600, int))
    chars_per_token: float = field(default_factory=lambda: _env("CHARS_PER_TOKEN", 3.8, float))
    boosted_score_cutoff: float = field(default_factory=lambda: _env("BOOSTED_SCORE_CUTOFF", 0.30, float))
    attach_max_mb: float = field(default_factory=lambda: _env("ATTACH_MAX_MB", 15.0, float))
    min_avg_score: float = field(default_factory=lambda: _env("MIN_AVG_SCORE", 0.2, float))
    rerank_alpha: float = field(default_factory=lambda: _env("RERANK_ALPHA", 0.35, float))
    mmr_lambda: float = field(default_factory=lambda: _env("MMR_LAMBDA", 0.70, float))
    mmr_k_cap: int = field(default_factory=lambda: _env("MMR_K_CAP", 250, int))
    k: int = 25

@dataclass
class EmailConfig:
    reply_policy: str = field(default_factory=lambda: _env("REPLY_POLICY", "reply_all"))
    sender_locked_name: str = field(default_factory=lambda: _env("SENDER_LOCKED_NAME", ""))
    sender_locked_email: str = field(default_factory=lambda: _env("SENDER_LOCKED_EMAIL", ""))
    message_id_domain: str = field(default_factory=lambda: _env("MESSAGE_ID_DOMAIN", ""))
    sender_reply_to: str = field(default_factory=lambda: _env("SENDER_REPLY_TO", ""))
    allowed_senders: set[str] = field(default_factory=lambda: {s.strip() for s in (_env("ALLOWED_SENDERS", "")).split(",") if s.strip()})

@dataclass
class SummarizerConfig:
    summarizer_version: str = field(default_factory=lambda: _env("SUMMARIZER_VERSION", "2.2-facts-ledger"))
    summarizer_thread_max_chars: int = field(default_factory=lambda: _env("SUMMARIZER_THREAD_MAX_CHARS", 16000, int))
    summarizer_critic_max_chars: int = field(default_factory=lambda: _env("SUMMARIZER_CRITIC_MAX_CHARS", 5000, int))
    summarizer_improve_max_chars: int = field(default_factory=lambda: _env("SUMMARIZER_IMPROVE_MAX_CHARS", 8000, int))
    summarizer_max_participants: int = field(default_factory=lambda: _env("SUMMARIZER_MAX_PARTICIPANTS", 25, int))
    summarizer_max_summary_points: int = field(default_factory=lambda: _env("SUMMARIZER_MAX_SUMMARY_POINTS", 25, int))
    summarizer_max_next_actions: int = field(default_factory=lambda: _env("SUMMARIZER_MAX_NEXT_ACTIONS", 50, int))
    summarizer_max_fact_items: int = field(default_factory=lambda: _env("SUMMARIZER_MAX_FACT_ITEMS", 50, int))
    summarizer_subject_max_len: int = field(default_factory=lambda: _env("SUMMARIZER_SUBJECT_MAX_LEN", 100, int))
    audit_target_min_score: int = field(default_factory=lambda: _env("AUDIT_TARGET_MIN_SCORE", 8, int))

@dataclass
class LimitsConfig:
    max_attachment_text_chars: int = field(default_factory=lambda: _env("MAX_ATTACHMENT_TEXT_CHARS", 100000, int))
    excel_max_cells: int = field(default_factory=lambda: _env("EXCEL_MAX_CELLS", 200000, int))
    skip_attachment_over_mb: float = field(default_factory=lambda: _env("SKIP_ATTACHMENT_OVER_MB", 20.0, float))
    max_total_attachments_mb: float = field(default_factory=lambda: _env("MAX_TOTAL_ATTACHMENTS_MB", 500.0, float))
    max_indexable_file_mb: float = field(default_factory=lambda: _env("MAX_INDEXABLE_FILE_MB", 50.0, float))
    max_indexable_chars: int = field(default_factory=lambda: _env("MAX_INDEXABLE_CHARS", 5000000, int))
    max_chat_snippets: int = field(default_factory=lambda: _env("MAX_CHAT_SNIPPETS", 50, int))
    max_chat_context_mb: float = field(default_factory=lambda: _env("MAX_CHAT_CONTEXT_MB", 10.0, float))

@dataclass
class SystemConfig:
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    command_timeout: int = field(default_factory=lambda: _env("COMMAND_TIMEOUT", 3600, int))
    active_window_seconds: int = field(default_factory=lambda: _env("ACTIVE_WINDOW_SECONDS", 180, int))
    file_encoding_cache_size: int = field(default_factory=lambda: _env("FILE_ENCODING_CACHE_SIZE", 128, int))
    pip_timeout: int = field(default_factory=lambda: _env("PIP_TIMEOUT", 300, int))

@dataclass
class SecurityConfig:
    allow_parent_traversal: bool = field(default_factory=lambda: _env("ALLOW_PARENT_TRAVERSAL", False, bool))
    allow_provider_override: bool = field(default_factory=lambda: _env("ALLOW_PROVIDER_OVERRIDE", False, bool))
    force_renorm: bool = field(default_factory=lambda: _env("FORCE_RENORM", False, bool))
    blocked_extensions: set[str] = field(default_factory=lambda: {
        s.strip().lower() for s in _env(
            "BLOCKED_EXTENSIONS",
            ".exe,.bat,.cmd,.scr,.vbs,.js,.jar,.msi,.dll"
        ).split(",") if s.strip()
    })

@dataclass
class SensitiveConfig:
    google_application_credentials: str | None = field(default_factory=lambda: _env("GOOGLE_APPLICATION_CREDENTIALS", None))
    openai_api_key: str | None = field(default_factory=lambda: _env("OPENAI_API_KEY", None))
    azure_openai_api_key: str | None = field(default_factory=lambda: _env("AZURE_OPENAI_API_KEY", None))
    azure_openai_endpoint: str | None = field(default_factory=lambda: _env("AZURE_OPENAI_ENDPOINT", None))
    azure_openai_deployment: str | None = field(default_factory=lambda: _env("AZURE_OPENAI_DEPLOYMENT", None))
    azure_openai_api_version: str | None = field(default_factory=lambda: _env("AZURE_OPENAI_API_VERSION", None))
    cohere_api_key: str | None = field(default_factory=lambda: _env("COHERE_API_KEY", None))
    hf_api_key: str | None = field(default_factory=lambda: _env("HF_API_KEY", None))
    huggingface_api_key: str | None = field(default_factory=lambda: _env("HUGGINGFACE_API_KEY", None))
    qwen_api_key: str | None = field(default_factory=lambda: _env("QWEN_API_KEY", None))
    qwen_base_url: str | None = field(default_factory=lambda: _env("QWEN_BASE_URL", None))
    qwen_timeout: int | None = field(default_factory=lambda: _env("QWEN_TIMEOUT", None, int))
    run_id: str | None = field(default_factory=lambda: _env("RUN_ID", None))

@dataclass
class FilePatternsConfig:
    allowed_file_patterns: list[str] = field(default_factory=lambda: [
        "*.pdf", "*.docx", "*.doc", "*.xlsx", "*.xls", "*.xlsm", "*.xlsb", "*.csv",
        "*.pptx", "*.ppt", "*.pptm", "*.txt", "*.md", "*.rtf", "*.png", "*.jpg",
        "*.jpeg", "*.gif", "*.tif", "*.tiff", "*.webp", "*.heic", "*.heif", "*.eml",
        "*.msg", "*.html", "*.htm", "*.json", "*.xml",
    ])

@dataclass
class UnifiedConfig:
    temperature: float = 0.2
    chat_session_id: str = "default"
    max_chat_history: int = 5
    last_to: str = ""
    last_cc: str = ""
    last_subject: str = ""
    timeout_seconds: float = 30.0
    cache_ttl_seconds: float = 0.0
    cache_max_entries: int = 1024
