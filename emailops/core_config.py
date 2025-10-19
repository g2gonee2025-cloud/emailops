from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from google.auth.exceptions import MalformedError
    from google.oauth2 import service_account
except ImportError:
    MalformedError = None  # type: ignore
    service_account = None  # type: ignore

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""
    pass

def _require_env(key: str, value_type: type = str) -> Any:
    env_key_new = f"EMAILOPS_{key}"
    value = os.getenv(env_key_new) or os.getenv(key)
    if value is None:
        raise ConfigurationError(f"Required environment variable '{key}' or '{env_key_new}' is not set.")
    try:
        if value_type is bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif value_type is int:
            return int(value)
        elif value_type is float:
            return float(value)
        else:
            return value
    except ValueError as e:
        raise ConfigurationError(f"Invalid value for '{key}' or '{env_key_new}': {e}") from e

def _optional_env(key: str, value_type: type = str) -> Any | None:
    env_key_new = f"EMAILOPS_{key}"
    value = os.getenv(env_key_new) or os.getenv(key)
    if value is None:
        return None
    try:
        if value_type is bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif value_type is int:
            return int(value)
        elif value_type is float:
            return float(value)
        else:
            return value
    except ValueError:
        return None

@dataclass
class EmailOpsConfig:
    """
    Centralized configuration for EmailOps.
    A unified configuration model incorporating all settings.
    """

    # ============================================================================
    # DIRECTORY CONFIGURATION (from original config.py)
    # ============================================================================
    index_dirname: str = field(default_factory=lambda: _require_env("INDEX_DIRNAME"))
    chunk_dirname: str = field(default_factory=lambda: _require_env("CHUNK_DIRNAME"))
    secrets_dir: Path = field(default_factory=lambda: Path(_require_env("SECRETS_DIR")))

    # ============================================================================
    # CORE (from unified_config.py)
    # ============================================================================
    export_root: str = ""
    provider: str = "vertex"  # Overlaps with EMBED_PROVIDER
    persona: str = os.getenv("PERSONA", "expert insurance CSR")  # Overlaps with PERSONA

    # ============================================================================
    # PROCESSING CONFIGURATION (merged)
    # ============================================================================
    chunk_size: int = 1600  # Overlaps with CHUNK_SIZE
    chunk_overlap: int = 200  # Overlaps with CHUNK_OVERLAP
    batch_size: int = 64  # Overlaps with EMBED_BATCH
    num_workers: int = 4  # Overlaps with NUM_WORKERS

    # ============================================================================
    # EMBEDDING PROVIDER CONFIGURATION (merged)
    # ============================================================================
    vertex_embed_model: str = "gemini-embedding-001"  # Overlaps with VERTEX_EMBED_MODEL
    vertex_model: str = field(default_factory=lambda: _require_env("VERTEX_MODEL"))
    vertex_embed_dim: int | None = field(default_factory=lambda: _optional_env("VERTEX_EMBED_DIM", int))
    vertex_output_dim: int | None = field(default_factory=lambda: _optional_env("VERTEX_OUTPUT_DIM", int))

    # ============================================================================
    # GCP CONFIGURATION (merged)
    # ============================================================================
    gcp_region: str = "global"  # Overlaps with GCP_REGION
    vertex_location: str = "us-central1"  # Overlaps with VERTEX_LOCATION
    gcp_project: str = field(default_factory=lambda: _optional_env("GCP_PROJECT") or "")

    # ============================================================================
    # RETRY & RATE LIMITING (merged)
    # ============================================================================
    max_retries: int = 4  # Overlaps with VERTEX_MAX_RETRIES
    initial_backoff_seconds: float = 0.25  # Overlaps with VERTEX_BACKOFF_INITIAL
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 8.0  # Overlaps with VERTEX_BACKOFF_MAX
    jitter: bool = True
    rate_limit_per_sec: float = 0.0
    rate_limit_capacity: int = 0  # Possibly overlaps with API_RATE_LIMIT
    circuit_failure_threshold: int = 5
    circuit_reset_seconds: float = 30.0

    # ============================================================================
    # SEARCH & DRAFT CONFIGURATION (merged)
    # ============================================================================
    half_life_days: int = 30  # Overlaps with HALF_LIFE_DAYS
    recency_boost_strength: float = field(default_factory=lambda: _require_env("RECENCY_BOOST_STRENGTH", float))
    candidates_multiplier: int = field(default_factory=lambda: _require_env("CANDIDATES_MULTIPLIER", int))
    sim_threshold: float = 0.30  # Overlaps with SIM_THRESHOLD_DEFAULT
    reply_tokens: int = 20000  # Overlaps with REPLY_TOKENS_TARGET_DEFAULT
    fresh_tokens: int = 10000  # Overlaps with FRESH_TOKENS_TARGET_DEFAULT
    context_snippet_chars: int = field(default_factory=lambda: _require_env("CONTEXT_SNIPPET_CHARS", int))
    chars_per_token: float = field(default_factory=lambda: _require_env("CHARS_PER_TOKEN", float))
    boosted_score_cutoff: float = field(default_factory=lambda: _require_env("BOOSTED_SCORE_CUTOFF", float))
    attach_max_mb: float = field(default_factory=lambda: _require_env("ATTACH_MAX_MB", float))
    min_avg_score: float = field(default_factory=lambda: _require_env("MIN_AVG_SCORE", float))
    rerank_alpha: float = 0.35  # Overlaps with RERANK_ALPHA
    mmr_lambda: float = 0.70  # Overlaps with MMR_LAMBDA
    mmr_k_cap: int = field(default_factory=lambda: _require_env("MMR_K_CAP", int))
    k: int = 25  # Possibly overlaps with MMR_K_CAP or CANDIDATES_MULTIPLIER

    # ============================================================================
    # EMAIL CONFIGURATION (merged)
    # ============================================================================
    reply_policy: str = "reply_all"  # Overlaps with REPLY_POLICY
    sender_locked_name: str = ""  # Overlaps with SENDER_LOCKED_NAME
    sender_locked_email: str = ""  # Overlaps with SENDER_LOCKED_EMAIL
    message_id_domain: str = ""  # Overlaps with MESSAGE_ID_DOMAIN
    sender_reply_to: str = field(default_factory=lambda: _optional_env("SENDER_REPLY_TO") or "")
    allowed_senders: set[str] = field(default_factory=lambda: {
        s.strip() for s in (_optional_env("ALLOWED_SENDERS") or "").split(",") if s.strip()
    })

    # ============================================================================
    # SUMMARIZER CONFIGURATION (from original config.py)
    # ============================================================================
    summarizer_version: str = field(default_factory=lambda: _require_env("SUMMARIZER_VERSION"))
    summarizer_thread_max_chars: int = field(default_factory=lambda: _require_env("SUMMARIZER_THREAD_MAX_CHARS", int))
    summarizer_critic_max_chars: int = field(default_factory=lambda: _require_env("SUMMARIZER_CRITIC_MAX_CHARS", int))
    summarizer_improve_max_chars: int = field(default_factory=lambda: _require_env("SUMMARIZER_IMPROVE_MAX_CHARS", int))
    summarizer_max_participants: int = field(default_factory=lambda: _require_env("SUMMARIZER_MAX_PARTICIPANTS", int))
    summarizer_max_summary_points: int = field(default_factory=lambda: _require_env("SUMMARIZER_MAX_SUMMARY_POINTS", int))
    summarizer_max_next_actions: int = field(default_factory=lambda: _require_env("SUMMARIZER_MAX_NEXT_ACTIONS", int))
    summarizer_max_fact_items: int = field(default_factory=lambda: _require_env("SUMMARIZER_MAX_FACT_ITEMS", int))
    summarizer_subject_max_len: int = field(default_factory=lambda: _require_env("SUMMARIZER_SUBJECT_MAX_LEN", int))
    audit_target_min_score: int = field(default_factory=lambda: _require_env("AUDIT_TARGET_MIN_SCORE", int))

    # ============================================================================
    # PROCESSING LIMITS (from original config.py)
    # ============================================================================
    max_attachment_text_chars: int = field(default_factory=lambda: _require_env("MAX_ATTACHMENT_TEXT_CHARS", int))
    excel_max_cells: int = field(default_factory=lambda: _require_env("EXCEL_MAX_CELLS", int))
    skip_attachment_over_mb: float = field(default_factory=lambda: _require_env("SKIP_ATTACHMENT_OVER_MB", float))
    max_indexable_file_mb: float = field(default_factory=lambda: _require_env("MAX_INDEXABLE_FILE_MB", float))
    max_indexable_chars: int = field(default_factory=lambda: _require_env("MAX_INDEXABLE_CHARS", int))
    max_chat_snippets: int = field(default_factory=lambda: _require_env("MAX_CHAT_SNIPPETS", int))
    max_chat_context_mb: float = field(default_factory=lambda: _require_env("MAX_CHAT_CONTEXT_MB", float))

    # ============================================================================
    # SYSTEM CONFIGURATION (from original config.py)
    # ============================================================================
    log_level: str = field(default_factory=lambda: _require_env("LOG_LEVEL"))
    command_timeout: int = field(default_factory=lambda: _require_env("COMMAND_TIMEOUT", int))
    active_window_seconds: int = field(default_factory=lambda: _require_env("ACTIVE_WINDOW_SECONDS", int))
    file_encoding_cache_size: int = field(default_factory=lambda: _require_env("FILE_ENCODING_CACHE_SIZE", int))
    pip_timeout: int = field(default_factory=lambda: _require_env("PIP_TIMEOUT", int))

    # ============================================================================
    # SECURITY SETTINGS (from original config.py)
    # ============================================================================
    allow_parent_traversal: bool = field(default_factory=lambda: _require_env("ALLOW_PARENT_TRAVERSAL", bool))
    allow_provider_override: bool = field(default_factory=lambda: _require_env("ALLOW_PROVIDER_OVERRIDE", bool))
    force_renorm: bool = field(default_factory=lambda: _require_env("FORCE_RENORM", bool))

    # ============================================================================
    # SENSITIVE CONFIGURATION (from original config.py)
    # ============================================================================
    google_application_credentials: str | None = field(default_factory=lambda: _optional_env("GOOGLE_APPLICATION_CREDENTIALS"))
    openai_api_key: str | None = field(default_factory=lambda: _optional_env("OPENAI_API_KEY"))
    azure_openai_api_key: str | None = field(default_factory=lambda: _optional_env("AZURE_OPENAI_API_KEY"))
    azure_openai_endpoint: str | None = field(default_factory=lambda: _optional_env("AZURE_OPENAI_ENDPOINT"))
    azure_openai_deployment: str | None = field(default_factory=lambda: _optional_env("AZURE_OPENAI_DEPLOYMENT"))
    azure_openai_api_version: str | None = field(default_factory=lambda: _optional_env("AZURE_OPENAI_API_VERSION"))
    cohere_api_key: str | None = field(default_factory=lambda: _optional_env("COHERE_API_KEY"))
    hf_api_key: str | None = field(default_factory=lambda: _optional_env("HF_API_KEY"))
    huggingface_api_key: str | None = field(default_factory=lambda: _optional_env("HUGGINGFACE_API_KEY"))
    qwen_api_key: str | None = field(default_factory=lambda: _optional_env("QWEN_API_KEY"))
    qwen_base_url: str | None = field(default_factory=lambda: _optional_env("QWEN_BASE_URL"))

    # Model-specific optional configs (from original config.py)
    openai_embed_model: str | None = field(default_factory=lambda: _optional_env("OPENAI_EMBED_MODEL"))
    cohere_embed_model: str | None = field(default_factory=lambda: _optional_env("COHERE_EMBED_MODEL"))
    cohere_input_type: str | None = field(default_factory=lambda: _optional_env("COHERE_INPUT_TYPE"))
    hf_embed_model: str | None = field(default_factory=lambda: _optional_env("HF_EMBED_MODEL"))
    qwen_embed_model: str | None = field(default_factory=lambda: _optional_env("QWEN_EMBED_MODEL"))
    qwen_timeout: int | None = field(default_factory=lambda: _optional_env("QWEN_TIMEOUT", int))
    local_embed_model: str | None = field(default_factory=lambda: _optional_env("LOCAL_EMBED_MODEL"))

    # Optional runtime identifiers (from original config.py)
    run_id: str | None = field(default_factory=lambda: _optional_env("RUN_ID"))

    # ============================================================================
    # FILE PATTERNS (merged)
    # ============================================================================
    allowed_file_patterns: list[str] = field(default_factory=lambda: [
        "*.pdf", "*.docx", "*.doc", "*.xlsx", "*.xls", "*.xlsm", "*.xlsb", "*.csv",
        "*.pptx", "*.ppt", "*.pptm", "*.txt", "*.md", "*.rtf",
        "*.png", "*.jpg", "*.jpeg", "*.gif", "*.tif", "*.tiff", "*.webp", "*.heic", "*.heif",
        "*.eml", "*.msg", "*.html", "*.htm", "*.json", "*.xml"
    ])  # Overlaps with ALLOWED_FILE_PATTERNS, using expanded list

    # ============================================================================
    # ADDITIONAL FROM UNIFIED (no overlaps)
    # ============================================================================
    temperature: float = 0.2
    chat_session_id: str = "default"
    max_chat_history: int = 5
    last_to: str = ""
    last_cc: str = ""
    last_subject: str = ""
    timeout_seconds: float = 30.0
    cache_ttl_seconds: float = 0.0
    cache_max_entries: int = 1024

    def save(self, path: Path) -> None:
        """Save the configuration to a file."""
        with path.open("w") as f:
            json.dump(asdict(self), f, indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization/GUI display."""
        return asdict(self)

    def update_environment(self) -> None:
        """Update environment variables from config (for child processes). Merged from both sources."""
        # From unified_config.py
        if self.gcp_project:
            os.environ["GCP_PROJECT"] = self.gcp_project
            os.environ["GOOGLE_CLOUD_PROJECT"] = self.gcp_project
        if self.vertex_location:
            os.environ["VERTEX_LOCATION"] = self.vertex_location
        if self.provider:
            os.environ["EMBED_PROVIDER"] = self.provider
        os.environ["EMBED_BATCH"] = str(self.batch_size)
        os.environ["NUM_WORKERS"] = str(self.num_workers)
        os.environ["CHUNK_SIZE"] = str(self.chunk_size)
        os.environ["CHUNK_OVERLAP"] = str(self.chunk_overlap)
        os.environ["SENDER_LOCKED_NAME"] = self.sender_locked_name
        os.environ["SENDER_LOCKED_EMAIL"] = self.sender_locked_email
        os.environ["MESSAGE_ID_DOMAIN"] = self.message_id_domain

        # From original config.py
        os.environ["INDEX_DIRNAME"] = self.index_dirname
        os.environ["CHUNK_DIRNAME"] = self.chunk_dirname
        os.environ["VERTEX_EMBED_MODEL"] = self.vertex_embed_model
        os.environ["VERTEX_MODEL"] = self.vertex_model
        os.environ["GCP_REGION"] = self.gcp_region
        os.environ["VERTEX_LOCATION"] = self.vertex_location
        os.environ["LOG_LEVEL"] = self.log_level

        # Credentials and project derivation (from original)
        cred_file = self.get_credential_file()
        if cred_file:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_file)
            if not self.gcp_project:
                try:
                    with cred_file.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    proj = str(data.get("project_id") or "").strip()
                    if proj:
                        self.gcp_project = proj
                        os.environ["GCP_PROJECT"] = proj
                        os.environ["GOOGLE_CLOUD_PROJECT"] = proj
                        os.environ["VERTEX_PROJECT"] = proj
                except Exception:
                    pass

    @classmethod
    def load(cls, path: Path | None = None) -> EmailOpsConfig:
        """Load the configuration. Merged logic: env-only if no path, JSON+env if path provided."""
        if path is None:
            # Original env-loading logic
            try:
                return cls()
            except ConfigurationError as e:
                raise ConfigurationError(
                    f"Configuration error: {e}\n\n"
                    "Please ensure all required environment variables are set in your .env file."
                ) from e
        else:
            # Unified JSON-loading logic
            if not path.exists():
                return cls()
            with path.open("r") as f:
                data = json.load(f)
            # Override with environment variables (using EMAILOPS_ prefix)
            for key in list(data.keys()):
                env_var = f"EMAILOPS_{key.upper()}"
                if env_var in os.environ:
                    data[key] = os.environ[env_var]
            return cls(**data)

    def get_secrets_dir(self) -> Path:
        """Get the secrets directory path, resolving relative paths."""
        if self.secrets_dir.is_absolute():
            return self.secrets_dir
        cwd_secrets = Path.cwd() / self.secrets_dir
        if cwd_secrets.exists():
            return cwd_secrets.resolve()
        package_secrets = Path(__file__).parent.parent / self.secrets_dir
        if package_secrets.exists():
            return package_secrets.resolve()
        return self.secrets_dir.resolve()

    def discover_credential_files(self) -> list[Path]:
        """Dynamically discover all valid GCP service account JSON files in secrets directory."""
        secrets_dir = self.get_secrets_dir()
        if not secrets_dir.exists():
            raise ConfigurationError(f"Secrets directory not found: {secrets_dir}")
        json_files = list(secrets_dir.glob("*.json"))
        if not json_files:
            raise ConfigurationError(f"No JSON files found in secrets directory: {secrets_dir}")
        valid_files = []
        for json_file in sorted(json_files):
            if self._is_valid_service_account_json(json_file):
                valid_files.append(json_file)
        if not valid_files:
            raise ConfigurationError(f"No valid GCP service account JSON files found in {secrets_dir}")
        return valid_files

    @staticmethod
    def _is_valid_service_account_json(p: Path) -> bool:
        """Strictly validate that a JSON file looks like a GCP service-account key."""
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return False
            required = {"type", "project_id", "private_key_id", "private_key", "client_email"}
            if not required.issubset(data):
                return False
            if data.get("type") != "service_account":
                return False
            private_key = data.get("private_key", "").strip()
            if not private_key.startswith("-----BEGIN PRIVATE KEY-----") or not private_key.endswith("-----END PRIVATE KEY-----"):
                return False
            key_id = data.get("private_key_id", "").strip()
            if not key_id or len(key_id) < 16:
                return False
            client_email = data.get("client_email", "").strip()
            if not client_email or "@" not in client_email or not client_email.endswith((".iam.gserviceaccount.com", ".gserviceaccount.com")):
                return False
            project_id = data.get("project_id", "").strip()
            if not project_id or len(project_id) < 6:
                return False
            if service_account is not None:
                try:
                    credentials = service_account.Credentials.from_service_account_info(data)
                    return not (hasattr(credentials, 'expired') and credentials.expired)
                except Exception as e:
                    logger.warning("Credential validation failed: %s", e)
                    return False
            return True
        except Exception:
            return False

    def get_credential_file(self) -> Path | None:
        """Find a valid credential file."""
        if self.google_application_credentials:
            creds_path = Path(self.google_application_credentials)
            if creds_path.exists() and self._is_valid_service_account_json(creds_path):
                return creds_path
        try:
            valid_files = self.discover_credential_files()
            if valid_files:
                return valid_files[0]
        except ConfigurationError:
            pass
        return None

    def get_all_credential_files(self) -> list[Path]:
        """Get all valid credential files for multi-account rotation."""
        try:
            return self.discover_credential_files()
        except ConfigurationError as e:
            raise ConfigurationError(f"Failed to discover credential files: {e}") from e

# Global configuration instance
_config: EmailOpsConfig | None = None

def get_config() -> EmailOpsConfig:
    """Get the global configuration instance (singleton pattern)."""
    global _config
    if _config is None:
        _config = EmailOpsConfig.load()
    return _config

def reset_config() -> None:
    """Reset the global configuration instance (mainly for testing)."""
    global _config
    _config = None
