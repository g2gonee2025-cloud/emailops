from __future__ import annotations

"""
Centralized configuration for EmailOps.
Manages all configuration values with strict validation.

Configuration Philosophy:
- Non-sensitive configs: Defined in this file, NO fallbacks (fail fast if missing)
- Sensitive configs: Only from .env file (API keys, credentials)
- Credential files: Dynamically discovered from secrets directory
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""
    pass


def _require_env(key: str, value_type: type = str) -> Any:
    """
    Get required environment variable without fallback.
    Raises ConfigurationError if not set.

    Args:
        key: Environment variable name
        value_type: Type to convert to (str, int, float, bool)

    Returns:
        Value converted to specified type

    Raises:
        ConfigurationError: If environment variable not set or conversion fails
    """
    value = os.getenv(key)
    if value is None:
        raise ConfigurationError(
            f"Required environment variable '{key}' is not set. "
            f"Please set it in your .env file or environment."
        )

    try:
        if value_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif value_type == int:
            return int(value)
        elif value_type == float:
            return float(value)
        else:
            return value
    except (ValueError, AttributeError) as e:
        raise ConfigurationError(
            f"Environment variable '{key}' has invalid value '{value}' "
            f"(expected {value_type.__name__}): {e}"
        ) from e


def _optional_env(key: str, value_type: type = str) -> Any | None:
    """Get optional environment variable (returns None if not set)."""
    value = os.getenv(key)
    if value is None:
        return None

    try:
        if value_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif value_type == int:
            return int(value)
        elif value_type == float:
            return float(value)
        else:
            return value
    except (ValueError, AttributeError):
        return None


@dataclass
class EmailOpsConfig:
    """
    Centralized configuration for EmailOps.

    All non-sensitive configurations are required and have NO fallback values.
    Sensitive configurations (API keys, credentials) are loaded from .env only.
    """

    # ============================================================================
    # DIRECTORY CONFIGURATION (Required - No Fallbacks)
    # ============================================================================
    INDEX_DIRNAME: str = field(default_factory=lambda: _require_env("INDEX_DIRNAME"))
    CHUNK_DIRNAME: str = field(default_factory=lambda: _require_env("CHUNK_DIRNAME"))
    SECRETS_DIR: Path = field(default_factory=lambda: Path(_require_env("SECRETS_DIR")))

    # ============================================================================
    # PROCESSING CONFIGURATION (Required - No Fallbacks)
    # ============================================================================
    CHUNK_SIZE: int = field(default_factory=lambda: _require_env("CHUNK_SIZE", int))
    CHUNK_OVERLAP: int = field(default_factory=lambda: _require_env("CHUNK_OVERLAP", int))
    EMBED_BATCH: int = field(default_factory=lambda: _require_env("EMBED_BATCH", int))
    NUM_WORKERS: int = field(default_factory=lambda: _require_env("NUM_WORKERS", int))

    # ============================================================================
    # EMBEDDING PROVIDER CONFIGURATION (Required - No Fallbacks)
    # ============================================================================
    EMBED_PROVIDER: str = field(default_factory=lambda: _require_env("EMBED_PROVIDER"))
    VERTEX_EMBED_MODEL: str = field(default_factory=lambda: _require_env("VERTEX_EMBED_MODEL"))
    VERTEX_MODEL: str = field(default_factory=lambda: _require_env("VERTEX_MODEL"))
    VERTEX_EMBED_DIM: int | None = field(default_factory=lambda: _optional_env("VERTEX_EMBED_DIM", int))
    VERTEX_OUTPUT_DIM: int | None = field(default_factory=lambda: _optional_env("VERTEX_OUTPUT_DIM", int))

    # ============================================================================
    # GCP CONFIGURATION (Required - No Fallbacks)
    # ============================================================================
    GCP_REGION: str = field(default_factory=lambda: _require_env("GCP_REGION"))
    VERTEX_LOCATION: str = field(default_factory=lambda: _require_env("VERTEX_LOCATION"))

    # ============================================================================
    # RETRY & RATE LIMITING (Required - No Fallbacks)
    # ============================================================================
    VERTEX_MAX_RETRIES: int = field(default_factory=lambda: _require_env("VERTEX_MAX_RETRIES", int))
    VERTEX_BACKOFF_INITIAL: float = field(default_factory=lambda: _require_env("VERTEX_BACKOFF_INITIAL", float))
    VERTEX_BACKOFF_MAX: float = field(default_factory=lambda: _require_env("VERTEX_BACKOFF_MAX", float))
    API_RATE_LIMIT: int = field(default_factory=lambda: _require_env("API_RATE_LIMIT", int))

    # ============================================================================
    # SEARCH & DRAFT CONFIGURATION (Required - No Fallbacks)
    # ============================================================================
    HALF_LIFE_DAYS: int = field(default_factory=lambda: _require_env("HALF_LIFE_DAYS", int))
    RECENCY_BOOST_STRENGTH: float = field(default_factory=lambda: _require_env("RECENCY_BOOST_STRENGTH", float))
    CANDIDATES_MULTIPLIER: int = field(default_factory=lambda: _require_env("CANDIDATES_MULTIPLIER", int))
    SIM_THRESHOLD_DEFAULT: float = field(default_factory=lambda: _require_env("SIM_THRESHOLD_DEFAULT", float))
    REPLY_TOKENS_TARGET_DEFAULT: int = field(default_factory=lambda: _require_env("REPLY_TOKENS_TARGET_DEFAULT", int))
    FRESH_TOKENS_TARGET_DEFAULT: int = field(default_factory=lambda: _require_env("FRESH_TOKENS_TARGET_DEFAULT", int))
    CONTEXT_SNIPPET_CHARS: int = field(default_factory=lambda: _require_env("CONTEXT_SNIPPET_CHARS", int))
    CHARS_PER_TOKEN: float = field(default_factory=lambda: _require_env("CHARS_PER_TOKEN", float))
    BOOSTED_SCORE_CUTOFF: float = field(default_factory=lambda: _require_env("BOOSTED_SCORE_CUTOFF", float))
    ATTACH_MAX_MB: float = field(default_factory=lambda: _require_env("ATTACH_MAX_MB", float))
    MIN_AVG_SCORE: float = field(default_factory=lambda: _require_env("MIN_AVG_SCORE", float))
    RERANK_ALPHA: float = field(default_factory=lambda: _require_env("RERANK_ALPHA", float))
    MMR_LAMBDA: float = field(default_factory=lambda: _require_env("MMR_LAMBDA", float))
    MMR_K_CAP: int = field(default_factory=lambda: _require_env("MMR_K_CAP", int))

    # ============================================================================
    # EMAIL CONFIGURATION (Required - No Fallbacks)
    # ============================================================================
    REPLY_POLICY: str = field(default_factory=lambda: _require_env("REPLY_POLICY"))
    PERSONA: str = field(default_factory=lambda: _require_env("PERSONA"))
    SENDER_LOCKED_NAME: str = field(default_factory=lambda: _require_env("SENDER_LOCKED_NAME"))
    SENDER_LOCKED_EMAIL: str = field(default_factory=lambda: _require_env("SENDER_LOCKED_EMAIL"))
    MESSAGE_ID_DOMAIN: str = field(default_factory=lambda: _require_env("MESSAGE_ID_DOMAIN"))
    SENDER_REPLY_TO: str = field(default_factory=lambda: _optional_env("SENDER_REPLY_TO") or "")
    ALLOWED_SENDERS: set[str] = field(default_factory=lambda: {
        s.strip() for s in (_optional_env("ALLOWED_SENDERS") or "").split(",") if s.strip()
    })

    # ============================================================================
    # SUMMARIZER CONFIGURATION (Required - No Fallbacks)
    # ============================================================================
    SUMMARIZER_VERSION: str = field(default_factory=lambda: _require_env("SUMMARIZER_VERSION"))
    SUMMARIZER_THREAD_MAX_CHARS: int = field(default_factory=lambda: _require_env("SUMMARIZER_THREAD_MAX_CHARS", int))
    SUMMARIZER_CRITIC_MAX_CHARS: int = field(default_factory=lambda: _require_env("SUMMARIZER_CRITIC_MAX_CHARS", int))
    SUMMARIZER_IMPROVE_MAX_CHARS: int = field(default_factory=lambda: _require_env("SUMMARIZER_IMPROVE_MAX_CHARS", int))
    SUMMARIZER_MAX_PARTICIPANTS: int = field(default_factory=lambda: _require_env("SUMMARIZER_MAX_PARTICIPANTS", int))
    SUMMARIZER_MAX_SUMMARY_POINTS: int = field(default_factory=lambda: _require_env("SUMMARIZER_MAX_SUMMARY_POINTS", int))
    SUMMARIZER_MAX_NEXT_ACTIONS: int = field(default_factory=lambda: _require_env("SUMMARIZER_MAX_NEXT_ACTIONS", int))
    SUMMARIZER_MAX_FACT_ITEMS: int = field(default_factory=lambda: _require_env("SUMMARIZER_MAX_FACT_ITEMS", int))
    SUMMARIZER_SUBJECT_MAX_LEN: int = field(default_factory=lambda: _require_env("SUMMARIZER_SUBJECT_MAX_LEN", int))
    AUDIT_TARGET_MIN_SCORE: int = field(default_factory=lambda: _require_env("AUDIT_TARGET_MIN_SCORE", int))

    # ============================================================================
    # PROCESSING LIMITS (Required - No Fallbacks)
    # ============================================================================
    MAX_ATTACHMENT_TEXT_CHARS: int = field(default_factory=lambda: _require_env("MAX_ATTACHMENT_TEXT_CHARS", int))
    EXCEL_MAX_CELLS: int = field(default_factory=lambda: _require_env("EXCEL_MAX_CELLS", int))
    SKIP_ATTACHMENT_OVER_MB: float = field(default_factory=lambda: _require_env("SKIP_ATTACHMENT_OVER_MB", float))
    MAX_INDEXABLE_FILE_MB: float = field(default_factory=lambda: _require_env("MAX_INDEXABLE_FILE_MB", float))
    MAX_INDEXABLE_CHARS: int = field(default_factory=lambda: _require_env("MAX_INDEXABLE_CHARS", int))
    MAX_CHAT_SNIPPETS: int = field(default_factory=lambda: _require_env("MAX_CHAT_SNIPPETS", int))
    MAX_CHAT_CONTEXT_MB: float = field(default_factory=lambda: _require_env("MAX_CHAT_CONTEXT_MB", float))

    # ============================================================================
    # SYSTEM CONFIGURATION (Required - No Fallbacks)
    # ============================================================================
    LOG_LEVEL: str = field(default_factory=lambda: _require_env("LOG_LEVEL"))
    COMMAND_TIMEOUT: int = field(default_factory=lambda: _require_env("COMMAND_TIMEOUT", int))
    ACTIVE_WINDOW_SECONDS: int = field(default_factory=lambda: _require_env("ACTIVE_WINDOW_SECONDS", int))
    FILE_ENCODING_CACHE_SIZE: int = field(default_factory=lambda: _require_env("FILE_ENCODING_CACHE_SIZE", int))
    PIP_TIMEOUT: int = field(default_factory=lambda: _require_env("PIP_TIMEOUT", int))

    # ============================================================================
    # SECURITY SETTINGS (Required - No Fallbacks)
    # ============================================================================
    ALLOW_PARENT_TRAVERSAL: bool = field(default_factory=lambda: _require_env("ALLOW_PARENT_TRAVERSAL", bool))
    ALLOW_PROVIDER_OVERRIDE: bool = field(default_factory=lambda: _require_env("ALLOW_PROVIDER_OVERRIDE", bool))
    FORCE_RENORM: bool = field(default_factory=lambda: _require_env("FORCE_RENORM", bool))

    # ============================================================================
    # SENSITIVE CONFIGURATION (Loaded from .env only - Optional)
    # These are the ONLY configs that can be None
    # ============================================================================
    GCP_PROJECT: str | None = field(default_factory=lambda: _optional_env("GCP_PROJECT"))
    GOOGLE_APPLICATION_CREDENTIALS: str | None = field(default_factory=lambda: _optional_env("GOOGLE_APPLICATION_CREDENTIALS"))

    # API Keys (all optional - only needed if using specific providers)
    OPENAI_API_KEY: str | None = field(default_factory=lambda: _optional_env("OPENAI_API_KEY"))
    AZURE_OPENAI_API_KEY: str | None = field(default_factory=lambda: _optional_env("AZURE_OPENAI_API_KEY"))
    AZURE_OPENAI_ENDPOINT: str | None = field(default_factory=lambda: _optional_env("AZURE_OPENAI_ENDPOINT"))
    AZURE_OPENAI_DEPLOYMENT: str | None = field(default_factory=lambda: _optional_env("AZURE_OPENAI_DEPLOYMENT"))
    AZURE_OPENAI_API_VERSION: str | None = field(default_factory=lambda: _optional_env("AZURE_OPENAI_API_VERSION"))
    COHERE_API_KEY: str | None = field(default_factory=lambda: _optional_env("COHERE_API_KEY"))
    HF_API_KEY: str | None = field(default_factory=lambda: _optional_env("HF_API_KEY"))
    HUGGINGFACE_API_KEY: str | None = field(default_factory=lambda: _optional_env("HUGGINGFACE_API_KEY"))
    QWEN_API_KEY: str | None = field(default_factory=lambda: _optional_env("QWEN_API_KEY"))
    QWEN_BASE_URL: str | None = field(default_factory=lambda: _optional_env("QWEN_BASE_URL"))

    # Model-specific optional configs
    OPENAI_EMBED_MODEL: str | None = field(default_factory=lambda: _optional_env("OPENAI_EMBED_MODEL"))
    COHERE_EMBED_MODEL: str | None = field(default_factory=lambda: _optional_env("COHERE_EMBED_MODEL"))
    COHERE_INPUT_TYPE: str | None = field(default_factory=lambda: _optional_env("COHERE_INPUT_TYPE"))
    HF_EMBED_MODEL: str | None = field(default_factory=lambda: _optional_env("HF_EMBED_MODEL"))
    QWEN_EMBED_MODEL: str | None = field(default_factory=lambda: _optional_env("QWEN_EMBED_MODEL"))
    QWEN_TIMEOUT: int | None = field(default_factory=lambda: _optional_env("QWEN_TIMEOUT", int))
    LOCAL_EMBED_MODEL: str | None = field(default_factory=lambda: _optional_env("LOCAL_EMBED_MODEL"))

    # Optional runtime identifiers
    RUN_ID: str | None = field(default_factory=lambda: _optional_env("RUN_ID"))

    # ============================================================================
    # FILE PATTERNS (Static Configuration)
    # ============================================================================
    ALLOWED_FILE_PATTERNS: list[str] = field(
        default_factory=lambda: ["*.txt", "*.pdf", "*.docx", "*.doc", "*.xlsx", "*.xls", "*.md", "*.csv"]
    )

    @classmethod
    def load(cls) -> EmailOpsConfig:
        """
        Load configuration from environment.

        Returns:
            Configured EmailOpsConfig instance

        Raises:
            ConfigurationError: If any required configuration is missing
        """
        try:
            return cls()
        except ConfigurationError as e:
            raise ConfigurationError(
                f"Configuration error: {e}\n\n"
                "Please ensure all required environment variables are set in your .env file."
            ) from e

    def get_secrets_dir(self) -> Path:
        """
        Get the secrets directory path, resolving relative paths.

        Returns:
            Absolute path to secrets directory
        """
        if self.SECRETS_DIR.is_absolute():
            return self.SECRETS_DIR

        # Try relative to current working directory
        cwd_secrets = Path.cwd() / self.SECRETS_DIR
        if cwd_secrets.exists():
            return cwd_secrets.resolve()

        # Try relative to this file's parent (emailops package root or repo root)
        package_secrets = Path(__file__).parent.parent / self.SECRETS_DIR
        if package_secrets.exists():
            return package_secrets.resolve()

        # Return default resolution (even if not present)
        return self.SECRETS_DIR.resolve()

    def discover_credential_files(self) -> list[Path]:
        """
        Dynamically discover all valid GCP service account JSON files in secrets directory.

        Returns:
            List of paths to valid service account JSON files, sorted by name

        Raises:
            ConfigurationError: If secrets directory doesn't exist or no valid files found
        """
        secrets_dir = self.get_secrets_dir()

        if not secrets_dir.exists():
            raise ConfigurationError(
                f"Secrets directory not found: {secrets_dir}\n"
                f"Please create it and add your GCP service account JSON files."
            )

        # Find all JSON files
        json_files = list(secrets_dir.glob("*.json"))

        if not json_files:
            raise ConfigurationError(
                f"No JSON files found in secrets directory: {secrets_dir}\n"
                f"Please add your GCP service account JSON files."
            )

        # Validate each file
        valid_files = []
        for json_file in sorted(json_files):
            if self._is_valid_service_account_json(json_file):
                valid_files.append(json_file)

        if not valid_files:
            raise ConfigurationError(
                f"No valid GCP service account JSON files found in {secrets_dir}\n"
                f"Found {len(json_files)} JSON file(s) but none are valid service accounts.\n"
                f"Please ensure your service account files have the correct format."
            )

        return valid_files

    @staticmethod
    def _is_valid_service_account_json(p: Path) -> bool:
        """
        Strictly validate that a JSON file looks like a GCP service-account key.
        Enhanced validation including key format, expiration, and basic token validity
        """
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return False
            required = {"type", "project_id", "private_key_id", "private_key", "client_email"}
            if not required.issubset(set(data.keys())):
                return False
            if str(data.get("type", "")).strip() != "service_account":
                return False

            # Enhanced key validation
            private_key = str(data.get("private_key", "")).strip()
            if not private_key.startswith("-----BEGIN PRIVATE KEY-----"):
                return False
            if not private_key.endswith("-----END PRIVATE KEY-----"):
                return False

            # Validate key ID format (should be hex string)
            key_id = str(data.get("private_key_id", "")).strip()
            if not key_id or len(key_id) < 16:  # GCP key IDs are typically 40+ chars
                return False

            # Validate email format
            client_email = str(data.get("client_email", "")).strip()
            if not client_email or "@" not in client_email:
                return False
            if not client_email.endswith((".iam.gserviceaccount.com", ".gserviceaccount.com")):
                return False

            # Validate project ID format
            project_id = str(data.get("project_id", "")).strip()
            if not project_id or len(project_id) < 6:  # GCP project IDs are typically longer
                return False

            # Basic token validity check (if google-auth is available)
            try:
                from google.auth.exceptions import MalformedError
                from google.oauth2 import service_account

                # Try to create credentials object (validates private key format)
                credentials = service_account.Credentials.from_service_account_info(data)

                # Check if credentials are expired (if they have expiry info)
                if hasattr(credentials, 'expired') and credentials.expired:
                    return False

                return True
            except ImportError:
                # If google-auth not available, fall back to basic checks
                return True
            except Exception:
                # MalformedError or other auth errors - treat as invalid
                return False

        except Exception:
            return False

    def get_credential_file(self) -> Path | None:
        """
        Find a valid credential file.
        First checks GOOGLE_APPLICATION_CREDENTIALS env var, then discovers from secrets dir.

        Returns:
            Path to credential file if found and validated, None otherwise
        """
        # 1) Check if explicitly specified in environment
        if self.GOOGLE_APPLICATION_CREDENTIALS:
            creds_path = Path(self.GOOGLE_APPLICATION_CREDENTIALS)
            if creds_path.exists() and self._is_valid_service_account_json(creds_path):
                return creds_path

        # 2) Discover from secrets directory
        try:
            valid_files = self.discover_credential_files()
            if valid_files:
                # Return first valid file
                return valid_files[0]
        except ConfigurationError:
            pass

        return None

    def get_all_credential_files(self) -> list[Path]:
        """
        Get all valid credential files for multi-account rotation.

        Returns:
            List of all valid credential file paths

        Raises:
            ConfigurationError: If no valid credential files found
        """
        try:
            return self.discover_credential_files()
        except ConfigurationError as e:
            raise ConfigurationError(
                f"Failed to discover credential files: {e}\n"
                "Multi-account rotation requires at least one valid service account file."
            ) from e

    def update_environment(self) -> None:
        """
        Update os.environ with configuration values.
        Ensures child processes inherit correct settings.
        Also derives project from the selected service-account key if env doesn't already provide one.
        """
        # Core knobs
        os.environ["INDEX_DIRNAME"] = self.INDEX_DIRNAME
        os.environ["CHUNK_DIRNAME"] = self.CHUNK_DIRNAME
        os.environ["CHUNK_SIZE"] = str(self.CHUNK_SIZE)
        os.environ["CHUNK_OVERLAP"] = str(self.CHUNK_OVERLAP)
        os.environ["EMBED_BATCH"] = str(self.EMBED_BATCH)
        os.environ["EMBED_PROVIDER"] = self.EMBED_PROVIDER
        os.environ["VERTEX_EMBED_MODEL"] = self.VERTEX_EMBED_MODEL
        os.environ["VERTEX_MODEL"] = self.VERTEX_MODEL
        os.environ["GCP_REGION"] = self.GCP_REGION
        os.environ["VERTEX_LOCATION"] = self.VERTEX_LOCATION
        os.environ["LOG_LEVEL"] = self.LOG_LEVEL

        # Project
        if self.GCP_PROJECT:
            os.environ["GCP_PROJECT"] = self.GCP_PROJECT
            os.environ["GOOGLE_CLOUD_PROJECT"] = self.GCP_PROJECT
            os.environ["VERTEX_PROJECT"] = self.GCP_PROJECT

        # Credentials (strict)
        cred_file = self.get_credential_file()
        if cred_file:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_file)

            # If project isn't already set, derive it from the service-account JSON
            if not self.GCP_PROJECT:
                try:
                    with cred_file.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    proj = str(data.get("project_id") or "").strip()
                    if proj:
                        os.environ["GCP_PROJECT"] = proj
                        os.environ["GOOGLE_CLOUD_PROJECT"] = proj
                        os.environ["VERTEX_PROJECT"] = proj
                except Exception:
                    # Silent fallback; Vertex init will still try ADC if needed.
                    pass

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "index_dirname": self.INDEX_DIRNAME,
            "chunk_dirname": self.CHUNK_DIRNAME,
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP,
            "embed_batch": self.EMBED_BATCH,
            "num_workers": self.NUM_WORKERS,
            "embed_provider": self.EMBED_PROVIDER,
            "vertex_embed_model": self.VERTEX_EMBED_MODEL,
            "vertex_model": self.VERTEX_MODEL,
            "gcp_project": self.GCP_PROJECT,
            "gcp_region": self.GCP_REGION,
            "vertex_location": self.VERTEX_LOCATION,
            "secrets_dir": str(self.SECRETS_DIR),
            "log_level": self.LOG_LEVEL,
            "active_window_seconds": self.ACTIVE_WINDOW_SECONDS,
            "command_timeout": self.COMMAND_TIMEOUT,
            "sender_locked_name": self.SENDER_LOCKED_NAME,
            "sender_locked_email": self.SENDER_LOCKED_EMAIL,
            "message_id_domain": self.MESSAGE_ID_DOMAIN,
            "reply_policy": self.REPLY_POLICY,
            "persona": self.PERSONA,
        }


# Global configuration instance
_config: EmailOpsConfig | None = None


def get_config() -> EmailOpsConfig:
    """
    Get the global configuration instance (singleton pattern).

    Returns:
        Global EmailOpsConfig instance

    Raises:
        ConfigurationError: If configuration cannot be loaded
    """
    global _config
    if _config is None:
        _config = EmailOpsConfig.load()
    return _config


def reset_config() -> None:
    """Reset the global configuration instance (mainly for testing)."""
    global _config
    _config = None
