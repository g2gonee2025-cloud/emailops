"""
Configuration loader for EmailOps.

Implements §2.3 of the Canonical Blueprint.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from cortex.common.exceptions import ConfigurationError
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, ValidationError

from .models import (
    CoreConfig,
    DatabaseConfig,
    DigitalOceanLLMConfig,
    DirectoryConfig,
    EmailConfig,
    EmbeddingConfig,
    FilePatternsConfig,
    LimitsConfig,
    PiiConfig,
    ProcessingConfig,
    QdrantConfig,
    RedisConfig,
    RetryConfig,
    SearchConfig,
    SecurityConfig,
    SensitiveConfig,
    StorageConfig,
    SummarizerConfig,
    SystemConfig,
    UnifiedConfig,
)

logger = logging.getLogger(__name__)
_dotenv_loaded = False


def _ensure_dotenv_loaded() -> None:
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    try:
        load_dotenv()
    except OSError as exc:
        logger.warning("Failed to load .env file: %s", exc)
    _dotenv_loaded = True


_SENSITIVE_KEY_MARKERS = (
    "key",
    "secret",
    "token",
    "password",
    "credential",
    "auth",
    "jwt",
    "private",
    "cert",
    "api",
)


def _redact_value(value: Any) -> Any:
    if value is None or value == "":
        return value
    return "***REDACTED***"


def _redact_dict(data: Any) -> Any:
    if isinstance(data, dict):
        redacted: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(key, str) and any(
                marker in key.lower() for marker in _SENSITIVE_KEY_MARKERS
            ):
                redacted[key] = _redact_value(value)
            else:
                redacted[key] = _redact_dict(value)
        return redacted
    if isinstance(data, list):
        return [_redact_dict(item) for item in data]
    return data


class EmailOpsConfig(BaseModel):
    """
    Centralized configuration for EmailOps.

    A unified configuration model incorporating all settings per Blueprint §2.3.
    All sub-configs are Pydantic models with validation.
    """

    directories: DirectoryConfig = Field(default_factory=DirectoryConfig)
    core: CoreConfig = Field(default_factory=CoreConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    digitalocean: DigitalOceanLLMConfig = Field(default_factory=DigitalOceanLLMConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    summarizer: SummarizerConfig = Field(default_factory=SummarizerConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    sensitive: SensitiveConfig = Field(default_factory=SensitiveConfig)
    file_patterns: FilePatternsConfig = Field(default_factory=FilePatternsConfig)
    unified: UnifiedConfig = Field(default_factory=UnifiedConfig)
    pii: PiiConfig = Field(default_factory=PiiConfig)

    model_config = {"extra": "forbid"}

    @property
    def secret_key(self) -> str | None:
        """
        JWT secret key for token signing.
        
        Checks environment variables in order of preference:
        1. OUTLOOKCORTEX_SECRET_KEY (canonical name)
        2. SECRET_KEY (fallback for compatibility)
        
        P0-1 FIX (S1845): Removed case-insensitive name clash (SECRET_KEY property).
        This property now serves as the single source of truth for JWT secrets.
        """
        return os.environ.get("OUTLOOKCORTEX_SECRET_KEY") or os.environ.get(
            "SECRET_KEY"
        )

    @property
    def object_storage(self) -> StorageConfig:
        """Alias for storage config (S3/Spaces)."""
        return self.storage

    def save(self, path: Path) -> None:
        """Save the configuration to a file."""
        with path.open("w") as f:
            f.write(self.model_dump_json(indent=2))

    def to_dict(self, redact: bool = True) -> dict[str, Any]:
        """Convert to dictionary for serialization/GUI display."""
        data = self.model_dump()
        if redact:
            result = _redact_dict(data)
            assert isinstance(result, dict)  # model_dump always returns dict
            return result
        return data

    def update_environment(self, include_secrets: bool = False) -> None:
        """Update environment variables from config (for child processes)."""

        # include_secrets=True will export credential fields.
        def _set_env(name: str, value: Any) -> None:
            if value is None:
                return
            if isinstance(value, SecretStr):
                value = value.get_secret_value()
                if value is None:
                    return
            os.environ[name] = str(value)

        # GCP settings removed

        # Core settings
        _set_env("EMBED_PROVIDER", self.core.provider)

        # Embedding settings
        _set_env("EMBED_MODEL", self.embedding.model_name)
        _set_env("EMBED_DIM", self.embedding.output_dimensionality)
        _set_env("VERTEX_EMBED_MODEL", self.embedding.model_name)
        # VERTEX_MODEL env var mapping removed

        # DigitalOcean LLM controls
        if include_secrets:
            _set_env("DIGITALOCEAN_TOKEN", self.digitalocean.scaling.token)

        # DigitalOcean Endpoint (DOKS Inference)
        _set_env("DO_LLM_BASE_URL", self.digitalocean.endpoint.BASE_URL)
        if include_secrets:
            _set_env("DO_LLM_API_KEY", self.digitalocean.endpoint.api_key)
        _set_env(
            "DO_LLM_COMPLETION_MODEL",
            self.digitalocean.endpoint.default_completion_model,
        )
        _set_env(
            "DO_LLM_EMBEDDING_MODEL",
            self.digitalocean.endpoint.default_embedding_model,
        )

        # Processing settings
        _set_env("EMBED_BATCH", self.processing.batch_size)
        _set_env("NUM_WORKERS", self.processing.num_workers)
        _set_env("CHUNK_SIZE", self.processing.chunk_size)
        _set_env("CHUNK_OVERLAP", self.processing.chunk_overlap)

        # Email settings
        _set_env("SENDER_LOCKED_NAME", self.email.sender_locked_name)
        _set_env("SENDER_LOCKED_EMAIL", self.email.sender_locked_email)

        # Directory settings
        _set_env("INDEX_DIRNAME", self.directories.index_dirname)
        _set_env("CHUNK_DIRNAME", self.directories.chunk_dirname)

        # Storage settings
        _set_env("S3_ENDPOINT", self.storage.endpoint_url)
        _set_env("S3_BUCKET_RAW", self.storage.bucket_raw)
        _set_env("S3_REGION", self.storage.region)
        if include_secrets:
            _set_env("S3_ACCESS_KEY", self.storage.access_key)
            _set_env("S3_SECRET_KEY", self.storage.secret_key)

        # System settings
        _set_env("LOG_LEVEL", self.system.log_level)
        _set_env("OUTLOOKCORTEX_DB_URL", self.database.url)

        # Credential file discovery removed
        # (Vertex AI dependencies removed)

    @classmethod
    def load(cls, path: Path | None = None) -> EmailOpsConfig:
        """
        Load the configuration.

        If path is None, creates config from environment variables.
        If path is provided, loads from JSON file (missing fields use env defaults).
        """
        _ensure_dotenv_loaded()

        def _build_default() -> EmailOpsConfig:
            try:
                return cls()
            except Exception as e:
                raise ConfigurationError(
                    f"Configuration error: {e}\n\n"
                    "Please ensure all required environment variables are set in your .env file."
                ) from e

        if path is None:
            return _build_default()

        if not path.exists():
            return _build_default()

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            try:
                return cls.model_validate(data)
            except ValidationError as e:
                logger.warning(
                    "Config validation failed for %s: %s. Falling back to defaults.",
                    path,
                    e,
                )
                return _build_default()
        except json.JSONDecodeError:
            logger.warning(
                "Corrupt JSON file at %s. Renaming and recreating with defaults.", path
            )
            try:
                import time

                ts = int(time.time())
                backup_path = path.with_suffix(f".corrupt.{ts}.json")
                path.rename(backup_path)
                logger.info("Backed up corrupt config to %s", backup_path)
            except OSError as e:
                logger.error("Failed to rename corrupt config file at %s: %s", path, e)

            # Create new default config
            default_config = _build_default()
            try:
                default_config.save(path)
                logger.info("Created new default configuration file at %s.", path)
            except OSError as e:
                logger.error(
                    "Failed to save new default configuration at %s: %s", path, e
                )
            return default_config
        except OSError as e:
            logger.error("Failed to read config file at %s: %s", path, e)
            return _build_default()

    def get_secrets_dir(self) -> Path:
        """Get the secrets directory path, resolving relative paths."""
        if self.directories.secrets_dir.is_absolute():
            return self.directories.secrets_dir

        cwd_secrets = Path.cwd() / self.directories.secrets_dir
        if cwd_secrets.exists():
            return cwd_secrets.resolve()

        package_secrets = (
            Path(__file__).parent.parent.parent.parent / self.directories.secrets_dir
        )
        if package_secrets.exists():
            return package_secrets.resolve()

        return self.directories.secrets_dir.resolve()

    # Credential/Service Account logic removed


_config: EmailOpsConfig | None = None
_config_lock = threading.RLock()


def get_config() -> EmailOpsConfig:
    """
    Get the global configuration instance (thread-safe singleton pattern).

    P0-2 FIX: Uses double-checked locking to prevent race conditions
    during initialization while maintaining performance for subsequent calls.
    """
    global _config

    # Fast path - no lock needed if already initialized
    if _config is not None:
        return _config

    # Slow path - acquire lock for initialization
    with _config_lock:
        # Double-check after acquiring lock (another thread may have initialized)
        if _config is None:
            _config = EmailOpsConfig.load()
        return _config


def get_default_config() -> EmailOpsConfig:
    """Get a new EmailOpsConfig instance with default values."""
    return EmailOpsConfig()


def reset_config() -> None:
    """
    Reset the global configuration instance (mainly for testing).

    P0-2 FIX: Thread-safe reset operation.
    """
    global _config
    with _config_lock:
        _config = None


def set_config(config: EmailOpsConfig) -> None:
    """
    Set the global configuration instance (mainly for testing).

    Thread-safe operation.
    """
    global _config
    with _config_lock:
        _config = config


def validate_env_prefix() -> list[str]:
    """
    Validate that all EMAILOPS_ prefixed env vars use OUTLOOKCORTEX_ instead.

    Returns list of warnings for deprecated env var usage.

    Blueprint §3.3:
    - Prefix all env vars with OUTLOOKCORTEX_
    """
    warnings: list[str] = []
    deprecated_prefix = "EMAILOPS_"
    canonical_prefix = "OUTLOOKCORTEX_"

    for key in os.environ:
        if key.startswith(deprecated_prefix):
            canonical_key = key.replace(deprecated_prefix, canonical_prefix, 1)
            warnings.append(
                f"Deprecated env var '{key}' found. Use '{canonical_key}' instead."
            )

    return warnings


def set_rls_tenant(connection: Any, tenant_id: str) -> None:
    """
    Set the Row-Level Security tenant context on a database connection.

    Blueprint §11.1:
    - Postgres RLS enforces tenant isolation
    - SET app.current_tenant = :tid

    Args:
        connection: SQLAlchemy connection or session
        tenant_id: The tenant ID to set for RLS
    """
    if not tenant_id:
        raise ConfigurationError(
            "tenant_id is required for RLS",
            error_code="RLS_TENANT_REQUIRED",
        )

    # Sanitize tenant_id to prevent injection

    if not re.match(r"^[a-zA-Z0-9_-]+$", tenant_id):
        raise ConfigurationError(
            f"Invalid tenant_id format: {tenant_id}",
            error_code="RLS_TENANT_INVALID",
        )

    # Execute SET command safely
    from sqlalchemy import text

    # Use parameters to prevent injection (even with regex check)
    try:
        connection.execute(
            text("SELECT set_config('app.current_tenant', :tid, false)"),
            {"tid": tenant_id},
        )
    except Exception as exc:
        raise ConfigurationError(
            "Failed to set RLS tenant",
            error_code="RLS_SET_FAILED",
        ) from exc


def validate_directories(config: EmailOpsConfig) -> list[str]:
    """
    Validate that configured directories exist.

    Returns list of warnings for missing directories.
    """
    warnings: list[str] = []

    # Check secrets_dir
    secrets_dir = config.get_secrets_dir()
    if not secrets_dir.exists():
        warnings.append(f"Secrets directory does not exist: {secrets_dir}")

    # Check export_root if configured
    if config.directories.export_root:
        export_root = Path(config.directories.export_root)
        if not export_root.exists():
            warnings.append(f"Export root directory does not exist: {export_root}")

    return warnings
