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
from typing import Any, Dict, List, cast

from cortex.common.exceptions import ConfigurationError
from dotenv import load_dotenv
from pydantic import BaseModel, Field

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

load_dotenv()


logger = logging.getLogger(__name__)


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
    def SECRET_KEY(self) -> str:
        """JWT secret key for token signing (dev fallback provided)."""
        return (
            os.environ.get("OUTLOOKCORTEX_SECRET_KEY")
            or "dev-secret-key-change-in-production"
        )

    @property
    def object_storage(self) -> StorageConfig:
        """Alias for storage config (S3/Spaces)."""
        return self.storage

    def save(self, path: Path) -> None:
        """Save the configuration to a file."""
        with path.open("w") as f:
            f.write(self.model_dump_json(indent=2))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization/GUI display."""
        return self.model_dump()

    def update_environment(self) -> None:
        """Update environment variables from config (for child processes)."""
        # GCP settings removed

        # Core settings
        if self.core.provider:
            os.environ["EMBED_PROVIDER"] = self.core.provider

        # Embedding settings
        os.environ["EMBED_MODEL"] = self.embedding.model_name
        os.environ["EMBED_DIM"] = str(self.embedding.output_dimensionality)
        os.environ["VERTEX_EMBED_MODEL"] = self.embedding.model_name
        # VERTEX_MODEL env var mapping removed

        # DigitalOcean LLM controls
        if self.digitalocean.scaling.token:
            os.environ["DIGITALOCEAN_TOKEN"] = self.digitalocean.scaling.token

        # DigitalOcean Endpoint (DOKS Inference)
        if self.digitalocean.endpoint.BASE_URL:
            os.environ["DO_LLM_BASE_URL"] = str(self.digitalocean.endpoint.BASE_URL)
        if self.digitalocean.endpoint.api_key:
            os.environ["DO_LLM_API_KEY"] = self.digitalocean.endpoint.api_key
        os.environ["DO_LLM_COMPLETION_MODEL"] = (
            self.digitalocean.endpoint.default_completion_model
        )
        os.environ["DO_LLM_EMBEDDING_MODEL"] = (
            self.digitalocean.endpoint.default_embedding_model
        )

        # Processing settings (apply environment overrides with type conversion)
        def _coerce_int(name: str, current: int) -> int:
            raw = os.getenv(name)
            if raw is None or raw == "":
                return current
            try:
                return int(raw)
            except (TypeError, ValueError):
                return current

        self.processing.batch_size = _coerce_int(
            "EMBED_BATCH", self.processing.batch_size
        )
        self.processing.num_workers = _coerce_int(
            "NUM_WORKERS", self.processing.num_workers
        )
        self.processing.chunk_size = _coerce_int(
            "CHUNK_SIZE", self.processing.chunk_size
        )
        self.processing.chunk_overlap = _coerce_int(
            "CHUNK_OVERLAP", self.processing.chunk_overlap
        )

        os.environ["EMBED_BATCH"] = str(self.processing.batch_size)
        os.environ["NUM_WORKERS"] = str(self.processing.num_workers)
        os.environ["CHUNK_SIZE"] = str(self.processing.chunk_size)
        os.environ["CHUNK_OVERLAP"] = str(self.processing.chunk_overlap)

        # Email settings
        os.environ["SENDER_LOCKED_NAME"] = self.email.sender_locked_name
        os.environ["SENDER_LOCKED_EMAIL"] = self.email.sender_locked_email

        # Directory settings
        os.environ["INDEX_DIRNAME"] = self.directories.index_dirname
        os.environ["CHUNK_DIRNAME"] = self.directories.chunk_dirname

        # Storage settings
        os.environ["S3_ENDPOINT"] = self.storage.endpoint_url
        os.environ["S3_BUCKET_RAW"] = self.storage.bucket_raw
        os.environ["S3_REGION"] = self.storage.region
        if self.storage.access_key:
            os.environ["S3_ACCESS_KEY"] = self.storage.access_key
        if self.storage.secret_key:
            os.environ["S3_SECRET_KEY"] = self.storage.secret_key

        # System settings
        os.environ["LOG_LEVEL"] = self.system.log_level
        os.environ["OUTLOOKCORTEX_DB_URL"] = self.database.url

        # Credential file discovery removed
        # (Vertex AI dependencies removed)

    @classmethod
    def load(cls, path: Path | None = None) -> EmailOpsConfig:
        """
        Load the configuration.

        If path is None, creates config from environment variables.
        If path is provided, loads from JSON file with env overrides.
        """
        if path is None:
            try:
                return cls()
            except Exception as e:
                raise ConfigurationError(
                    f"Configuration error: {e}\n\n"
                    "Please ensure all required environment variables are set in your .env file."
                ) from e

        if not path.exists():
            return cls()

        try:
            with path.open("r") as f:
                data = json.load(f)
            return cls.model_validate(data)
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
            default_config = cls()
            try:
                default_config.save(path)
                logger.info("Created new default configuration file at %s.", path)
            except OSError as e:
                logger.error(
                    "Failed to save new default configuration at %s: %s", path, e
                )
            return default_config

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
    warnings: List[str] = []
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
    connection.execute(
        text("SELECT set_config('app.current_tenant', :tid, false)"), {"tid": tenant_id}
    )


def validate_directories(config: EmailOpsConfig) -> list[str]:
    """
    Validate that configured directories exist.

    Returns list of warnings for missing directories.
    """
    warnings: List[str] = []

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
