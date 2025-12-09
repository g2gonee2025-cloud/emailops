"""
Configuration loader for EmailOps.

Implements §2.3 of the Canonical Blueprint.
"""
from __future__ import annotations

import json
import logging
import os
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
    GcpConfig,
    LimitsConfig,
    PiiConfig,
    ProcessingConfig,
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

try:
    from google.oauth2 import service_account
except ImportError:
    service_account = None

service_account = cast(Any, service_account)

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
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    gcp: GcpConfig = Field(default_factory=GcpConfig)
    digitalocean: DigitalOceanLLMConfig = Field(default_factory=DigitalOceanLLMConfig)
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
    def s3(self) -> StorageConfig:
        """Alias for storage config (S3/Spaces)."""
        return self.storage

    def save(self, path: Path) -> None:
        """Save the configuration to a file."""

        def convert_for_json(obj: Any) -> Any:
            """Recursively convert objects for JSON serialization."""
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, set):
                obj_set = cast(set[Any], obj)
                return [convert_for_json(item) for item in sorted(obj_set)]
            elif isinstance(obj, dict):
                obj_dict = cast(Dict[Any, Any], obj)
                return {str(k): convert_for_json(v) for k, v in obj_dict.items()}
            elif isinstance(obj, list):
                obj_list = cast(List[Any], obj)
                return [convert_for_json(item) for item in obj_list]
            else:
                return obj

        with path.open("w") as f:
            # Use model_dump for Pydantic models
            data = self.model_dump()
            data = convert_for_json(data)
            json.dump(data, f, indent=2, default=str)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization/GUI display."""
        return self.model_dump()

    def update_environment(self) -> None:
        """Update environment variables from config (for child processes)."""
        # GCP settings
        if self.gcp.gcp_project:
            os.environ["GCP_PROJECT"] = self.gcp.gcp_project
            os.environ["GOOGLE_CLOUD_PROJECT"] = self.gcp.gcp_project
        if self.gcp.vertex_location:
            os.environ["VERTEX_LOCATION"] = self.gcp.vertex_location

        # Core settings
        if self.core.provider:
            os.environ["EMBED_PROVIDER"] = self.core.provider

        # Embedding settings
        os.environ["EMBED_MODEL"] = self.embedding.model_name
        os.environ["EMBED_DIM"] = str(self.embedding.output_dimensionality)
        os.environ["VERTEX_EMBED_MODEL"] = self.embedding.model_name
        os.environ["VERTEX_MODEL"] = self.embedding.vertex_model

        # DigitalOcean LLM controls
        if self.digitalocean.scaling.token:
            os.environ["DIGITALOCEAN_TOKEN"] = self.digitalocean.scaling.token

        # Processing settings
        os.environ["EMBED_BATCH"] = str(self.processing.batch_size)
        os.environ["NUM_WORKERS"] = str(self.processing.num_workers)
        os.environ["CHUNK_SIZE"] = str(self.processing.chunk_size)
        os.environ["CHUNK_OVERLAP"] = str(self.processing.chunk_overlap)

        # Email settings
        os.environ["SENDER_LOCKED_NAME"] = self.email.sender_locked_name
        os.environ["SENDER_LOCKED_EMAIL"] = self.email.sender_locked_email
        os.environ["MESSAGE_ID_DOMAIN"] = self.email.message_id_domain

        # Directory settings
        os.environ["INDEX_DIRNAME"] = self.directories.index_dirname
        os.environ["CHUNK_DIRNAME"] = self.directories.chunk_dirname

        # GCP region settings
        os.environ["GCP_REGION"] = self.gcp.gcp_region

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

        # Credential file discovery
        cred_file = self.get_credential_file()
        if cred_file:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_file)
            if not self.gcp.gcp_project:
                try:
                    with cred_file.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    proj = str(data.get("project_id") or "").strip()
                    if proj:
                        # Note: Can't directly modify Pydantic model after creation
                        # Set environment variables instead
                        os.environ["GCP_PROJECT"] = proj
                        os.environ["GOOGLE_CLOUD_PROJECT"] = proj
                        os.environ["VERTEX_PROJECT"] = proj
                except Exception:
                    pass

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

            # Override with environment variables (canonical OUTLOOKCORTEX_ prefix)
            for key in list(data.keys()):
                canonical_env = f"OUTLOOKCORTEX_{key.upper()}"
                legacy_env = f"EMAILOPS_{key.upper()}"

                if canonical_env in os.environ:
                    data[key] = os.environ[canonical_env]
                elif legacy_env in os.environ:
                    logger.warning(
                        "Deprecated env var '%s' found. Use '%s' instead.",
                        legacy_env,
                        canonical_env,
                    )
                    data[key] = os.environ[legacy_env]

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
            except Exception as e:
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

    def discover_credential_files(self) -> list[Path]:
        """Dynamically discover all valid GCP service account JSON files in secrets directory."""
        secrets_dir = self.get_secrets_dir()
        if not secrets_dir.exists():
            return []

        json_files = list(secrets_dir.glob("*.json"))
        if not json_files:
            return []

        valid_files: List[Path] = []
        for json_file in sorted(json_files):
            if self._is_valid_service_account_json(json_file):
                valid_files.append(json_file)
        return valid_files

    @staticmethod
    def _is_valid_service_account_json(p: Path) -> bool:
        """Strictly validate that a JSON file looks like a GCP service-account key."""
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return False

            data_dict = cast(Dict[str, Any], data)

            required = {
                "type",
                "project_id",
                "private_key_id",
                "private_key",
                "client_email",
            }
            if not required.issubset(set(data_dict)):
                return False

            if data_dict.get("type") != "service_account":
                return False

            private_key: str = str(data_dict.get("private_key", "")).strip()
            if not private_key.startswith(
                "-----BEGIN PRIVATE KEY-----"
            ) or not private_key.endswith("-----END PRIVATE KEY-----"):
                return False

            key_id: str = str(data_dict.get("private_key_id", "")).strip()
            if not key_id or len(key_id) < 16:
                return False

            client_email: str = str(data_dict.get("client_email", "")).strip()
            if (
                not client_email
                or "@" not in client_email
                or not client_email.endswith(
                    (".iam.gserviceaccount.com", ".gserviceaccount.com")
                )
            ):
                return False

            project_id: str = str(data_dict.get("project_id", "")).strip()
            if not project_id or len(project_id) < 6:
                return False

            if service_account is not None:
                try:
                    credentials = cast(
                        Any, service_account
                    ).Credentials.from_service_account_info(data_dict)
                    return not (hasattr(credentials, "expired") and credentials.expired)
                except Exception as e:
                    logger.warning("Credential validation failed: %s", e)
                    return False

            return True
        except Exception:
            return False

    def get_credential_file(self) -> Path | None:
        """Find a valid credential file."""
        if self.sensitive.google_application_credentials:
            creds_path = Path(self.sensitive.google_application_credentials)
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
    import re

    if not re.match(r"^[a-zA-Z0-9_-]+$", tenant_id):
        raise ConfigurationError(
            f"Invalid tenant_id format: {tenant_id}",
            error_code="RLS_TENANT_INVALID",
        )

    # Execute SET command
    connection.execute(f"SET app.current_tenant = '{tenant_id}'")


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
