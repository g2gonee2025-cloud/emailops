"""
Centralized configuration management for the EmailOps application.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Load environment variables from .env file
from dotenv import load_dotenv

from .core_config_models import (
    CoreConfig,
    DirectoryConfig,
    EmailConfig,
    EmbeddingConfig,
    FilePatternsConfig,
    GcpConfig,
    LimitsConfig,
    ProcessingConfig,
    RetryConfig,
    SearchConfig,
    SecurityConfig,
    SensitiveConfig,
    SummarizerConfig,
    SystemConfig,
    UnifiedConfig,
)

load_dotenv()

try:
    from google.oauth2 import service_account
except ImportError:
    service_account = None

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass
class EmailOpsConfig:
    """
    Centralized configuration for EmailOps.
    A unified configuration model incorporating all settings.
    """
    directories: DirectoryConfig = field(default_factory=DirectoryConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    gcp: GcpConfig = field(default_factory=GcpConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    sensitive: SensitiveConfig = field(default_factory=SensitiveConfig)
    file_patterns: FilePatternsConfig = field(default_factory=FilePatternsConfig)
    unified: UnifiedConfig = field(default_factory=UnifiedConfig)

    def save(self, path: Path) -> None:
        """Save the configuration to a file."""
        def convert_paths(obj):
            """Recursively convert Path objects to strings."""
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, set):
                return list(obj)  # Convert sets to lists for JSON
            else:
                return obj

        with path.open("w") as f:
            # Convert Path objects to strings for JSON serialization
            data = asdict(self)
            data = convert_paths(data)
            json.dump(data, f, indent=2, default=str)  # default=str as final fallback

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization/GUI display."""
        return asdict(self)

    def update_environment(self) -> None:
        """Update environment variables from config (for child processes). Merged from both sources."""
        # From unified_config.py
        if self.gcp.gcp_project:
            os.environ["GCP_PROJECT"] = self.gcp.gcp_project
            os.environ["GOOGLE_CLOUD_PROJECT"] = self.gcp.gcp_project
        if self.gcp.vertex_location:
            os.environ["VERTEX_LOCATION"] = self.gcp.vertex_location
        if self.core.provider:
            os.environ["EMBED_PROVIDER"] = self.core.provider
        os.environ["EMBED_BATCH"] = str(self.processing.batch_size)
        os.environ["NUM_WORKERS"] = str(self.processing.num_workers)
        os.environ["CHUNK_SIZE"] = str(self.processing.chunk_size)
        os.environ["CHUNK_OVERLAP"] = str(self.processing.chunk_overlap)
        os.environ["SENDER_LOCKED_NAME"] = self.email.sender_locked_name
        os.environ["SENDER_LOCKED_EMAIL"] = self.email.sender_locked_email
        os.environ["MESSAGE_ID_DOMAIN"] = self.email.message_id_domain

        # From original config.py
        os.environ["INDEX_DIRNAME"] = self.directories.index_dirname
        os.environ["CHUNK_DIRNAME"] = self.directories.chunk_dirname
        os.environ["VERTEX_EMBED_MODEL"] = self.embedding.vertex_embed_model
        os.environ["VERTEX_MODEL"] = self.embedding.vertex_model
        os.environ["GCP_REGION"] = self.gcp.gcp_region
        os.environ["VERTEX_LOCATION"] = self.gcp.vertex_location
        os.environ["LOG_LEVEL"] = self.system.log_level

        # Credentials and project derivation (from original)
        cred_file = self.get_credential_file()
        if cred_file:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_file)
            if not self.gcp.gcp_project:
                try:
                    with cred_file.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    proj = str(data.get("project_id") or "").strip()
                    if proj:
                        self.gcp.gcp_project = proj
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
            try:
                with path.open("r") as f:
                    data = json.load(f)
                # Override with environment variables (using EMAILOPS_ prefix)
                for key in list(data.keys()):
                    env_var = f"EMAILOPS_{key.upper()}"
                    if env_var in os.environ:
                        data[key] = os.environ[env_var]
                return cls(**data)
            except json.JSONDecodeError:
                logger.warning(
                    "Corrupt JSON file at %s. Renaming and recreating with defaults.", path
                )
                try:
                    # P0-9 FIX: Rename instead of delete to prevent data loss
                    import time
                    ts = int(time.time())
                    backup_path = path.with_suffix(f".corrupt.{ts}.json")
                    path.rename(backup_path)
                    logger.info("Backed up corrupt config to %s", backup_path)
                except OSError as e:
                    logger.error("Failed to rename corrupt config file at %s: %s", path, e)

                # Create a new default config, save it, and return it
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
        package_secrets = Path(__file__).parent.parent / self.directories.secrets_dir
        if package_secrets.exists():
            return package_secrets.resolve()
        return self.directories.secrets_dir.resolve()

    def discover_credential_files(self) -> list[Path]:
        """Dynamically discover all valid GCP service account JSON files in secrets directory."""
        secrets_dir = self.get_secrets_dir()
        if not secrets_dir.exists():
            raise ConfigurationError(f"Secrets directory not found: {secrets_dir}")
        json_files = list(secrets_dir.glob("*.json"))
        if not json_files:
            raise ConfigurationError(
                f"No JSON files found in secrets directory: {secrets_dir}"
            )
        valid_files = []
        for json_file in sorted(json_files):
            if self._is_valid_service_account_json(json_file):
                valid_files.append(json_file)
        if not valid_files:
            raise ConfigurationError(
                f"No valid GCP service account JSON files found in {secrets_dir}"
            )
        return valid_files

    @staticmethod
    def _is_valid_service_account_json(p: Path) -> bool:
        """Strictly validate that a JSON file looks like a GCP service-account key."""
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return False
            required = {
                "type",
                "project_id",
                "private_key_id",
                "private_key",
                "client_email",
            }
            if not required.issubset(data):
                return False
            if data.get("type") != "service_account":
                return False
            private_key = data.get("private_key", "").strip()
            if not private_key.startswith(
                "-----BEGIN PRIVATE KEY-----"
            ) or not private_key.endswith("-----END PRIVATE KEY-----"):
                return False
            key_id = data.get("private_key_id", "").strip()
            if not key_id or len(key_id) < 16:
                return False
            client_email = data.get("client_email", "").strip()
            if (
                not client_email
                or "@" not in client_email
                or not client_email.endswith(
                    (".iam.gserviceaccount.com", ".gserviceaccount.com")
                )
            ):
                return False
            project_id = data.get("project_id", "").strip()
            if not project_id or len(project_id) < 6:
                return False
            if service_account is not None:
                try:
                    credentials = service_account.Credentials.from_service_account_info(
                        data
                    )
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
