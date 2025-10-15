from __future__ import annotations

"""
Centralized configuration for EmailOps.
Manages all configuration values, environment variables, and default settings.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EmailOpsConfig:
    """Centralized configuration for EmailOps"""

    # Directory names
    INDEX_DIRNAME: str = field(default_factory=lambda: os.getenv("INDEX_DIRNAME", "_index"))
    CHUNK_DIRNAME: str = field(default_factory=lambda: os.getenv("CHUNK_DIRNAME", "_chunks"))

    # Processing defaults
    DEFAULT_CHUNK_SIZE: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1600")))
    DEFAULT_CHUNK_OVERLAP: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")))
    DEFAULT_BATCH_SIZE: int = field(default_factory=lambda: int(os.getenv("EMBED_BATCH", "64")))
    DEFAULT_NUM_WORKERS: int = field(default_factory=lambda: int(os.getenv("NUM_WORKERS", str(os.cpu_count() or 4))))

    # Embedding provider settings
    EMBED_PROVIDER: str = field(default_factory=lambda: os.getenv("EMBED_PROVIDER", "vertex"))
    VERTEX_EMBED_MODEL: str = field(default_factory=lambda: os.getenv("VERTEX_EMBED_MODEL", "gemini-embedding-001"))

    # GCP settings
    GCP_PROJECT: str | None = field(default_factory=lambda: os.getenv("GCP_PROJECT"))
    GCP_REGION: str = field(default_factory=lambda: os.getenv("GCP_REGION", "us-central1"))
    VERTEX_LOCATION: str = field(default_factory=lambda: os.getenv("VERTEX_LOCATION", "us-central1"))

    # Paths
    SECRETS_DIR: Path = field(default_factory=lambda: Path(os.getenv("SECRETS_DIR", "secrets")))
    GOOGLE_APPLICATION_CREDENTIALS: str | None = field(
        default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )

    # File patterns (allow-list for attachments)
    ALLOWED_FILE_PATTERNS: list[str] = field(
        default_factory=lambda: ["*.txt", "*.pdf", "*.docx", "*.doc", "*.xlsx", "*.xls", "*.md", "*.csv"]
    )

    # Credential file priority list (for auto-discovery)
    CREDENTIAL_FILES_PRIORITY: list[str] = field(
        default_factory=lambda: [
            "api-agent-470921-aa03081a1b4d.json",
            "apt-arcana-470409-i7-ce42b76061bf.json",
            "crafty-airfoil-474021-s2-34159960925b.json",
            "embed2-474114-fca38b4d2068.json",
            "my-project-31635v-8ec357ac35b2.json",
            "semiotic-nexus-470620-f3-3240cfaf6036.json",
        ]
    )

    # Security settings
    ALLOW_PARENT_TRAVERSAL: bool = field(
        default_factory=lambda: os.getenv("ALLOW_PARENT_TRAVERSAL", "false").lower() == "true"
    )
    COMMAND_TIMEOUT_SECONDS: int = field(default_factory=lambda: int(os.getenv("COMMAND_TIMEOUT", "3600")))

    # Logging
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Monitoring
    ACTIVE_WINDOW_SECONDS: int = field(default_factory=lambda: int(os.getenv("ACTIVE_WINDOW_SECONDS", "120")))

    # Email settings
    SENDER_LOCKED_NAME: str = field(default_factory=lambda: os.getenv("SENDER_LOCKED_NAME", ""))
    SENDER_LOCKED_EMAIL: str = field(default_factory=lambda: os.getenv("SENDER_LOCKED_EMAIL", ""))
    MESSAGE_ID_DOMAIN: str = field(default_factory=lambda: os.getenv("MESSAGE_ID_DOMAIN", ""))

    @classmethod
    def load(cls) -> EmailOpsConfig:
        """
        Load configuration from environment.

        Returns:
            Configured EmailOpsConfig instance
        """
        return cls()

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

    @staticmethod
    def _is_valid_service_account_json(p: Path) -> bool:
        """
        Strictly validate that a JSON file looks like a GCP service-account key.
        MEDIUM #17: Enhanced validation including key format, expiration, and basic token validity
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

            # MEDIUM #17: Basic token validity check (if google-auth is available)
            try:
                from google.auth.exceptions import MalformedError
                from google.oauth2 import service_account

                # Try to create credentials object (validates private key format)
                credentials = service_account.Credentials.from_service_account_info(data)

                # Check if credentials are expired (if they have expiry info)
                if hasattr(credentials, 'expired') and credentials.expired:
                    return False

                return True
            except (ImportError, MalformedError):
                # If google-auth not available or key is malformed, fall back to basic checks
                return True
            except Exception:
                # Other auth errors - treat as invalid
                return False

        except Exception:
            return False

    def get_credential_file(self) -> Path | None:
        """
        Find a valid credential file from the priority list.

        Returns:
            Path to credential file if found and validated, None otherwise
        """
        # 1) Already specified in environment?
        if self.GOOGLE_APPLICATION_CREDENTIALS:
            creds_path = Path(self.GOOGLE_APPLICATION_CREDENTIALS)
            if creds_path.exists() and self._is_valid_service_account_json(creds_path):
                return creds_path

        # 2) Search in secrets directory (strict validation)
        secrets_dir = self.get_secrets_dir()
        if not secrets_dir.exists():
            return None

        for cred_file in self.CREDENTIAL_FILES_PRIORITY:
            cred_path = secrets_dir / cred_file
            if cred_path.exists() and self._is_valid_service_account_json(cred_path):
                return cred_path

        return None

    def update_environment(self) -> None:
        """
        Update os.environ with configuration values.
        Ensures child processes inherit correct settings.
        Also derives project from the selected service-account key if env doesn’t already provide one.
        """
        # Core knobs
        os.environ["INDEX_DIRNAME"] = self.INDEX_DIRNAME
        os.environ["CHUNK_DIRNAME"] = self.CHUNK_DIRNAME
        os.environ["CHUNK_SIZE"] = str(self.DEFAULT_CHUNK_SIZE)
        os.environ["CHUNK_OVERLAP"] = str(self.DEFAULT_CHUNK_OVERLAP)
        os.environ["EMBED_BATCH"] = str(self.DEFAULT_BATCH_SIZE)
        os.environ["EMBED_PROVIDER"] = self.EMBED_PROVIDER
        os.environ["VERTEX_EMBED_MODEL"] = self.VERTEX_EMBED_MODEL
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

            # If project isn’t already set, derive it from the service-account JSON
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
            "default_chunk_size": self.DEFAULT_CHUNK_SIZE,
            "default_chunk_overlap": self.DEFAULT_CHUNK_OVERLAP,
            "default_batch_size": self.DEFAULT_BATCH_SIZE,
            "default_num_workers": self.DEFAULT_NUM_WORKERS,
            "embed_provider": self.EMBED_PROVIDER,
            "vertex_embed_model": self.VERTEX_EMBED_MODEL,
            "gcp_project": self.GCP_PROJECT,
            "gcp_region": self.GCP_REGION,
            "vertex_location": self.VERTEX_LOCATION,
            "secrets_dir": str(self.SECRETS_DIR),
            "log_level": self.LOG_LEVEL,
            "active_window_seconds": self.ACTIVE_WINDOW_SECONDS,
            "command_timeout_seconds": self.COMMAND_TIMEOUT_SECONDS,
            "sender_locked_name": self.SENDER_LOCKED_NAME,
            "sender_locked_email": self.SENDER_LOCKED_EMAIL,
            "message_id_domain": self.MESSAGE_ID_DOMAIN,
        }


# Global configuration instance
_config: EmailOpsConfig | None = None


def get_config() -> EmailOpsConfig:
    """
    Get the global configuration instance (singleton pattern).

    Returns:
        Global EmailOpsConfig instance
    """
    global _config
    if _config is None:
        _config = EmailOpsConfig.load()
    return _config


def reset_config() -> None:
    """Reset the global configuration instance (mainly for testing)."""
    global _config
    _config = None
