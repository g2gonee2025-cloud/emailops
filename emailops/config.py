#!/usr/bin/env python3
"""
Centralized configuration for EmailOps.
Manages all configuration values, environment variables, and default settings.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


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
    GCP_PROJECT: Optional[str] = field(default_factory=lambda: os.getenv("GCP_PROJECT"))
    GCP_REGION: str = field(default_factory=lambda: os.getenv("GCP_REGION", "us-central1"))
    VERTEX_LOCATION: str = field(default_factory=lambda: os.getenv("VERTEX_LOCATION", "us-central1"))
    
    # Paths
    SECRETS_DIR: Path = field(default_factory=lambda: Path(os.getenv("SECRETS_DIR", "secrets")))
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )
    
    # File patterns
    ALLOWED_FILE_PATTERNS: List[str] = field(
        default_factory=lambda: ["*.txt", "*.pdf", "*.docx", "*.doc", "*.xlsx", "*.xls", "*.md", "*.csv"]
    )
    
    # Credential file priority list (for auto-discovery)
    CREDENTIAL_FILES_PRIORITY: List[str] = field(default_factory=lambda: [
        "embed2-474114-fca38b4d2068.json",
        "api-agent-470921-4e2065b2ecf9.json",
        "apt-arcana-470409-i7-ce42b76061bf.json",
        "crafty-airfoil-474021-s2-34159960925b.json",
        "my-project-31635v-8ec357ac35b2.json",
        "semiotic-nexus-470620-f3-3240cfaf6036.json",
    ])
    
    # Security settings
    ALLOW_PARENT_TRAVERSAL: bool = field(default_factory=lambda: os.getenv("ALLOW_PARENT_TRAVERSAL", "false").lower() == "true")
    COMMAND_TIMEOUT_SECONDS: int = field(default_factory=lambda: int(os.getenv("COMMAND_TIMEOUT", "3600")))
    
    # Logging
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    # Monitoring
    ACTIVE_WINDOW_SECONDS: int = field(default_factory=lambda: int(os.getenv("ACTIVE_WINDOW_SECONDS", "120")))
    
    @classmethod
    def load(cls) -> 'EmailOpsConfig':
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
        
        # Try relative to this file's parent (emailops package root)
        package_secrets = Path(__file__).parent.parent / self.SECRETS_DIR
        if package_secrets.exists():
            return package_secrets.resolve()
        
        # Return default, even if it doesn't exist
        return self.SECRETS_DIR.resolve()
    
    def get_credential_file(self) -> Optional[Path]:
        """
        Find a valid credential file from the priority list.
        
        Returns:
            Path to credential file if found, None otherwise
        """
        # Check if already specified in environment
        if self.GOOGLE_APPLICATION_CREDENTIALS:
            creds_path = Path(self.GOOGLE_APPLICATION_CREDENTIALS)
            if creds_path.exists():
                return creds_path
        
        # Search in secrets directory
        secrets_dir = self.get_secrets_dir()
        if not secrets_dir.exists():
            return None
        
        for cred_file in self.CREDENTIAL_FILES_PRIORITY:
            cred_path = secrets_dir / cred_file
            if cred_path.exists():
                try:
                    # Validate it's a proper JSON file
                    import json
                    with open(cred_path, 'r') as f:
                        data = json.load(f)
                        if "project_id" in data and "client_email" in data:
                            return cred_path
                except Exception:
                    continue
        
        return None
    
    def update_environment(self) -> None:
        """
        Update os.environ with configuration values.
        Useful for ensuring child processes inherit correct settings.
        """
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
        
        if self.GCP_PROJECT:
            os.environ["GCP_PROJECT"] = self.GCP_PROJECT
            os.environ["GOOGLE_CLOUD_PROJECT"] = self.GCP_PROJECT
            os.environ["VERTEX_PROJECT"] = self.GCP_PROJECT
        
        # Set credentials if found
        cred_file = self.get_credential_file()
        if cred_file:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_file)
    
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
        }


# Global configuration instance
_config: Optional[EmailOpsConfig] = None


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