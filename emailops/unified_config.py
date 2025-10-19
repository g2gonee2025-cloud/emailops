"""
unified_config.py

A new, unified configuration model for the emailops application.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class UnifiedConfig:
    """
    A unified configuration class for the emailops application.
    """
    # Core
    export_root: str = ""
    provider: str = "vertex"
    persona: str = os.getenv("PERSONA", "expert insurance CSR")

    # Search
    sim_threshold: float = 0.30
    k: int = 25
    mmr_lambda: float = 0.70
    rerank_alpha: float = 0.35

    # Email Generation
    reply_tokens: int = 20000
    fresh_tokens: int = 10000
    reply_policy: str = "reply_all"
    temperature: float = 0.2

    # Chat
    chat_session_id: str = "default"
    max_chat_history: int = 5

    # Last used values
    last_to: str = ""
    last_cc: str = ""
    last_subject: str = ""

    # Vertex/GCP
    vertex_embed_model: str = "gemini-embedding-001"
    gcp_project: str = ""
    gcp_region: str = "global"
    vertex_location: str = "us-central1"

    # Email
    sender_locked_name: str = ""
    sender_locked_email: str = ""
    message_id_domain: str = ""

    # Processing
    num_workers: int = 4
    batch_size: int = 64
    chunk_size: int = 1600
    chunk_overlap: int = 200

    # File Patterns
    allowed_file_patterns: list[str] = field(default_factory=lambda: [
        "*.pdf", "*.docx", "*.doc", "*.xlsx", "*.xls", "*.xlsm", "*.xlsb", "*.csv",
        "*.pptx", "*.ppt", "*.pptm", "*.txt", "*.md", "*.rtf",
        "*.png", "*.jpg", "*.jpeg", "*.gif", "*.tif", "*.tiff", "*.webp", "*.heic", "*.heif",
        "*.eml", "*.msg", "*.html", "*.htm", "*.json", "*.xml"
    ])

    # Time Decay
    half_life_days: int = 30

    # LLM Client
    timeout_seconds: float = 30.0
    max_retries: int = 4
    initial_backoff_seconds: float = 0.25
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 8.0
    jitter: bool = True
    rate_limit_per_sec: float = 0.0
    rate_limit_capacity: int = 0
    circuit_failure_threshold: int = 5
    circuit_reset_seconds: float = 30.0
    cache_ttl_seconds: float = 0.0
    cache_max_entries: int = 1024

    def save(self, path: Path) -> None:
        """Save the configuration to a file."""
        with Path.open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization/GUI display."""
        return asdict(self)

    def update_environment(self) -> None:
        """Update environment variables from config (for child processes)."""
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

    @classmethod
    def load(cls, path: Path) -> UnifiedConfig:
        """Load the configuration from a file."""
        if not path.exists():
            return cls()
        with Path.open(path) as f:
            data = json.load(f)

        # Override with environment variables
        for key, _value in data.items():
            env_var = f"EMAILOPS_{key.upper()}"
            if env_var in os.environ:
                data[key] = os.environ[env_var]

        return cls(**data)
