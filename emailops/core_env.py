from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .core_exceptions import LLMError

"""
env_utils.py — Environment and account management utilities.

Provides helpers for managing GCP/Vertex AI accounts and credentials.
"""

__all__ = [
    "DEFAULT_ACCOUNTS",
    "LLMError",
    "VertexAccount",
    "_init_vertex",
    "load_validated_accounts",
    "reset_vertex_init",
    "save_validated_accounts",
    "validate_account",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VertexAccount:
    """Configuration for a single Vertex AI account/project."""
    project_id: str
    location: str = "us-central1"
    credentials_path: str = ""

    def __post_init__(self):
        if not self.project_id:
            raise ValueError("project_id is required for VertexAccount")


# Default accounts loaded from config
def _load_default_accounts() -> list[VertexAccount]:
    """Load default accounts from .env or config."""
    accounts = []

    # Load accounts from validated_accounts.json (created by setup script)
    accounts_path = Path("~/.emailops/validated_accounts.json").expanduser()
    if accounts_path.exists():
        try:
            with accounts_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for item in (data if isinstance(data, list) else [data]):
                if isinstance(item, dict) and item.get("project_id"):
                    accounts.append(VertexAccount(**item))
            logger.info("Loaded %d accounts from validated_accounts.json", len(accounts))
        except Exception as e:
            logger.warning("Failed to load validated accounts: %s", e)

    # Fallback: try to build from environment variables
    if not accounts:
        project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GCP_LOCATION", "us-central1")
        creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

        if project:
            accounts.append(VertexAccount(
                project_id=project,
                location=location,
                credentials_path=creds
            ))
            logger.info("Loaded 1 account from environment variables")

    if not accounts:
        logger.warning("No GCP accounts found. Set up accounts with: python -m setup.enable_vertex_apis")

    return accounts

DEFAULT_ACCOUNTS: list[VertexAccount] = _load_default_accounts()


def load_validated_accounts(
    accounts_file: str | Path | None = None,
    default_accounts: list[VertexAccount] | None = None
) -> list[VertexAccount]:
    """
    Load validated Vertex accounts from file or use defaults.

    Args:
        accounts_file: Optional path to accounts JSON file
        default_accounts: Optional default accounts list

    Returns:
        List of validated VertexAccount objects
    """
    if accounts_file:
        p = Path(accounts_file).expanduser()
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                accounts = []
                for obj in (raw if isinstance(raw, list) else [raw]):
                    if isinstance(obj, dict):
                        accounts.append(VertexAccount(**obj))
                return accounts
            except Exception as e:
                logger.warning("Failed to load accounts from %s: %s", p, e)

    # Use provided defaults or module-level DEFAULT_ACCOUNTS
    return list(default_accounts or DEFAULT_ACCOUNTS)


def save_validated_accounts(
    accounts_file: str | Path,
    accounts: list[VertexAccount]
) -> None:
    """
    Save validated accounts to JSON file.

    Args:
        accounts_file: Path to save accounts
        accounts: List of VertexAccount objects
    """
    p = Path(accounts_file).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = [
        {
            "project_id": a.project_id,
            "location": a.location,
            "credentials_path": a.credentials_path,
        }
        for a in accounts
    ]

    # Atomic write
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        tmp.replace(p)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def validate_account(account: VertexAccount) -> None:
    """
    Validate that a VertexAccount has required fields.

    Args:
        account: VertexAccount to validate

    Raises:
        LLMError: If account is invalid
    """
    if not account.project_id:
        raise LLMError("VertexAccount must have project_id")
    if not account.location:
        raise LLMError("VertexAccount must have location")


# Vertex initialization state (for lazy init patterns)
_vertex_initialized = False


def _init_vertex() -> None:
    """Mark Vertex as initialized (stateful helper for lazy init)."""
    global _vertex_initialized
    _vertex_initialized = True


def reset_vertex_init() -> None:
    """Reset Vertex initialization state (useful for testing)."""
    global _vertex_initialized
    _vertex_initialized = False
