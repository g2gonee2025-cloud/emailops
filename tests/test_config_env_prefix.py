"""Tests for configuration environment variable handling."""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure backend sources are importable during tests
ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = ROOT / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from cortex.config.models import DatabaseConfig  # noqa: E402


def test_canonical_prefix_takes_precedence(monkeypatch):
    monkeypatch.delenv("DB_URL", raising=False)
    monkeypatch.setenv(
        "OUTLOOKCORTEX_DB_URL", "postgresql://canonical:5432/emailops"
    )
    monkeypatch.setenv("EMAILOPS_DB_URL", "postgresql://legacy:5432/emailops")

    cfg = DatabaseConfig()

    assert cfg.url == "postgresql://canonical:5432/emailops"


def test_legacy_prefix_is_still_supported(monkeypatch):
    monkeypatch.delenv("DB_URL", raising=False)
    monkeypatch.delenv("OUTLOOKCORTEX_DB_URL", raising=False)
    monkeypatch.setenv("EMAILOPS_DB_URL", "postgresql://legacy:5432/emailops")

    cfg = DatabaseConfig()

    assert cfg.url == "postgresql://legacy:5432/emailops"
