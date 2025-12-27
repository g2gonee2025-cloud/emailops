from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = REPO_ROOT / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from cortex.config.loader import reset_config

# Ensure the CLI package is importable when running tests from repo root
CLI_SRC = REPO_ROOT / "cli" / "src"
if str(CLI_SRC) not in sys.path:
    sys.path.insert(0, str(CLI_SRC))

from cortex_cli import main as cli_main


def test_show_config_uses_embedding_section(capsys, monkeypatch) -> None:
    """The config command should render the embedding configuration without errors."""
    reset_config()
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    cli_main._show_config(validate=False, export_format=None)

    output = capsys.readouterr().out
    reset_config()

    assert "Embeddings" in output
    assert "Model" in output
    assert "Dimensions" in output
    assert "Batch Size" in output
