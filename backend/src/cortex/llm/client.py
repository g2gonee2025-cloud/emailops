"""
LLM Client Shim.

Implements §7.2.2 of the Canonical Blueprint.
Dynamic proxy to runtime module-level functions.

Usage:
    from cortex.llm.client import embed_texts, complete_text, complete_json
"""
import importlib
from typing import Any

_rt = importlib.import_module("cortex.llm.runtime")


def __getattr__(name: str) -> Any:
    """Forward attribute access to runtime module."""
    return getattr(_rt, name)


def __dir__() -> list[str]:
    """List available names from runtime module."""
    return sorted(set(globals().keys()) | set(dir(_rt)))