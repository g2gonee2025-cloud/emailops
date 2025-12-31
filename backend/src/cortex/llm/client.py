"""
LLM Client Shim.

Implements §7.2.2 of the Canonical Blueprint.
Dynamic proxy to runtime module-level functions.

Usage:
    from cortex.llm.client import embed_texts, complete_text, complete_json
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

_RUNTIME: Any | None = None


def _get_runtime() -> Any:
    global _RUNTIME
    if _RUNTIME is None:
        try:
            _RUNTIME = importlib.import_module("cortex.llm.runtime")
        except ImportError as exc:
            raise ImportError(
                "cortex.llm.runtime is required for LLM operations."
            ) from exc
    return _RUNTIME


def embed_documents(documents: list[str]) -> np.ndarray:
    return _get_runtime().embed_documents(documents)


def embed_queries(queries: list[str]) -> np.ndarray:
    return _get_runtime().embed_queries(queries)


def embed_texts(texts: list[str]) -> np.ndarray:
    return _get_runtime().embed_texts(texts)


def complete_messages(messages: list[dict[str, str]], **kwargs: Any) -> str:
    return _get_runtime().complete_messages(messages, **kwargs)


def complete_text(prompt: str, **kwargs: Any) -> str:
    return _get_runtime().complete_text(prompt, **kwargs)


def complete_json(prompt: str, schema: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return _get_runtime().complete_json(prompt, schema, **kwargs)


__all__ = [
    "complete_json",
    "complete_messages",
    "complete_text",
    "embed_documents",
    "embed_queries",
    "embed_texts",
]


def __dir__() -> list[str]:
    return sorted(__all__)
