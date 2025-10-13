from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

# Eager import of the runtime (unchanged); keep llm_runtime free of imports back here.
from . import llm_runtime as _rt

"""
emailops.llm_client
-------------------
Back-compat shim over the unified runtime in `llm_runtime`.

Production guarantees:
- Stable import surface for legacy code: complete_text, complete_json, embed_texts, LLMError
- Backwards aliases for older call sites: complete(), json_complete(), embed()
- Transparent passthrough of any new symbols added to `llm_runtime` via __getattr__

Changes vs prior version:
- Avoid import-time coupling to specific runtime functions (no @wraps on _rt.*).
- Do not cache forwarded attributes; results always reflect the current runtime.
- LLMError is resolved dynamically, never pinned at import time.
- Hardened embed(): rejects bytes-like inputs and validates element types.
- __all__ / __dir__ are dynamic (reflect latest runtime exports).
"""


# ---- Internal helpers ---------------------------------------------------------
def _rt_attr(name: str) -> Any:
    """Resolve an attribute on the runtime with a consistent error message."""
    try:
        return getattr(_rt, name)
    except AttributeError as exc:
        raise AttributeError(f"llm_runtime.{name} is not available") from exc


# ---- Primary, supported surface (lazy target lookup at call time) ------------
def complete_text(*args: Any, **kwargs: Any) -> Any:
    """
    Thin wrapper over `llm_runtime.complete_text`.
    Signature and behavior follow the underlying runtime at call-time.
    """
    return _rt_attr("complete_text")(*args, **kwargs)


def complete_json(*args: Any, **kwargs: Any) -> Any:
    """
    Thin wrapper over `llm_runtime.complete_json`.
    Signature and behavior follow the underlying runtime at call-time.
    """
    return _rt_attr("complete_json")(*args, **kwargs)


def embed_texts(texts: Iterable[str], **kwargs: Any) -> Any:
    """
    Thin passthrough to runtime; the runtime now realizes non-list iterables once
    at the boundary for provider compatibility and batching efficiency.
    """
    return _rt_attr("embed_texts")(texts, **kwargs)


# ---- Backwards-friendly aliases ----------------------------------------------
def complete(*args: Any, **kwargs: Any) -> Any:
    """Alias for complete_text(...)."""
    return complete_text(*args, **kwargs)


def json_complete(*args: Any, **kwargs: Any) -> Any:
    """Alias for complete_json(...)."""
    return complete_json(*args, **kwargs)


def embed(texts: Iterable[str], **kwargs: Any) -> Any:
    """
    Alias for embed_texts(...).

    Safety:
    - Passing a single string is almost always a bug (str is an iterable of chars).
      This will raise TypeError; wrap it in a list instead: embed([text]).
    - To meet common provider expectations, this realizes non-list iterables into a list.
      For very large streams/generators, chunk externally before calling.
    """
    if isinstance(texts, (str, bytes, bytearray, memoryview)):
        raise TypeError(
            "embed(texts) expects an iterable of strings, not a single string/bytes; "
            "wrap the value in a list, e.g., embed([text])."
        )
    seq = list(texts)
    if not all(isinstance(t, str) for t in seq):
        raise TypeError("embed(texts) expects an iterable of str.")
    return embed_texts(seq, **kwargs)


# ---- Public API surface (dynamic) --------------------------------------------
# Names we always export from this shim (LLMError handled dynamically in __all__).
_CORE_EXPORTS = [
    # Core surface
    "complete_text",
    "complete_json",
    "embed_texts",
    # Convenience/compat aliases
    "complete",
    "json_complete",
    "embed",
    # Error class is resolved dynamically; see __getattr__("__all__").
    "LLMError",
]


def _runtime_exports() -> list[str]:
    """Fetch runtime's declared public exports *now* (no caching)."""
    rt_all = getattr(_rt, "__all__", None)
    # Accept common iterables of strings, not just list/tuple.
    if isinstance(rt_all, (list, tuple)):
        return [n for n in rt_all if isinstance(n, str)]
    try:
        from collections.abc import Iterable as _Iter
        if isinstance(rt_all, _Iter):
            return [n for n in rt_all if isinstance(n, str)]
    except Exception:
        pass
    # If runtime doesn't declare a usable __all__, expose nothing extra by default.
    return []


def __getattr__(name: str) -> Any:  # PEP 562 - module-level getattr
    """
    Dynamic attribute forwarding to the runtime.

    Special case: when asked for __all__, compose a live export list consisting of the
    shim's core exports plus the runtime's current exports. 'LLMError' is only included
    when the runtime actually provides it, preventing fragile 'import *' behavior.
    """
    if name == "__all__":
        # Build core exports but only include LLMError if present in the runtime.
        core = []
        has_llm_error = hasattr(_rt, "LLMError")
        for n in _CORE_EXPORTS:
            if n == "LLMError" and not has_llm_error:
                continue
            core.append(n)

        # Deduplicate while preserving order.
        seen: set[str] = set()
        merged: list[str] = []
        for n in core + _runtime_exports():
            if n not in seen:
                seen.add(n)
                merged.append(n)
        return merged

    # Forward to runtime for any other attribute (including LLMError and new symbols).
    try:
        return getattr(_rt, name)
    except AttributeError as exc:
        raise AttributeError(
            f"module 'emailops.llm_client' has no attribute '{name}'"
        ) from exc


def __dir__() -> list[str]:
    """Expose just the public API for cleaner IDE/tooling completion (live view)."""
    return sorted(set(__getattr__("__all__")))


# ---- Optional: help static type checkers without changing runtime -------------
if TYPE_CHECKING:
    # Import for type checkers only (no runtime cost), helping IDEs infer signatures.
    from .llm_runtime import (  # noqa: F401
        complete_text as _complete_text_t,
    )
