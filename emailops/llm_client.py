# emailops/llm_client.py
from __future__ import annotations

"""
emailops.llm_client
-------------------
Back-compat shim over the unified runtime in `llm_runtime`.

Production guarantees:
- Stable import surface for legacy code: complete_text, complete_json, embed_texts, LLMError
- Backwards aliases for older call sites: complete(), json_complete(), embed()
- Transparent passthrough of any new symbols added to `llm_runtime` via __getattr__

Behavior: this module does not implement any logic itself; it delegates to
`emailops.llm_runtime` for all functionality.
"""

from typing import Any, Iterable

# Import the runtime once (same behavior as the original shim), then re-export.
from . import llm_runtime as _rt

# ---- Primary, supported surface (kept for existing callers) ------------------
complete_text = _rt.complete_text
complete_json = _rt.complete_json
embed_texts = _rt.embed_texts
LLMError = _rt.LLMError

# ---- Backwards-friendly aliases (no behavior change) -------------------------
def complete(system: str, user: str, **kwargs: Any) -> str:
    """Alias for complete_text(...). See llm_runtime.complete_text for parameters."""
    return complete_text(system, user, **kwargs)

def json_complete(system: str, user: str, **kwargs: Any) -> str:
    """Alias for complete_json(...). See llm_runtime.complete_json for parameters."""
    return complete_json(system, user, **kwargs)

def embed(texts: Iterable[str], **kwargs: Any):
    """
    Alias for embed_texts(...).
    Accepts any iterable of strings and forwards provider/model kwargs unmodified.
    """
    return embed_texts(list(texts), **kwargs)

__all__ = [
    # Core surface
    "complete_text", "complete_json", "embed_texts", "LLMError",
    # Convenience/compat aliases
    "complete", "json_complete", "embed",
]

# ---- Transparent passthrough for any new helpers added to llm_runtime --------
def __getattr__(name: str) -> Any:  # PEP 562 – module-level getattr
    try:
        return getattr(_rt, name)
    except AttributeError as exc:
        raise AttributeError(f"module 'emailops.llm_client' has no attribute '{name}'") from exc

def __dir__() -> list[str]:
    """Expose both shim symbols and llm_runtime symbols to tooling/IDE introspection."""
    return sorted(set(list(globals().keys()) + dir(_rt)))
