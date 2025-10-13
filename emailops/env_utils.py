from __future__ import annotations

from . import llm_runtime as _rt

# emailops/env_utils.py
"""
Back-compat shim over the merged runtime module.

This module re-exports selected symbols from `emailops.llm_runtime`.
It must not mutate or extend types from `llm_runtime`; avoid monkey-patching
or import-time side effects here to keep behavior predictable.
"""

# Import the runtime module once and re-export the required names. This pattern
# prevents linter "imported but unused" warnings (F401) and keeps the shim pure.
LLMError = _rt.LLMError
VertexAccount = _rt.VertexAccount
load_validated_accounts = _rt.load_validated_accounts
save_validated_accounts = _rt.save_validated_accounts
validate_account = _rt.validate_account
DEFAULT_ACCOUNTS = _rt.DEFAULT_ACCOUNTS
_init_vertex = _rt._init_vertex
reset_vertex_init = _rt.reset_vertex_init

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
