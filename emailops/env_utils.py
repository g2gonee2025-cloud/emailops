# emailops/env_utils.py
from __future__ import annotations
# Back-compat shim over the merged runtime module
from .llm_runtime import (
    LLMError,
    VertexAccount,
    load_validated_accounts,
    save_validated_accounts,
    validate_account,
    DEFAULT_ACCOUNTS,
    _init_vertex,          # keep underscore name to preserve existing imports
    reset_vertex_init,
)

__all__ = [
    "LLMError", "VertexAccount", "load_validated_accounts", "save_validated_accounts",
    "validate_account", "DEFAULT_ACCOUNTS", "_init_vertex", "reset_vertex_init",
]
