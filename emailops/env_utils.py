from __future__ import annotations

"""
LOW #31: DEPRECATED - Back-compat shim over the merged runtime module.

This module re-exports symbols from `emailops.llm_runtime` for backward compatibility.

DEPRECATION WARNING: This module will be removed in a future version.
Please update imports to use llm_runtime directly:
    from emailops.llm_runtime import <function_name>
instead of:
    from emailops.env_utils import <function_name>
"""

import warnings

from . import llm_runtime as _rt

# Issue deprecation warning when module is imported
warnings.warn(
    "env_utils.py is deprecated. Import from llm_runtime instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export required symbols
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
