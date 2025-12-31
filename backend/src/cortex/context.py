"""
Context Management.

Defines ContextVars for request-scoped data (tenant, user, correlation ID).
"""

from collections.abc import Mapping
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any

# Context variables for request-scoped data
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="unknown")
tenant_id_ctx: ContextVar[str] = ContextVar("tenant_id", default="default")
user_id_ctx: ContextVar[str] = ContextVar("user_id", default="anonymous")
claims_ctx: ContextVar[Mapping[str, Any]] = ContextVar(
    "claims", default=MappingProxyType({})
)
