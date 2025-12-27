"""
Context Management.

Defines ContextVars for request-scoped data (tenant, user, correlation ID).
"""

from contextvars import ContextVar
from typing import Any, Dict

# Context variables for request-scoped data
correlation_id_ctx: ContextVar[str | None] = ContextVar("correlation_id", default=None)
tenant_id_ctx: ContextVar[str | None] = ContextVar("tenant_id", default=None)
user_id_ctx: ContextVar[str | None] = ContextVar("user_id", default=None)
claims_ctx: ContextVar[Dict[str, Any] | None] = ContextVar("claims", default=None)
