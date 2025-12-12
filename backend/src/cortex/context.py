from contextvars import ContextVar
from typing import Any, Dict

# Context variables for request-scoped data
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="")
tenant_id_ctx: ContextVar[str] = ContextVar("tenant_id", default="")
user_id_ctx: ContextVar[str] = ContextVar("user_id", default="")
claims_ctx: ContextVar[Dict[str, Any]] = ContextVar("claims", default={})
