"""
Security Dependencies.

Reusable dependencies for endpoint-level security.
Implements §11.2 of the Canonical Blueprint.
"""
from __future__ import annotations

from typing import Any, Dict

from cortex.context import claims_ctx, tenant_id_ctx, user_id_ctx
from fastapi import HTTPException
from pydantic import BaseModel, Field


class CurrentUser(BaseModel):
    """Represents the current authenticated user's identity."""

    tenant_id: str = Field(..., description="Tenant ID")
    user_id: str = Field(..., description="User ID")
    claims: Dict[str, Any] = Field(..., description="JWT claims")


def get_current_user() -> CurrentUser:
    """
    FastAPI dependency to get the current user from context.

    Raises:
        HTTPException: If the user is not authenticated (i.e., is anonymous).

    Returns:
        CurrentUser: The authenticated user's identity.
    """
    user_id = user_id_ctx.get()
    tenant_id = tenant_id_ctx.get()
    claims = claims_ctx.get()

    # The TenantUserMiddleware sets user_id to "anonymous" if no valid
    # JWT or header is found. This is the canonical check for authentication.
    if user_id == "anonymous":
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return CurrentUser(tenant_id=tenant_id, user_id=user_id, claims=claims)
