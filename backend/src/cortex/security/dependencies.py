"""
Security Dependencies.

Provides FastAPI dependencies for enforcing authentication and authorization.
"""
from __future__ import annotations

from typing import Any, Dict

from cortex.context import tenant_id_ctx, user_id_ctx
from fastapi import HTTPException, status


def get_current_user() -> Dict[str, Any]:
    """
    FastAPI dependency to enforce user authentication.

    Retrieves the user and tenant ID from context variables populated
    by the `TenantUserMiddleware`. If the user is anonymous or the
    tenant is the default, it raises a 401 Unauthorized error.

    This ensures that endpoints are protected and have a valid user context.

    Returns:
        A dictionary containing the current user's ID and tenant ID.

    Raises:
        HTTPException: If the user is not authenticated.
    """
    user_id = user_id_ctx.get()
    tenant_id = tenant_id_ctx.get()

    # The middleware sets 'anonymous' or 'default' if no valid identity is found.
    if user_id == "anonymous" or tenant_id == "default":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {"user_id": user_id, "tenant_id": tenant_id}
