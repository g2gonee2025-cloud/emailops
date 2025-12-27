"""
Authentication & Authorization Dependencies.
"""
from __future__ import annotations

from cortex.context import user_id_ctx
from fastapi import HTTPException


async def get_current_user(
):
    """
    FastAPI dependency to enforce user authentication.

    This dependency checks if a user_id is present in the context,
    which is set by the TenantUserMiddleware. If the user_id is
    "anonymous", it means the request was unauthenticated.
    """
    user_id = user_id_ctx.get()
    if not user_id or user_id == "anonymous":
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id