from __future__ import annotations

from cortex.context import user_id_ctx
from fastapi import HTTPException


async def get_current_user() -> str:
    """
    FastAPI dependency to enforce authentication.

    Retrieves user_id from context set by TenantUserMiddleware.
    If user_id is 'anonymous', it means no valid JWT was provided.
    """
    user_id = user_id_ctx.get()
    if user_id == "anonymous":
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id
