"""Security dependencies for FastAPI."""

from __future__ import annotations

from cortex.context import claims_ctx
from fastapi import Depends, HTTPException, Request


async def get_current_user(request: Request) -> dict:
    """
    Dependency to get the current user from context.

    In a real app, this would be more robust, likely decoding a JWT
    and fetching user details from a database.
    """
    # For now, we'll rely on the claims set by the middleware.
    # A more robust implementation would re-verify the JWT here.
    claims = claims_ctx.get()
    if not claims or not claims.get("sub"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return claims
