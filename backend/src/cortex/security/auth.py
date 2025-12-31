"""
Authentication & Authorization Dependencies.
"""

from __future__ import annotations

import inspect
from typing import Any

from cortex.common.exceptions import (
    SecurityError,
)
from cortex.config.loader import get_config
from cortex.context import claims_ctx, user_id_ctx
from cortex.security.validators import validate_email_format
from fastapi import HTTPException, Request

# Reference to JWT decoder configured in main.py
_jwt_decoder: Any = None


async def get_current_user():
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


async def require_admin() -> dict[str, Any]:
    """FastAPI dependency to enforce admin access."""
    user_id = user_id_ctx.get()
    if not user_id or user_id == "anonymous":
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    claims = claims_ctx.get() or {}
    role = claims.get("role")
    roles = claims.get("roles") or []
    if isinstance(roles, str):
        roles = [roles]
    if role:
        roles = [*roles, role]

    role_names = {r.lower() for r in roles if isinstance(r, str)}
    if "admin" not in role_names:
        raise HTTPException(status_code=403, detail="Admin access required")

    return claims


async def _extract_identity(request: Request) -> tuple[str, str, dict[str, Any]]:
    """Resolve tenant/user identity from JWT or headers."""
    config = get_config()
    auth_header = request.headers.get("Authorization", "")
    claims: dict[str, Any] = {}

    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        if not token:
            raise SecurityError("Bearer token missing", threat_type="auth_missing")
        if _jwt_decoder is None:
            raise SecurityError(
                "JWT decoder not configured", threat_type="auth_invalid"
            )
        # Ensure we await the decoder result
        try:
            decoded = _jwt_decoder(token)
            claims = await decoded if inspect.isawaitable(decoded) else decoded
        except TypeError:
            # Handle case where decoder is synchronous (legacy/fallback fallback)
            claims = _jwt_decoder(token)  # type: ignore[assignment]

    elif config.core.env == "prod":
        raise SecurityError("Authorization header required", threat_type="auth_missing")

    tenant_id = (
        claims.get("tenant_id")
        or claims.get("tid")
        or request.headers.get("X-Tenant-ID")
        or "default"
    )
    user_id = (
        claims.get("sub")
        or claims.get("email")
        or request.headers.get("X-User-ID")
        or "anonymous"
    )

    if isinstance(user_id, str):
        result = validate_email_format(user_id)
        if not result.ok:
            claims["email_invalid"] = user_id
        else:
            user_id = result.value or user_id

    # Extract role, default to "user"
    claims["role"] = claims.get("role", "user")

    return str(tenant_id), str(user_id), claims
