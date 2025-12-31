"""
Security dependencies for FastAPI.
"""

import logging
from typing import Any, Optional

from cortex.common.exceptions import SecurityError
from cortex.security.auth import extract_identity
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)
auth_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    token: HTTPAuthorizationCredentials | None = Depends(auth_scheme),
) -> dict[str, Any]:
    """
    Dependency to get current user from JWT.
    """
    if token is None or not token.credentials:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        _tenant_id, user_id, claims = await extract_identity(
            request, token=token.credentials
        )
        if user_id == "anonymous":
            raise HTTPException(
                status_code=401,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not isinstance(claims, dict):
            logger.warning("Token claims invalid type; using empty claims.")
            claims = {}
        return claims
    except SecurityError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        logger.exception("Token validation error")
        raise HTTPException(status_code=500, detail="Authentication failed")
