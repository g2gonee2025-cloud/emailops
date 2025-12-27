"""
Security dependencies for FastAPI.
"""
import logging
from typing import Any, Dict, Optional

from cortex.security.auth import _extract_identity
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)
auth_scheme = HTTPBearer()

async def get_current_user(
    request: Request, token: HTTPAuthorizationCredentials = Depends(auth_scheme)
) -> Dict[str, Any]:
    """
    Dependency to get current user from JWT.
    """
    if token is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        tenant_id, user_id, claims = await _extract_identity(request)
        if user_id == "anonymous":
            raise HTTPException(status_code=401, detail="Invalid token")
        return claims
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")