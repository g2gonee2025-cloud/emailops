"""
Security Dependencies.

FastAPI dependencies for authentication, authorization, and other security measures.
"""

from typing import Any, Dict

import jwt
from cortex.config.loader import get_config
from cortex.context import tenant_id_ctx, user_id_ctx
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
config = get_config()


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    Decode and validate JWT token to extract user information.

    - Verifies token signature and expiration.
    - Extracts user_id and tenant_id.
    - Sets user_id and tenant_id in the context for downstream use.
    """
    credentials_exception = HTTPException(
        status_code=401,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        if not config.SECRET_KEY:
            raise RuntimeError("SECRET_KEY is not configured.")

        payload = jwt.decode(
            token,
            config.SECRET_KEY,
            algorithms=["HS256"],
            issuer="emailops-cortex",
        )
        username: str | None = payload.get("sub")
        tenant_id: str | None = payload.get("tenant_id")

        if username is None or tenant_id is None:
            raise credentials_exception

        # Set context for downstream services
        user_id_ctx.set(username)
        tenant_id_ctx.set(tenant_id)

        return {"username": username, "tenant_id": tenant_id}
    except jwt.PyJWTError:
        raise credentials_exception
    except RuntimeError as e:
        # 500 Internal Server Error for configuration issues
        raise HTTPException(status_code=500, detail=str(e))
