"""
Security dependencies for FastAPI.
"""
from __future__ import annotations

import logging

from cortex.config.loader import get_config
from cortex.context import tenant_id_ctx, user_id_ctx
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Decodes the JWT token to authenticate the user and populates context variables.

    This implementation validates a JWT, extracts user and tenant claims,
    and sets them in the application context for downstream services.
    """
    config = get_config()
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decode the JWT using the application's secret key and algorithm
        payload = jwt.decode(
            token,
            config.SECRET_KEY,
            algorithms=[config.security.jwt_algorithm],
        )
        user_id: str | None = payload.get("sub")  # Standard claim for subject (user ID)
        tenant_id: str | None = payload.get("tid")  # Custom claim for tenant ID

        if user_id is None or tenant_id is None:
            logger.warning("Token missing required 'sub' or 'tid' claims.")
            raise credentials_exception

    except JWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise credentials_exception from e

    # CRITICAL FIX: Populate context variables for use in the service layer.
    user_id_ctx.set(user_id)
    tenant_id_ctx.set(tenant_id)

    logger.info(f"User '{user_id}' in tenant '{tenant_id}' authenticated successfully.")

    # The return value can be used directly in the endpoint if needed.
    return {"user_id": user_id, "tenant_id": tenant_id}
