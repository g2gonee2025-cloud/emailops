"""
Authentication Routes (Mock)

Provides a development-only login endpoint that issues JWTs.
In production, this would be replaced by an external IdP (Keycloak, Auth0, etc.)
"""

import logging
import secrets
from datetime import datetime, timedelta
from typing import Any

import jwt
from cortex.config.loader import get_config
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Constants
SECONDS_PER_DAY = 86400


# Initialize logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])
config = get_config()

# CRITICAL: This entire router should be disabled in production.
# The mock login endpoint is a security risk if exposed.
if config.core.env != "dev":
    logger.warning(
        "Disabling mock auth router in non-dev environment (%s)", config.core.env
    )
    # This prevents the routes from being registered with the FastAPI app
    router = APIRouter()


class LoginRequest(BaseModel):
    """Login credentials."""

    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    """Login response with JWT token."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int  # seconds


# Mock user database (dev only)
# WARNING: These are development credentials. In production, use a real IDP.
MOCK_USERS = {
    "admin": {"password": "admin", "tenant_id": "acme-corp", "roles": ["admin"]},
    "user": {"password": "user", "tenant_id": "acme-corp", "roles": ["user"]},
    "demo": {"password": "demo", "tenant_id": "demo-tenant", "roles": ["user"]},
}


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest) -> LoginResponse:
    """
    Mock login endpoint for development.

    Issues a JWT signed with SECRET_KEY containing user claims.

    **Dev Credentials:**
    - admin/admin (admin role)
    - user/user (user role)
    - demo/demo (demo tenant)
    """
    config = get_config()

    # Validate credentials
    user_data = MOCK_USERS.get(request.username)
    # Securely compare passwords to prevent timing attacks in mock auth
    password_match = user_data and secrets.compare_digest(
        user_data["password"], request.password
    )
    if not password_match:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create JWT payload
    if not config.SECRET_KEY:
        raise HTTPException(
            status_code=500, detail="Server misconfiguration: missing SECRET_KEY"
        )

    now_utc = datetime.utcnow()
    expires_seconds = 86400  # 24 hours

    payload: dict[str, Any] = {
        "sub": request.username,
        "tenant_id": user_data["tenant_id"],
        "roles": user_data["roles"],
        "iat": now_utc,
        "exp": now_utc + timedelta(seconds=expires_seconds),
        "iss": "emailops-cortex",
    }

    try:
        token = jwt.encode(payload, config.SECRET_KEY, algorithm="HS256")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Token generation failed: {str(e)}"
        )

    return LoginResponse(access_token=token, expires_in=expires_seconds)
