"""
Authentication Routes (Mock)

Provides a development-only login endpoint that issues JWTs.
In production, this would be replaced by an external IdP (Keycloak, Auth0, etc.)
"""

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
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
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int  # seconds


class RefreshRequest(BaseModel):
    """Token refresh request."""

    refresh_token: str = Field(..., min_length=1)


# Mock user database (dev only)
# WARNING: These are development credentials. In production, use a real IDP.
MOCK_USERS = {
    "admin": {
        "password": os.getenv("MOCK_USER_ADMIN_PASSWORD", "admin"),
        "tenant_id": "acme-corp",
        "roles": ["admin"],
    },
    "user": {
        "password": os.getenv("MOCK_USER_USER_PASSWORD", "user"),
        "tenant_id": "acme-corp",
        "roles": ["user"],
    },
    "demo": {
        "password": os.getenv("MOCK_USER_DEMO_PASSWORD", "demo"),
        "tenant_id": "demo-tenant",
        "roles": ["user"],
    },
    # Email-style users for frontend compatibility
    "testuser@emailops.ai": {
        "password": os.getenv("MOCK_USER_TESTUSER_PASSWORD", "test"),
        "tenant_id": "acme-corp",
        "roles": ["admin"],
    },
    "admin@emailops.ai": {
        "password": os.getenv("MOCK_USER_ADMIN_EMAIL_PASSWORD", "admin"),
        "tenant_id": "acme-corp",
        "roles": ["admin"],
    },
}


# Token expiry constants
ACCESS_TOKEN_EXPIRES_SECONDS = 3600  # 1 hour
REFRESH_TOKEN_EXPIRES_SECONDS = 7 * SECONDS_PER_DAY  # 7 days


def _create_tokens(username: str, user_data: dict[str, Any]) -> LoginResponse:
    """Create access and refresh tokens for a user."""
    config = get_config()

    if not config.secret_key:
        raise HTTPException(
            status_code=500, detail="Server misconfiguration: missing SECRET_KEY"
        )

    now_utc = datetime.now(timezone.utc)

    # Access token payload
    access_payload: dict[str, Any] = {
        "sub": username,
        "tenant_id": user_data["tenant_id"],
        "roles": user_data["roles"],
        "iat": now_utc,
        "exp": now_utc + timedelta(seconds=ACCESS_TOKEN_EXPIRES_SECONDS),
        "iss": "emailops-cortex",
        "type": "access",
    }

    # Refresh token payload (longer-lived, minimal claims)
    refresh_payload: dict[str, Any] = {
        "sub": username,
        "tenant_id": user_data["tenant_id"],
        "iat": now_utc,
        "exp": now_utc + timedelta(seconds=REFRESH_TOKEN_EXPIRES_SECONDS),
        "iss": "emailops-cortex",
        "type": "refresh",
    }

    try:
        access_token = jwt.encode(access_payload, config.secret_key, algorithm="HS256")
        refresh_token = jwt.encode(
            refresh_payload, config.secret_key, algorithm="HS256"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token generation failed: {e!s}")

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRES_SECONDS,
    )


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
    # Validate credentials
    user_data = MOCK_USERS.get(request.username)
    # Securely compare passwords to prevent timing attacks in mock auth
    password_match = user_data and secrets.compare_digest(
        user_data["password"], request.password
    )
    if not password_match:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return _create_tokens(request.username, user_data)


@router.post("/refresh", response_model=LoginResponse)
async def refresh(request: RefreshRequest) -> LoginResponse:
    """
    Token refresh endpoint for development.

    Validates the refresh token and issues new access and refresh tokens.
    """
    config = get_config()

    if not config.secret_key:
        raise HTTPException(
            status_code=500, detail="Server misconfiguration: missing SECRET_KEY"
        )

    try:
        # Decode and validate the refresh token
        payload = jwt.decode(
            request.refresh_token,
            config.secret_key,
            algorithms=["HS256"],
            issuer="emailops-cortex",
        )

        # Verify this is a refresh token
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=401,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Get user data from mock database
        username = payload.get("sub")
        user_data = MOCK_USERS.get(username)

        if not user_data:
            raise HTTPException(
                status_code=401,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Issue new tokens
        return _create_tokens(username, user_data)

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Refresh token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        logger.warning("Invalid refresh token: %s", str(e))
        raise HTTPException(
            status_code=401,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
