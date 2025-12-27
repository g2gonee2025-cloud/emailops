"""
Authentication and identity extraction functions.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Awaitable, Callable, Optional

import jwt
import requests
from cortex.common.exceptions import SecurityError
from cortex.config.loader import get_config
from cortex.security.validators import validate_email_format
from fastapi import HTTPException, Request
from jwt.exceptions import PyJWTError as JWTError

logger = logging.getLogger(__name__)

# JWT/JWKS helpers
_jwt_decoder: Optional[Callable[[str], Any]] = None  # Returns Awaitable[dict] or dict
_jwks_cache: dict[str, Any] = {}


def _load_jwks(jwks_url: str) -> dict[str, Any]:
    if jwks_url in _jwks_cache:
        return _jwks_cache[jwks_url]
    response = requests.get(jwks_url, timeout=5)
    response.raise_for_status()
    data = response.json()
    _jwks_cache[jwks_url] = data
    return data


def configure_jwt_decoder(
    *, jwks_url: Optional[str], audience: Optional[str], issuer: Optional[str]
) -> None:
    global _jwt_decoder

    config = get_config()

    if jwks_url:
        _jwt_decoder = _create_jwks_decoder(jwks_url, audience, issuer)
        return

    # Fallback logic
    if config.core.env == "prod":
        _jwt_decoder = _create_prod_reject_decoder()
        return

    # Dev mode: Allow verified secret
    _jwt_decoder = _create_dev_secret_decoder(config)


def _create_jwks_decoder(
    jwks_url: str, audience: Optional[str], issuer: Optional[str]
) -> Callable[[str], Awaitable[dict[str, Any]]]:
    """Create a JWKS-based JWT decoder."""
    from fastapi.concurrency import run_in_threadpool

    async def decode(token: str) -> dict[str, Any]:
        try:
            jwks = await run_in_threadpool(_load_jwks, jwks_url)
            key_data = _find_jwks_key(jwks, token)

            from jwt import algorithms

            public_key = algorithms.RSAAlgorithm.from_jwk(key_data)

            decode_options = {}
            if not audience:
                decode_options["verify_aud"] = False
            if not issuer:
                decode_options["verify_iss"] = False

            return jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience=audience if audience else None,
                issuer=issuer if issuer else None,
                options=decode_options,
            )
        except JWTError as exc:
            raise SecurityError("Invalid JWT", threat_type="auth_invalid") from exc
        except Exception as exc:
            raise SecurityError(
                "Failed to load JWKS", threat_type="auth_invalid"
            ) from exc

    return decode


def _find_jwks_key(jwks: dict[str, Any], token: str) -> dict[str, Any]:
    """Find the matching key in JWKS for the given token."""
    headers = jwt.get_unverified_header(token)
    kid = headers.get("kid")
    for candidate in jwks.get("keys", []):
        if candidate.get("kid") == kid:
            return candidate
    raise SecurityError("JWT key id not found in JWKS", threat_type="auth_invalid")


def _create_prod_reject_decoder() -> Callable[[str], Awaitable[dict[str, Any]]]:
    """Create a decoder that rejects all tokens in production without JWKS."""

    async def reject(_: str) -> dict[str, Any]:
        raise SecurityError(
            "JWKS configuration required in production",
            threat_type="auth_configuration",
        )

    return reject


def _create_dev_secret_decoder(
    config: Any,
) -> Callable[[str], Awaitable[dict[str, Any]]]:
    """Create a dev-mode decoder using secret key."""

    async def decode_verified_secret(token: str) -> dict[str, Any]:
        try:
            payload = jwt.decode(token, config.SECRET_KEY, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
        except JWTError as exc:
            raise SecurityError("Invalid JWT", threat_type="auth_invalid") from exc

    return decode_verified_secret


async def extract_identity(request: Request) -> tuple[str, str, dict[str, Any]]:
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
