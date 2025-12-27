from fastapi import Request
from pydantic import BaseModel
from cortex.common.exceptions import SecurityError
from typing import Dict, Any
import inspect

class UserIdentity(BaseModel):
    """Represents the authenticated identity of a user."""
    tenant_id: str
    user_id: str
    claims: Dict[str, Any]

async def extract_identity(request: Request) -> UserIdentity:
    """
    Extracts identity from the request and returns a structured object.
    Relies on `jwt_decoder` and `config` being attached to the `app.state`.
    """
    from cortex.security.validators import validate_email_format

    try:
        _jwt_decoder = request.app.state.jwt_decoder
        config = request.app.state.config
    except AttributeError as e:
        raise RuntimeError(f"FastAPI app.state is not configured correctly: {e}")

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
        try:
            decoded = _jwt_decoder(token)
            claims = await decoded if inspect.isawaitable(decoded) else decoded
        except TypeError:
            claims = _jwt_decoder(token)

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

    claims["role"] = claims.get("role", "user")

    return UserIdentity(tenant_id=str(tenant_id), user_id=str(user_id), claims=claims)
