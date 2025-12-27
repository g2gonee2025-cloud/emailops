from fastapi import HTTPException, Request
from cortex.common.exceptions import SecurityError
from cortex.security.auth import extract_identity, UserIdentity

async def get_current_user(request: Request) -> UserIdentity:
    """
    Dependency to get the current authenticated user's identity.
    Protects endpoints by ensuring a valid, non-anonymous user exists.
    """
    try:
        identity = await extract_identity(request)
        if identity.user_id == "anonymous":
            raise SecurityError("Authentication required", threat_type="auth_missing")
        return identity
    except SecurityError as e:
        raise HTTPException(status_code=401, detail=str(e))
