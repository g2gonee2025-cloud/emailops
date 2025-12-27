"""
Security Dependencies.

FastAPI dependencies for enforcing authentication and authorization.
"""
from fastapi import Depends, HTTPException
from cortex.context import user_id_ctx

def get_user_id() -> str:
    """Retrieves the user ID from the context."""
    return user_id_ctx.get()

def get_current_user(user_id: str = Depends(get_user_id)) -> str:
    """
    Enforces authentication by requiring a valid user_id.

    Raises:
        HTTPException: If the user is anonymous.
    """
    if user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user_id