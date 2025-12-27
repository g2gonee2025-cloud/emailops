import logging
import os
from typing import Any, Dict, List

from cortex.config.loader import get_config
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import check functions from CLI module
# We can reuse the individual check functions which return (success, details, error)
try:
    from cortex_cli.cmd_doctor import (
        _probe_embeddings,
        check_postgres,
        check_redis,
        check_reranker,
    )

    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False


router = APIRouter(prefix="/admin", tags=["admin"])
logger = logging.getLogger(__name__)


class DoctorCheckResult(BaseModel):
    name: str
    status: str  # "pass", "fail", "warn"
    message: str | None = None
    details: Dict[str, Any] | None = None


class DoctorReport(BaseModel):
    overall_status: str  # "healthy", "degraded", "unhealthy"
    checks: List[DoctorCheckResult]


@router.get("/config")
async def get_config_info() -> Dict[str, Any]:
    """Get sanitized safe configuration."""
    config = get_config()
    return {
        "environment": config.core.env if config.core else None,
        "provider": config.core.provider if config.core else None,
        "log_level": config.system.log_level if config.system else None,
        "database_url": "***" if config.database and config.database.url else None,
        # Add other safe fields as needed
    }


@router.get("/status")
async def get_system_status() -> Dict[str, Any]:
    """Get simple system status info."""
    return {
        "status": "online",
        "service": "cortex-backend",
        "env": os.getenv("OUTLOOKCORTEX_ENV", "unknown"),
    }


@router.post("/doctor", response_model=DoctorReport)
async def run_doctor() -> DoctorReport:
    """Run system diagnostics."""
    if not CLI_AVAILABLE:
        raise HTTPException(
            status_code=500, detail="CLI module not available to run diagnostics"
        )

    config = get_config()
    checks: List[DoctorCheckResult] = []
    has_failure = False

    # 1. Database Check
    try:
        res = check_postgres(config)
        if len(res) == 3:
            success, details, error = res
        else:
            success, error = res[0], res[-1]
            details = {}
    except Exception as e:
        success, details, error = False, {}, str(e)

    checks.append(
        DoctorCheckResult(
            name="PostgreSQL",
            status="pass" if success else "fail",
            message="Connected" if success else error,
            details=details if isinstance(details, dict) else {"error": str(details)},
        )
    )
    if not success:
        has_failure = True

    # 2. Redis Check
    try:
        res = check_redis(config)
        if len(res) == 3:
            success, details, error = res
        else:
            success, error = res[0], res[-1]
    except Exception as e:
        success, error = False, str(e)

    checks.append(
        DoctorCheckResult(
            name="Redis",
            status="pass" if success else "fail",
            message="Connected" if success else error,
        )
    )
    if not success:
        has_failure = True

    # 3. Embeddings Check
    try:
        if config.core and config.core.provider:
            res = _probe_embeddings(config.core.provider)
            if len(res) == 2:
                success, dim = res
            else:
                success = res[0]
                dim = res[1] if success else -1
        else:
            success, dim = False, -1
    except Exception:
        success, dim = False, -1

    checks.append(
        DoctorCheckResult(
            name="Embeddings API",
            status="pass" if success else "fail",
            message=f"Dimension: {dim}" if success else "Probe failed",
            details={"dimension": dim},
        )
    )
    if not success:
        has_failure = True

    # 4. Reranker Check
    try:
        res = check_reranker(config)
        if len(res) == 3:
            success, details, error = res
        else:
            success, error = res[0], res[-1]
    except Exception as e:
        success, error = False, str(e)

    checks.append(
        DoctorCheckResult(
            name="Reranker API",
            status="pass" if success else "fail",
            message="Connected" if success else error,
        )
    )
    # Reranker optional? Treat as warn if fails? For now fail.
    if not success:
        has_failure = True

    return DoctorReport(
        overall_status="unhealthy" if has_failure else "healthy", checks=checks
    )
