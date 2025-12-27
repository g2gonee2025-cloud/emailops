import logging
import os
from typing import Any, Dict, List

from cortex.config.loader import get_config
from fastapi import APIRouter
from pydantic import BaseModel

# Import new health check functions
from cortex.health import (
    CheckResult,
    check_embeddings,
    check_postgres,
    check_redis,
    check_reranker,
)

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


def _to_doctor_check(result: CheckResult) -> DoctorCheckResult:
    """Convert a CheckResult to a DoctorCheckResult for the API response."""
    status = "pass" if result.success else "fail"
    message = result.error
    if result.success:
        if result.details and "message" in result.details:
            message = result.details["message"]
        elif result.name == "Embeddings":
            dim = result.details.get("dimension") if result.details else "N/A"
            message = f"Dimension: {dim}"
        else:
            message = "Connected"

    return DoctorCheckResult(
        name=result.name,
        status=status,
        message=message,
        details=result.details,
    )


@router.post("/doctor", response_model=DoctorReport)
async def run_doctor() -> DoctorReport:
    """Run system diagnostics."""
    config = get_config()
    check_results: List[CheckResult] = []
    has_failure = False

    # Define all checks to be run
    all_checks = [
        check_postgres,
        check_redis,
        check_embeddings,
        check_reranker,
    ]

    # Execute checks and collect results
    for check_func in all_checks:
        try:
            result = check_func(config)
            check_results.append(result)
            if not result.success:
                has_failure = True
        except Exception as e:
            # This is a fallback for unexpected errors within the check function itself
            check_results.append(
                CheckResult(name=check_func.__name__, success=False, error=str(e))
            )
            has_failure = True

    # Convert to API model
    api_checks = [_to_doctor_check(res) for res in check_results]

    return DoctorReport(
        overall_status="unhealthy" if has_failure else "healthy", checks=api_checks
    )
