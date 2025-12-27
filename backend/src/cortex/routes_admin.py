import asyncio
import logging
import os
from typing import Any

from cortex.config.loader import get_config
from cortex.health import (
    DoctorCheckResult,
    check_postgres,
    check_redis,
    check_reranker,
    probe_embeddings,
)
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/admin", tags=["admin"])
logger = logging.getLogger(__name__)


class DoctorReport(BaseModel):
    overall_status: str  # "healthy", "degraded", "unhealthy"
    checks: list[DoctorCheckResult]


@router.get("/config")
async def get_config_info() -> dict[str, Any]:
    """Get sanitized safe configuration."""
    config = get_config()
    # Safely access nested attributes
    core_config = getattr(config, "core", None)
    system_config = getattr(config, "system", None)
    db_config = getattr(config, "database", None)

    return {
        "environment": getattr(core_config, "env", None),
        "provider": getattr(core_config, "provider", None),
        "log_level": getattr(system_config, "log_level", None),
        "database_url": (
            "***" if db_config and getattr(db_config, "url", None) else None
        ),
    }


@router.get("/status")
async def get_system_status() -> dict[str, Any]:
    """Get simple system status info."""
    return {
        "status": "online",
        "service": "cortex-backend",
        "env": os.getenv("OUTLOOKCORTEX_ENV", "unknown"),
    }


@router.post("/doctor", response_model=DoctorReport)
async def run_doctor() -> DoctorReport:
    """Run system diagnostics."""
    config = get_config()

    # Run checks concurrently
    tasks = [
        check_postgres(config),
        check_redis(config),
        probe_embeddings(config),
        check_reranker(config),
    ]
    results: list[DoctorCheckResult] = await asyncio.gather(*tasks)

    # Determine overall status
    has_failure = any(check.status == "fail" for check in results)
    has_warning = any(check.status == "warn" for check in results)

    if has_failure:
        overall_status = "unhealthy"
    elif has_warning:
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    return DoctorReport(overall_status=overall_status, checks=results)
