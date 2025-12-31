import asyncio
import logging
import os
from typing import Any, Literal

from cortex.config.loader import get_config
from cortex.health import (
    DoctorCheckResult,
    check_postgres,
    check_redis,
    check_reranker,
    probe_embeddings,
)
from cortex.security.auth import require_admin
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)
logger = logging.getLogger(__name__)


OverallStatus = Literal["healthy", "degraded", "unhealthy"]


class DoctorReport(BaseModel):
    overall_status: OverallStatus
    checks: list[DoctorCheckResult]


@router.get("/config")
async def get_config_info() -> dict[str, Any]:
    """Get sanitized safe configuration."""
    try:
        config = get_config()
    except Exception as exc:
        logger.exception("Admin config lookup failed")
        raise HTTPException(
            status_code=500, detail="Configuration unavailable"
        ) from exc
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
    try:
        config = get_config()
    except Exception as exc:
        logger.exception("Doctor config lookup failed")
        raise HTTPException(
            status_code=500, detail="Configuration unavailable"
        ) from exc

    # Run checks concurrently
    checks = [
        ("PostgreSQL", check_postgres),
        ("Redis", check_redis),
        ("Embeddings API", probe_embeddings),
        ("Reranker API", check_reranker),
    ]
    tasks = [check(config) for _, check in checks]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[DoctorCheckResult] = []
    for (check_name, _), result in zip(checks, raw_results, strict=False):
        if isinstance(result, Exception):
            logger.error("Doctor check %s failed: %s", check_name, result)
            results.append(
                DoctorCheckResult(
                    name=check_name,
                    status="fail",
                    message="Check failed",
                    details={"error": str(result)},
                )
            )
        elif isinstance(result, DoctorCheckResult):
            results.append(result)
        else:
            results.append(
                DoctorCheckResult(
                    name=check_name,
                    status="fail",
                    message="Unexpected check result",
                    details={"result_type": type(result).__name__},
                )
            )

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
