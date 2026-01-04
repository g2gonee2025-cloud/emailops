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
    import time
    start_time = time.perf_counter()
    logger.info("Admin config request started")

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

    result = {
        "environment": getattr(core_config, "env", None),
        "provider": getattr(core_config, "provider", None),
        "log_level": getattr(system_config, "log_level", None),
        "database_url": (
            "***" if db_config and getattr(db_config, "url", None) else None
        ),
    }

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Admin config request completed, environment=%s, elapsed_ms=%.2f",
        result.get("environment"),
        elapsed_ms,
    )

    return result


@router.get("/status")
async def get_system_status() -> dict[str, Any]:
    """Get simple system status info."""
    import time
    start_time = time.perf_counter()
    logger.info("System status request started")

    result = {
        "status": "online",
        "service": "cortex-backend",
        "env": os.getenv("OUTLOOKCORTEX_ENV", "unknown"),
    }

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "System status request completed, status=%s, env=%s, elapsed_ms=%.2f",
        result["status"],
        result["env"],
        elapsed_ms,
    )

    return result


@router.post("/doctor", response_model=DoctorReport)
async def run_doctor() -> DoctorReport:
    """Run system diagnostics."""
    import time
    start_time = time.perf_counter()
    logger.info("System diagnostics (doctor) started")

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
    logger.debug("Running %d diagnostic checks: %s", len(checks), [c[0] for c in checks])

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
            logger.debug(
                "Doctor check %s completed: status=%s, message=%s",
                check_name,
                result.status,
                result.message,
            )
            results.append(result)
        else:
            logger.warning(
                "Doctor check %s returned unexpected result type: %s",
                check_name,
                type(result).__name__,
            )
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

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    pass_count = sum(1 for r in results if r.status == "pass")
    fail_count = sum(1 for r in results if r.status == "fail")
    warn_count = sum(1 for r in results if r.status == "warn")

    logger.info(
        "System diagnostics completed: overall_status=%s, pass=%d, fail=%d, warn=%d, elapsed_ms=%.2f",
        overall_status,
        pass_count,
        fail_count,
        warn_count,
        elapsed_ms,
    )

    return DoctorReport(overall_status=overall_status, checks=results)
