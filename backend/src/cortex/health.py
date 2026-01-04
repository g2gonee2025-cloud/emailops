"""System health and diagnostic checks."""

from __future__ import annotations

import logging
import os
from typing import Any

from cortex.config.loader import EmailOpsConfig
from pydantic import BaseModel

logger = logging.getLogger(__name__)
DEFAULT_REDIS_URL = "redis://localhost:6379"
EMBEDDINGS_API_NAME = "Embeddings API"
RERANKER_API_NAME = "Reranker API"


class DoctorCheckResult(BaseModel):
    """Represents the result of a single diagnostic check."""

    name: str
    status: str  # "pass", "fail", "warn"
    message: str | None = None
    details: dict[str, Any] | None = None


class ComponentHealth(BaseModel):
    """Health status for a single component in deep health check."""

    status: str  # "healthy" | "unhealthy"
    latency_ms: float | None = None
    error: str | None = None


class DeepHealthResponse(BaseModel):
    """Response schema for deep health check endpoint."""

    status: str  # "healthy" | "degraded" | "unhealthy"
    components: dict[str, ComponentHealth]


async def check_postgres(config: EmailOpsConfig) -> DoctorCheckResult:
    """Check PostgreSQL connectivity."""
    try:
        from sqlalchemy import create_engine, text

        if not config.database or not config.database.url:
            return DoctorCheckResult(
                name="PostgreSQL",
                status="fail",
                message="Database URL not configured",
            )

        engine = create_engine(str(config.database.url))
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return DoctorCheckResult(name="PostgreSQL", status="pass", message="Connected")
    except ImportError:
        return DoctorCheckResult(
            name="PostgreSQL", status="fail", message="sqlalchemy is not installed"
        )
    except Exception as e:
        logger.error("PostgreSQL health check failed: %s", e)
        return DoctorCheckResult(
            name="PostgreSQL", status="fail", message=f"Connection failed: {e}"
        )


async def check_redis(config: EmailOpsConfig) -> DoctorCheckResult:
    """Check Redis connectivity."""
    try:
        import redis

        redis_url = (
            str(config.redis.url)
            if config.redis and config.redis.url
            else os.getenv("OUTLOOKCORTEX_REDIS_URL", DEFAULT_REDIS_URL)
        )
        if not redis_url:
            return DoctorCheckResult(
                name="Redis", status="fail", message="Redis URL not configured"
            )

        r = redis.from_url(redis_url)
        r.ping()
        return DoctorCheckResult(name="Redis", status="pass", message="Connected")
    except ImportError:
        return DoctorCheckResult(
            name="Redis", status="fail", message="redis-py is not installed"
        )
    except Exception as e:
        logger.error("Redis health check failed: %s", e)
        return DoctorCheckResult(
            name="Redis", status="fail", message=f"Connection failed: {e}"
        )


async def probe_embeddings(config: EmailOpsConfig) -> DoctorCheckResult:
    """Test embedding functionality with the configured provider."""
    try:
        from cortex.llm.client import embed_texts

        if not config.core or not config.core.provider:
            return DoctorCheckResult(
                name=EMBEDDINGS_API_NAME,
                status="fail",
                message="Embedding provider not configured",
            )

        result = embed_texts(["test"])
        if result is not None and len(result) > 0:
            dim = result.shape[1] if hasattr(result, "shape") else len(result[0])
            return DoctorCheckResult(
                name=EMBEDDINGS_API_NAME,
                status="pass",
                message=f"Dimension: {dim}",
                details={"dimension": dim},
            )
        else:
            return DoctorCheckResult(
                name=EMBEDDINGS_API_NAME,
                status="fail",
                message="Probe returned empty result",
            )
    except Exception as e:
        logger.error("Embedding probe failed: %s", e, exc_info=True)
        return DoctorCheckResult(
            name=EMBEDDINGS_API_NAME, status="fail", message=f"Probe failed: {e}"
        )


async def check_reranker(config: EmailOpsConfig) -> DoctorCheckResult:
    """
    Check reranker endpoint connectivity.

    This check is considered optional. A failure will result in a "warn" status.
    """
    try:
        import httpx

        reranker_endpoint = getattr(config.search, "reranker_endpoint", None)
        if not reranker_endpoint:
            return DoctorCheckResult(
                name=RERANKER_API_NAME,
                status="pass",
                message="Not configured",
                details={"reason": "No reranker endpoint in config."},
            )

        health_url = f"{reranker_endpoint.rstrip('/')}/health"
        async with httpx.AsyncClient() as client:
            resp = await client.get(health_url, timeout=5.0)

            if resp.status_code == 200:
                return DoctorCheckResult(
                    name=RERANKER_API_NAME, status="pass", message="Connected"
                )
            else:
                return DoctorCheckResult(
                    name=RERANKER_API_NAME,
                    status="warn",
                    message=f"Reranker returned status {resp.status_code}",
                )
    except ImportError:
        return DoctorCheckResult(
            name=RERANKER_API_NAME,
            status="warn",
            message="httpx not installed (pip install httpx)",
        )
    except httpx.ConnectError:
        return DoctorCheckResult(
            name=RERANKER_API_NAME,
            status="warn",
            message=f"Cannot connect to reranker at {reranker_endpoint}",
        )
    except httpx.TimeoutException:
        return DoctorCheckResult(
            name=RERANKER_API_NAME,
            status="warn",
            message=f"Reranker timeout at {reranker_endpoint}",
        )
    except Exception as e:
        logger.warning("Reranker health check failed: %s", e)
        return DoctorCheckResult(
            name=RERANKER_API_NAME, status="warn", message=f"Connection failed: {e}"
        )
