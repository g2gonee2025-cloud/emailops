"""System health and diagnostic checks."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from cortex.config.loader import EmailOpsConfig
from pydantic import BaseModel

logger = logging.getLogger(__name__)
DEFAULT_REDIS_URL = "redis://localhost:6379"


class DoctorCheckResult(BaseModel):
    """Represents the result of a single diagnostic check."""

    name: str
    status: str  # "pass", "fail", "warn"
    message: str | None = None
    details: Dict[str, Any] | None = None


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

        engine = create_engine(config.database.url)
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
            config.redis.url
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
                name="Embeddings API",
                status="fail",
                message="Embedding provider not configured",
            )

        result = embed_texts(["test"])
        if result is not None and len(result) > 0:
            dim = result.shape[1] if hasattr(result, "shape") else len(result[0])
            return DoctorCheckResult(
                name="Embeddings API",
                status="pass",
                message=f"Dimension: {dim}",
                details={"dimension": dim},
            )
        else:
            return DoctorCheckResult(
                name="Embeddings API",
                status="fail",
                message="Probe returned empty result",
            )
    except Exception as e:
        logger.error("Embedding probe failed: %s", e, exc_info=True)
        return DoctorCheckResult(
            name="Embeddings API", status="fail", message=f"Probe failed: {e}"
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
                name="Reranker API",
                status="pass",
                message="Not configured",
                details={"reason": "No reranker endpoint in config."},
            )

        health_url = f"{reranker_endpoint.rstrip('/')}/health"
        async with httpx.AsyncClient() as client:
            resp = await client.get(health_url, timeout=5.0)

            if resp.status_code == 200:
                return DoctorCheckResult(
                    name="Reranker API", status="pass", message="Connected"
                )
            else:
                return DoctorCheckResult(
                    name="Reranker API",
                    status="warn",
                    message=f"Reranker returned status {resp.status_code}",
                )
    except ImportError:
        return DoctorCheckResult(
            name="Reranker API",
            status="warn",
            message="httpx not installed (pip install httpx)",
        )
    except httpx.ConnectError:
        return DoctorCheckResult(
            name="Reranker API",
            status="warn",
            message=f"Cannot connect to reranker at {reranker_endpoint}",
        )
    except httpx.TimeoutException:
        return DoctorCheckResult(
            name="Reranker API",
            status="warn",
            message=f"Reranker timeout at {reranker_endpoint}",
        )
    except Exception as e:
        logger.warning("Reranker health check failed: %s", e)
        return DoctorCheckResult(
            name="Reranker API", status="warn", message=f"Connection failed: {e}"
        )
