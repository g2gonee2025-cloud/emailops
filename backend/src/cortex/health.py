from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict

# Library-safe logger
logger = logging.getLogger(__name__)


DEFAULT_REDIS_URL = "redis://localhost:6379"


@dataclass(frozen=True)
class CheckResult:
    """Standardized result for a single health check."""

    name: str
    success: bool
    details: Dict[str, Any] | None = None
    error: str | None = None


def check_postgres(config: Any) -> CheckResult:
    """Check PostgreSQL connectivity."""
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(config.database.url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return CheckResult(
            name="PostgreSQL", success=True, details={"message": "Connected"}
        )
    except Exception as e:
        return CheckResult(name="PostgreSQL", success=False, error=str(e))


def check_redis(config: Any) -> CheckResult:
    """Check Redis connectivity."""
    try:
        import redis

        redis_url = getattr(config.cache, "url", DEFAULT_REDIS_URL)
        r = redis.from_url(redis_url)
        r.ping()
        return CheckResult(name="Redis", success=True, details={"message": "Connected"})
    except Exception as e:
        return CheckResult(name="Redis", success=False, error=str(e))


def check_embeddings(config: Any) -> CheckResult:
    """Test embedding functionality with the configured provider."""
    try:
        from cortex.llm.client import embed_texts

        result = embed_texts(["test"])
        if result is not None and len(result) > 0:
            dim = result.shape[1] if hasattr(result, "shape") else len(result[0])
            return CheckResult(
                name="Embeddings", success=True, details={"dimension": dim}
            )
        return CheckResult(
            name="Embeddings", success=False, error="Probe returned empty result"
        )
    except Exception as e:
        logger.warning("Embedding probe failed: %s", e)
        return CheckResult(name="Embeddings", success=False, error=str(e))


def check_reranker(config: Any) -> CheckResult:
    """Check reranker endpoint connectivity."""
    try:
        import httpx

        reranker_endpoint = getattr(config.search, "reranker_endpoint", None)
        if not reranker_endpoint:
            return CheckResult(
                name="Reranker",
                success=False,
                error="No reranker endpoint configured",
            )

        health_url = f"{reranker_endpoint.rstrip('/')}/health"
        try:
            resp = httpx.get(health_url, timeout=5.0)
            if resp.status_code == 200:
                return CheckResult(
                    name="Reranker", success=True, details={"message": "Connected"}
                )
            return CheckResult(
                name="Reranker",
                success=False,
                error=f"Reranker returned status {resp.status_code}",
            )
        except httpx.ConnectError:
            return CheckResult(
                name="Reranker",
                success=False,
                error=f"Cannot connect to reranker at {reranker_endpoint}",
            )
        except httpx.TimeoutException:
            return CheckResult(
                name="Reranker",
                success=False,
                error=f"Reranker timeout at {reranker_endpoint}",
            )
    except ImportError:
        return CheckResult(name="Reranker", success=False, error="httpx not installed")
    except Exception as e:
        return CheckResult(name="Reranker", success=False, error=str(e))
