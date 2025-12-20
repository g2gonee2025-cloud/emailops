"""
Search API Routes.

Implements §9.2 of the Canonical Blueprint.
"""

from __future__ import annotations

import hashlib
import logging
import time

from cortex.audit import log_audit_event
from cortex.common.exceptions import CortexError
from cortex.context import tenant_id_ctx, user_id_ctx
from cortex.rag_api.models import SearchRequest, SearchResponse
from cortex.retrieval.hybrid_search import KBSearchInput, tool_kb_search_hybrid
from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_endpoint(
    request: SearchRequest, http_request: Request
) -> SearchResponse:
    """
    Search endpoint.

    Blueprint §9.2:
    * POST /api/v1/search
    * Request: SearchRequest
    * Response: SearchResponse with correlation_id
    """
    # Get correlation_id from request state (set by middleware)
    correlation_id = getattr(http_request.state, "correlation_id", None)

    try:
        start_time = time.perf_counter()

        # Map API request to tool input
        tool_input = KBSearchInput(
            tenant_id=tenant_id_ctx.get(),
            user_id=user_id_ctx.get(),
            query=request.query,
            k=request.k,
            filters=request.filters,
        )

        # Call retrieval tool
        from fastapi.concurrency import run_in_threadpool

        results = await run_in_threadpool(tool_kb_search_hybrid, tool_input)

        query_time_ms = (time.perf_counter() - start_time) * 1000

        # Audit log
        try:
            input_str = request.model_dump_json()
            input_hash = hashlib.sha256(input_str.encode()).hexdigest()

            log_audit_event(
                tenant_id=tenant_id_ctx.get(),
                user_or_agent=user_id_ctx.get(),
                action="search",
                input_hash=input_hash,
                risk_level="low",
                correlation_id=correlation_id,
                metadata={
                    "query": request.query,
                    "result_count": len(results.results),
                    "query_time_ms": query_time_ms,
                },
            )
        except Exception as audit_err:
            logger.error(f"Audit logging failed: {audit_err}")

        # Convert SearchResults to list of dicts for response
        results_dicts = (
            [r.model_dump() for r in results.results] if results.results else []
        )

        return SearchResponse(
            correlation_id=correlation_id,
            results=results_dicts,
            total_count=len(results_dicts),
            query_time_ms=query_time_ms,
        )

    except CortexError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))
