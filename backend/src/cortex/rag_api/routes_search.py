"""
Search API Routes.

Implements §9.2 of the Canonical Blueprint.
"""

from __future__ import annotations

import hashlib
import logging
import time

from cortex.audit import log_audit_event
from cortex.common.exceptions import (
    CortexError,
)
from cortex.common.exceptions import ValidationError as CortexValidationError
from cortex.config.loader import get_config
from cortex.context import tenant_id_ctx, user_id_ctx
from cortex.observability import trace_operation  # P2 Fix: Enable tracing
from cortex.rag_api.models import SearchRequest, SearchResponse
from cortex.retrieval.hybrid_search import KBSearchInput, tool_kb_search_hybrid
from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/search", response_model=SearchResponse)
@trace_operation("api_search")  # P2 Fix: Enable request tracing
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
    tenant_id = tenant_id_ctx.get("default") or "default"
    user_id = user_id_ctx.get("anonymous") or "anonymous"

    try:
        start_time = time.perf_counter()
        query = request.query
        if not isinstance(query, str) or not query.strip():
            raise HTTPException(status_code=400, detail="Query is required")

        config = get_config()
        max_k = config.search.k
        requested_k = request.k if request.k is not None else max_k
        safe_k = min(requested_k, max_k)

        # Map API request to tool input
        tool_input = KBSearchInput(
            tenant_id=tenant_id,
            user_id=user_id,
            query=query,
            k=safe_k,
            filters=request.filters,
            fusion_method=request.fusion_method,
        )

        # Call retrieval tool
        result_wrapper = await tool_kb_search_hybrid(tool_input)

        if result_wrapper.is_err():
            tool_error = result_wrapper.unwrap_err()
            if isinstance(tool_error, CortexError):
                raise tool_error
            logger.error("Search tool returned non-Cortex error: %s", tool_error)
            raise CortexError("Search failed")

        results = result_wrapper.unwrap()
        results_list = results.results if results and results.results else []

        query_time_ms = (time.perf_counter() - start_time) * 1000

        # Audit log
        try:
            input_str = request.model_dump_json()
            input_hash = hashlib.sha256(input_str.encode()).hexdigest()

            # Safely get result count
            res_count = len(results_list)

            log_audit_event(
                tenant_id=tenant_id,
                user_or_agent=user_id,
                action="search",
                input_hash=input_hash,
                risk_level="low",
                correlation_id=correlation_id,
                metadata={
                    "query_hash": hashlib.sha256(
                        query.encode()
                    ).hexdigest(),  # PII mask
                    "result_count": res_count,
                    "query_time_ms": query_time_ms,
                },
            )
        except Exception:
            logger.exception("Audit logging failed")

        # Convert SearchResults to list of dicts for response
        results_dicts = [r.model_dump() for r in results_list]

        return SearchResponse(
            correlation_id=correlation_id,
            results=results_dicts,
            total_count=len(results_dicts),
            query_time_ms=query_time_ms,
        )

    except CortexError as e:
        if isinstance(e, CortexValidationError):
            raise HTTPException(status_code=400, detail=e.to_dict())
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Search failed")
        # Do not leak internal exception details to client
        raise HTTPException(status_code=500, detail="Internal Server Error")
