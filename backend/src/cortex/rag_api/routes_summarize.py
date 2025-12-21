"""
Summarize Thread API Routes.

Implements §9.5 of the Canonical Blueprint.
"""

from __future__ import annotations

import hashlib
import logging

from cortex.audit import log_audit_event
from cortex.context import tenant_id_ctx, user_id_ctx
from cortex.observability import trace_operation  # P2 Fix: Enable tracing
from cortex.orchestration.graphs import build_summarize_graph
from cortex.rag_api.models import SummarizeThreadRequest, SummarizeThreadResponse
from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)
router = APIRouter()

# Lazily compile graph to avoid startup cost (fallback if app.state.graphs not available)
_summarize_graph = None


def get_summarize_graph(http_request: Request = None):
    """Get pre-compiled graph from app.state or lazy load as fallback."""
    # P0 Fix: Prefer pre-compiled graph from lifespan
    if http_request and hasattr(http_request.app.state, "graphs"):
        cached = http_request.app.state.graphs.get("summarize")
        if cached:
            return cached

    # Fallback to lazy loading
    global _summarize_graph
    if _summarize_graph is None:
        logger.info("Lazily Initializing Summarize Graph (prefer app.state.graphs)...")
        _summarize_graph = build_summarize_graph().compile()
    return _summarize_graph


@router.post("/summarize", response_model=SummarizeThreadResponse)
@trace_operation("api_summarize")  # P2 Fix: Enable request tracing
async def summarize_thread_endpoint(
    request: SummarizeThreadRequest, http_request: Request
) -> SummarizeThreadResponse:
    """
    Summarize an email thread.

    Blueprint §9.5:
    * POST /api/v1/summarize
    * Request: SummarizeThreadRequest
    * Response: SummarizeThreadResponse
    """
    correlation_id = getattr(http_request.state, "correlation_id", None)

    try:
        initial_state = {
            "tenant_id": tenant_id_ctx.get(),
            "user_id": user_id_ctx.get(),
            "thread_id": request.thread_id,
            "max_length": request.max_length,
            "iteration_count": 0,
            "error": None,
            "correlation_id": correlation_id,
        }

        # Use pre-compiled graph from app.state if available
        graph = get_summarize_graph(http_request)
        final_state = await graph.ainvoke(initial_state)

        if final_state.get("error"):
            raise HTTPException(status_code=500, detail=final_state["error"])

        summary = final_state.get("summary")
        if not summary:
            raise HTTPException(status_code=500, detail="No summary generated")

        # P2 Fix: Use context vars (authoritative) not request body
        try:
            input_hash = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
            log_audit_event(
                tenant_id=tenant_id_ctx.get(),  # P2 Fix: Use context
                user_or_agent=user_id_ctx.get(),  # P2 Fix: Use context
                action="summarize_thread",
                input_hash=input_hash,
                risk_level="low",
                correlation_id=correlation_id,
                metadata={"thread_id": request.thread_id},
            )
        except Exception as audit_err:
            logger.error(f"Audit logging failed: {audit_err}")

        return SummarizeThreadResponse(
            correlation_id=correlation_id,
            summary=summary,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Summarize API failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
