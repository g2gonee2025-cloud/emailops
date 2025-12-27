"""
Summarize Thread API Routes.

Implements §9.5 of the Canonical Blueprint.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging

from cortex.audit import log_audit_event
from cortex.context import tenant_id_ctx, user_id_ctx
from cortex.observability import trace_operation  # P2 Fix: Enable tracing
from cortex.orchestration.graphs import build_summarize_graph
from cortex.rag_api.models import SummarizeThreadRequest, SummarizeThreadResponse
from cortex.security.dependencies import get_current_user
from fastapi import APIRouter, Depends, HTTPException, Request

logger = logging.getLogger(__name__)
router = APIRouter()

# Lazily compile graph to avoid startup cost (fallback if app.state.graphs not available)

_summarize_graph = None
_graph_lock = asyncio.Lock()


async def get_summarize_graph(http_request: Request = None):
    """Get pre-compiled graph from app.state or lazy load as fallback in thread-safe way."""
    if http_request and hasattr(http_request.app.state, "graphs"):
        cached = http_request.app.state.graphs.get("summarize")
        if cached:
            return cached

    global _summarize_graph
    if _summarize_graph is None:
        async with _graph_lock:
            if _summarize_graph is None:
                logger.info(
                    "Lazily Initializing Summarize Graph (prefer app.state.graphs)..."
                )
                _summarize_graph = build_summarize_graph().compile()
    return _summarize_graph


@router.post("/summarize", response_model=SummarizeThreadResponse)
@trace_operation("api_summarize")
async def summarize_thread_endpoint(
    request: SummarizeThreadRequest,
    http_request: Request,
    _: str = Depends(get_current_user),
) -> SummarizeThreadResponse:
    # ... (docstring same)
    correlation_id = getattr(http_request.state, "correlation_id", None)

    try:
        # ... (setup same until graph invoke)
        initial_state = {
            "tenant_id": tenant_id_ctx.get(),
            "user_id": user_id_ctx.get(),
            "thread_id": request.thread_id,
            "max_length": request.max_length,
            "iteration_count": 0,
            "error": None,
            "correlation_id": correlation_id,
        }

        graph = await get_summarize_graph(http_request)
        final_state = await graph.ainvoke(initial_state)

        if final_state.get("error"):
            # P2 Fix: Sanitize error
            logger.error(f"Graph execution error: {final_state['error']}")
            raise HTTPException(status_code=500, detail="Summarization workflow failed")

        summary = final_state.get("summary")
        if not summary:
            raise HTTPException(status_code=500, detail="No summary generated")

        try:
            input_hash = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
            log_audit_event(
                tenant_id=tenant_id_ctx.get(),
                user_or_agent=user_id_ctx.get(),
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
    except Exception:
        logger.exception("Summarize API failed")
        raise HTTPException(status_code=500, detail="Internal Server Error")
