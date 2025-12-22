"""
Draft API Routes.

Implements §9.2 of the Canonical Blueprint.
"""

from __future__ import annotations

import hashlib
import logging

from cortex.audit import log_audit_event
from cortex.common.exceptions import CortexError
from cortex.context import tenant_id_ctx, user_id_ctx
from cortex.observability import trace_operation  # P2 Fix: Enable tracing
from cortex.orchestration.graphs import build_draft_graph
from cortex.rag_api.models import DraftEmailRequest, DraftEmailResponse
from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)
router = APIRouter()

# Lazy load graph (fallback if app.state.graphs not available)
_draft_graph = None


def get_draft_graph(http_request: Request = None):
    """Get pre-compiled graph from app.state or lazy load as fallback."""
    # P0 Fix: Prefer pre-compiled graph from lifespan
    if http_request and hasattr(http_request.app.state, "graphs"):
        cached = http_request.app.state.graphs.get("draft")
        if cached:
            return cached

    # Fallback to lazy loading
    global _draft_graph
    if _draft_graph is None:
        logger.info("Lazily initializing Draft Graph (prefer app.state.graphs)...")
        _draft_graph = build_draft_graph().compile()
    return _draft_graph


@router.post("/draft-email", response_model=DraftEmailResponse)
@trace_operation("api_draft_email")  # P2 Fix: Enable request tracing
async def draft_endpoint(
    request: DraftEmailRequest, http_request: Request
) -> DraftEmailResponse:
    """
    Draft email endpoint.

    Blueprint §9.2:
    * POST /api/v1/draft-email
    * Request: DraftEmailRequest
    * Response: DraftEmailResponse with correlation_id
    """
    # Get correlation_id from request state (set by middleware)
    correlation_id = getattr(http_request.state, "correlation_id", None)

    try:
        # Initialize state
        initial_state = {
            "tenant_id": tenant_id_ctx.get(),
            "user_id": user_id_ctx.get(),
            "thread_id": str(request.thread_id) if request.thread_id else None,
            "tone": request.tone,
            "reply_to_message_id": request.reply_to_message_id,
            "explicit_query": request.instruction,
            "draft_query": None,
            "retrieval_results": None,
            "assembled_context": None,
            "draft": None,
            "critique": None,
            "iteration_count": 0,
            "error": None,
            "correlation_id": correlation_id,
        }

        # Invoke graph - use pre-compiled from app.state if available
        graph = get_draft_graph(http_request)
        final_state = await graph.ainvoke(initial_state)

        # Check for errors
        if final_state.get("error"):
            raise HTTPException(status_code=500, detail=final_state["error"])

        # Return draft
        draft = final_state.get("draft")
        if not draft:
            raise HTTPException(status_code=500, detail="No draft generated")

        # Audit log - P2 Fix: Use context vars (authoritative) not request body
        try:
            input_str = request.model_dump_json()
            input_hash = hashlib.sha256(input_str.encode()).hexdigest()

            log_audit_event(
                tenant_id=tenant_id_ctx.get(),  # P2 Fix: Use context
                user_or_agent=user_id_ctx.get(),  # P2 Fix: Use context
                action="draft_email",
                input_hash=input_hash,
                risk_level="medium",
                correlation_id=correlation_id,
                metadata={
                    "query": request.instruction,
                    "thread_id": str(request.thread_id) if request.thread_id else None,
                    "iterations": final_state.get("iteration_count", 0),
                },
            )
        except Exception as audit_err:
            logger.error(f"Audit logging failed: {audit_err}")

        return DraftEmailResponse(
            correlation_id=correlation_id,
            draft=draft,
            iterations=final_state.get("iteration_count", 0),
        )

    except CortexError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Draft failed")
        raise HTTPException(status_code=500, detail=str(e))
