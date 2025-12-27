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
from cortex.rag_api.models import DraftEmailRequest, DraftEmailResponse
from fastapi import APIRouter, Depends, HTTPException, Request

logger = logging.getLogger(__name__)
router = APIRouter()


def get_draft_graph(http_request: Request):
    """
    Get pre-compiled graph from app.state.

    This is intended to be used as a FastAPI dependency.
    """
    try:
        return http_request.app.state.graphs["draft"]
    except (AttributeError, KeyError) as e:
        logger.exception("Draft graph not found in app.state.graphs. Check lifespan configuration.")
        raise HTTPException(
            status_code=500,
            detail="Internal server error: Draft service not configured",
        ) from e


@router.post("/draft-email", response_model=DraftEmailResponse)
@trace_operation("api_draft_email")  # P2 Fix: Enable request tracing
async def draft_endpoint(
    request: DraftEmailRequest,
    http_request: Request,
    graph=Depends(get_draft_graph),
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

        # Invoke graph
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
