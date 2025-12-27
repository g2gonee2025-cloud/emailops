"""
Draft API Routes.

Implements §9.2 of the Canonical Blueprint.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict

from cortex.audit import log_audit_event
from cortex.common.exceptions import CortexError
from cortex.context import tenant_id_ctx, user_id_ctx
from cortex.observability import trace_operation
from cortex.rag_api.models import DraftEmailRequest, DraftEmailResponse
from cortex.security.dependencies import get_current_user
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
@trace_operation("api_draft_email")
async def draft_endpoint(
    request: DraftEmailRequest,
    http_request: Request,
    graph=Depends(get_draft_graph),
    current_user: Dict[str, Any] = Depends(get_current_user),
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

        # Check for errors from the graph execution
        if final_state.get("error"):
            logger.error(
                f"Draft generation failed with error: {final_state['error']}",
                extra={"correlation_id": correlation_id},
            )
            raise HTTPException(
                status_code=500, detail="Internal server error: Failed to generate draft"
            )

        # Return draft
        draft = final_state.get("draft")
        if not draft:
            logger.error(
                "Draft generation completed without a draft.",
                extra={"correlation_id": correlation_id},
            )
            raise HTTPException(
                status_code=500, detail="Internal server error: No draft was generated"
            )

        # Audit log
        try:
            # Create a PII-free representation of the input for hashing.
            # Hashing the raw instruction would be a PII leak.
            pii_free_input = {
                "thread_id": str(request.thread_id) if request.thread_id else None,
                "reply_to_message_id": request.reply_to_message_id,
                "tone": request.tone,
            }
            input_str = str(pii_free_input)
            input_hash = hashlib.sha256(input_str.encode()).hexdigest()

            log_audit_event(
                tenant_id=tenant_id_ctx.get(),
                user_or_agent=user_id_ctx.get(),
                action="draft_email",
                input_hash=input_hash,
                risk_level="medium",
                correlation_id=correlation_id,
                metadata={
                    "query": "[REDACTED]",  # Redact PII
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
        # Handle domain-specific errors with a 400 status code
        raise HTTPException(status_code=400, detail=e.to_dict())
    except HTTPException:
        # Re-raise exceptions we've already handled
        raise
    except Exception:
        # Obfuscate unexpected errors to prevent implementation leakage
        logger.exception("Draft failed due to an unexpected error")
        raise HTTPException(
            status_code=500, detail="Internal server error"
        )
