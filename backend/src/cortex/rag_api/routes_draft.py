"""
Draft API Routes.

Implements §9.2 of the Canonical Blueprint.
"""
from __future__ import annotations

import hashlib
import logging

from cortex.audit import log_audit_event
from cortex.common.exceptions import CortexError
from cortex.models.api import DraftEmailRequest, DraftEmailResponse
from cortex.orchestration.graphs import build_draft_graph
from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)
router = APIRouter()

# Compile graph once
_draft_graph = build_draft_graph().compile()


@router.post("/draft-email", response_model=DraftEmailResponse)
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
            "tenant_id": request.tenant_id,
            "user_id": request.user_id,
            "thread_id": str(request.thread_id) if request.thread_id else None,
            "explicit_query": request.query,
            "draft_query": None,
            "retrieval_results": None,
            "assembled_context": None,
            "draft": None,
            "critique": None,
            "iteration_count": 0,
            "error": None,
            "_correlation_id": correlation_id,
        }

        # Invoke graph
        final_state = await _draft_graph.ainvoke(initial_state)

        # Check for errors
        if final_state.get("error"):
            raise HTTPException(status_code=500, detail=final_state["error"])

        # Return draft
        draft = final_state.get("draft")
        if not draft:
            raise HTTPException(status_code=500, detail="No draft generated")

        # Audit log
        try:
            input_str = request.model_dump_json()
            input_hash = hashlib.sha256(input_str.encode()).hexdigest()

            log_audit_event(
                tenant_id=request.tenant_id,
                user_or_agent=request.user_id,
                action="draft_email",
                input_hash=input_hash,
                risk_level="medium",
                correlation_id=correlation_id,
                metadata={
                    "query": request.query,
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
