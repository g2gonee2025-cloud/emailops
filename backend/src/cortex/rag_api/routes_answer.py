"""
RAG API: Answer routes.

Implements §9.1 of the Canonical Blueprint.
"""
from __future__ import annotations

import hashlib
import logging

from cortex.audit import log_audit_event
from cortex.models.api import AnswerRequest, AnswerResponse
from cortex.observability import trace_operation
from cortex.orchestration.graphs import build_answer_graph
from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)
router = APIRouter()

# Compile graph once at module level
_answer_graph = build_answer_graph().compile()


@router.post("/answer", response_model=AnswerResponse)
@trace_operation("api_answer")
async def answer_api(request: AnswerRequest, http_request: Request):
    """
    Generate an answer for the user query using RAG.

    Blueprint §9.1:
    * POST /api/v1/answer
    * Request: AnswerRequest
    * Response: AnswerResponse with correlation_id
    """
    # Get correlation_id from request state (set by middleware)
    correlation_id = getattr(http_request.state, "correlation_id", None)

    try:
        # Initialize state
        initial_state = {
            "query": request.query,
            "tenant_id": request.tenant_id,
            "user_id": request.user_id,
            "classification": None,
            "retrieval_results": None,
            "assembled_context": None,
            "answer": None,
            "error": None,
            "_correlation_id": correlation_id,  # Pass to nodes if needed
        }

        # Invoke graph
        final_state = await _answer_graph.ainvoke(initial_state)

        # Check for errors
        if final_state.get("error"):
            raise HTTPException(status_code=500, detail=final_state["error"])

        # Return answer
        answer = final_state.get("answer")
        if not answer:
            raise HTTPException(status_code=500, detail="No answer generated")

        # Audit log
        try:
            input_str = request.model_dump_json()
            input_hash = hashlib.sha256(input_str.encode()).hexdigest()

            log_audit_event(
                tenant_id=request.tenant_id,
                user_or_agent=request.user_id,
                action="answer_question",
                input_hash=input_hash,
                risk_level="low",
                correlation_id=correlation_id,
                metadata={
                    "query": request.query,
                    "confidence": answer.confidence_overall,
                },
            )
        except Exception as audit_err:
            logger.error(f"Audit logging failed: {audit_err}")

        return AnswerResponse(
            correlation_id=correlation_id,
            answer=answer,
            debug_info={"classification": str(final_state.get("classification"))}
            if request.debug
            else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Answer API failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
