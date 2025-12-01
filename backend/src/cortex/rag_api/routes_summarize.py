"""
RAG API: Summarize routes.

Implements §9.3 of the Canonical Blueprint.
"""
from __future__ import annotations

import logging
import hashlib

from fastapi import APIRouter, HTTPException, Request

from cortex.models.api import SummarizeThreadRequest, SummarizeThreadResponse
from cortex.models.rag import ThreadSummary
from cortex.observability import trace_operation
from cortex.orchestration.graphs import build_summarize_graph
from cortex.audit import log_audit_event

logger = logging.getLogger(__name__)
router = APIRouter()

# Compile graph once
_summarize_graph = build_summarize_graph().compile()


@router.post("/summarize-thread", response_model=SummarizeThreadResponse)
@trace_operation("api_summarize_thread")
async def summarize_thread_api(request: SummarizeThreadRequest, http_request: Request):
    """
    Summarize an email thread.
    
    Blueprint §9.3:
    * POST /api/v1/summarize-thread
    * Request: SummarizeThreadRequest
    * Response: SummarizeThreadResponse with correlation_id
    """
    # Get correlation_id from request state (set by middleware)
    correlation_id = getattr(http_request.state, "correlation_id", None)
    
    try:
        # Initialize state
        initial_state = {
            "tenant_id": request.tenant_id,
            "user_id": request.user_id,
            "thread_id": str(request.thread_id),
            "thread_context": None,
            "facts_ledger": None,
            "critique": None,
            "iteration_count": 0,
            "summary": None,
            "error": None,
            "_correlation_id": correlation_id,
        }
        
        # Invoke graph
        final_state = await _summarize_graph.ainvoke(initial_state)
        
        # Check for errors
        if final_state.get("error"):
            raise HTTPException(status_code=500, detail=final_state["error"])
            
        # Return summary
        summary = final_state.get("summary")
        if not summary:
            raise HTTPException(status_code=500, detail="No summary generated")
            
        # Audit log
        try:
            input_str = request.model_dump_json()
            input_hash = hashlib.sha256(input_str.encode()).hexdigest()
            
            log_audit_event(
                tenant_id=request.tenant_id,
                user_or_agent=request.user_id,
                action="summarize_thread",
                input_hash=input_hash,
                risk_level="low",
                correlation_id=correlation_id,
                metadata={"thread_id": str(request.thread_id)}
            )
        except Exception as audit_err:
            logger.error(f"Audit logging failed: {audit_err}")
            
        return SummarizeThreadResponse(
            correlation_id=correlation_id,
            summary=summary,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))