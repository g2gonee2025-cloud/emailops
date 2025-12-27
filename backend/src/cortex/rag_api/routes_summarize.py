"""
Summarize Thread API Routes.

Implements §9.5 of the Canonical Blueprint.
"""
from __future__ import annotations

import logging

from cortex.observability import trace_operation
from cortex.rag_api.models import SummarizeThreadRequest, SummarizeThreadResponse
from cortex.rag_api.services.summarize_service import summarize_thread as summarize_thread_service
from cortex.security.dependencies import get_current_user
from fastapi import APIRouter, Depends, HTTPException, Request

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/summarize", response_model=SummarizeThreadResponse)
@trace_operation("api_summarize")
async def summarize_thread_endpoint(
    request: SummarizeThreadRequest,
    http_request: Request,
    user: str = Depends(get_current_user),
) -> SummarizeThreadResponse:
    """
    Summarizes a thread of emails.
    """
    correlation_id = getattr(http_request.state, "correlation_id", None)

    try:
        summary = await summarize_thread_service(
            thread_id=request.thread_id,
            max_length=request.max_length,
            correlation_id=correlation_id,
            http_request=http_request,
        )
        return SummarizeThreadResponse(
            correlation_id=correlation_id,
            summary=summary,
        )
    except ValueError as e:
        logger.warning(f"Summarization failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Summarize API failed")
        raise HTTPException(status_code=500, detail="Internal Server Error")