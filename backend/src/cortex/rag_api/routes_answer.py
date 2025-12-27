"""
RAG API: Answer routes.

Implements §9.1 of the Canonical Blueprint.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Optional

from cortex.audit import log_audit_event
from cortex.context import tenant_id_ctx, user_id_ctx
from cortex.domain_models.rag import Answer
from cortex.observability import trace_operation  # P2 Fix: Enable tracing
from cortex.orchestration.graphs import build_answer_graph
from cortex.rag_api.models import AnswerRequest, AnswerResponse
from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)
router = APIRouter()

# Lazy load graph (fallback if app.state.graphs not available)

_answer_graph = None
_graph_lock = asyncio.Lock()


async def get_answer_graph(http_request: Optional[Request] = None):
    """Get pre-compiled graph from app.state or lazy load as fallback in thread-safe way."""
    if http_request and hasattr(http_request.app.state, "graphs"):
        cached = http_request.app.state.graphs.get("answer")
        if cached:
            return cached

    global _answer_graph
    if _answer_graph is None:
        async with _graph_lock:
            if _answer_graph is None:
                logger.info(
                    "Lazily initializing Answer Graph (prefer app.state.graphs)..."
                )
                _answer_graph = build_answer_graph().compile()
    return _answer_graph


@router.post("/answer", response_model=AnswerResponse)
@trace_operation("api_answer")
async def answer_api(request: AnswerRequest, http_request: Request):
    # ... (docstring same)
    correlation_id = getattr(http_request.state, "correlation_id", None)

    try:
        initial_state = {
            "query": request.query,
            "tenant_id": tenant_id_ctx.get(),
            "user_id": user_id_ctx.get(),
            "thread_id": request.thread_id,
            "k": request.k,
            "debug": request.debug,
            "classification": None,
            "retrieval_results": None,
            "assembled_context": None,
            "answer": None,
            "error": None,
            "correlation_id": correlation_id,
        }

        graph = await get_answer_graph(http_request)
        final_state = await graph.ainvoke(initial_state)

        if final_state.get("error"):
            logger.error(f"Graph execution error: {final_state['error']}")
            raise HTTPException(status_code=500, detail="Answer workflow failed")

        answer_dict = final_state.get("answer")
        if not answer_dict:
            raise HTTPException(status_code=500, detail="No answer generated")

        answer = Answer.model_validate(answer_dict)

        try:
            # Audit logic (preserved)
            input_str = request.model_dump_json()
            input_hash = hashlib.sha256(input_str.encode()).hexdigest()

            log_audit_event(
                tenant_id=tenant_id_ctx.get(),
                user_or_agent=user_id_ctx.get(),
                action="answer_question",
                input_hash=input_hash,
                risk_level="low",
                correlation_id=correlation_id,
                metadata={
                    "query_hash": hashlib.sha256(
                        request.query.encode()
                    ).hexdigest(),  # Hash query
                    "confidence": answer.confidence_overall,
                },
            )
        except Exception as audit_err:
            logger.error(f"Audit logging failed: {audit_err}")

        return AnswerResponse(
            correlation_id=correlation_id,
            answer=answer,
            confidence=answer.confidence_overall,
            debug_info=(
                {"classification": str(final_state.get("classification"))}
                if request.debug
                else None
            ),
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Answer API failed")
        raise HTTPException(status_code=500, detail="Internal Server Error")
