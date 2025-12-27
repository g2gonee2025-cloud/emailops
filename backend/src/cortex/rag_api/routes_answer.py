"""
RAG API: Answer routes.

Implements §9.1 of the Canonical Blueprint.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

from cortex.audit import log_audit_event
from cortex.context import tenant_id_ctx, user_id_ctx
from cortex.domain_models.rag import Answer
from cortex.observability import trace_operation
from cortex.rag_api.models import AnswerRequest, AnswerResponse
from fastapi import APIRouter, Depends, HTTPException, Request
from langgraph.graph import StateGraph

logger = logging.getLogger(__name__)
router = APIRouter()

# Configuration for debug mode
IS_DEV_MODE = os.getenv("ENVIRONMENT") == "dev"


def get_answer_graph(request: Request) -> StateGraph:
    """Get the answer graph from application state."""
    graph = request.app.state.graphs.get("answer")
    if not graph:
        logger.error("Answer graph not found in app.state.graphs")
        raise HTTPException(status_code=500, detail="Graph not initialized")
    return graph


@router.post("/answer", response_model=AnswerResponse)
@trace_operation("api_answer")
async def answer_api(
    request: AnswerRequest,
    http_request: Request,
    graph: StateGraph = Depends(get_answer_graph),
):
    correlation_id = getattr(http_request.state, "correlation_id", None)
    query_hash = hashlib.sha256(request.query.encode()).hexdigest()

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
            "query_hash": query_hash,
        }

        final_state = await graph.ainvoke(initial_state)

        if final_state.get("error"):
            logger.error(
                f"Graph execution error for query_hash='{query_hash}': {final_state['error']}"
            )
            raise HTTPException(status_code=500, detail="Answer workflow failed")

        answer_dict = final_state.get("answer")
        if not answer_dict:
            logger.error(f"No answer generated for query_hash='{query_hash}'")
            raise HTTPException(status_code=500, detail="No answer generated")

        answer = Answer.model_validate(answer_dict)

        try:
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
                    "query_hash": query_hash,
                    "confidence": answer.confidence_overall,
                },
            )
        except Exception as audit_err:
            logger.error(f"Audit logging failed for query_hash='{query_hash}': {audit_err}")

        debug_info = None
        if request.debug:
            if not IS_DEV_MODE:
                logger.warning(
                    f"Debug mode requested in non-dev environment by user '{user_id_ctx.get()}'"
                )
            else:
                debug_info = {
                    "classification": str(final_state.get("classification")),
                    "retrieval_results": final_state.get("retrieval_results"),
                }

        return AnswerResponse(
            correlation_id=correlation_id,
            answer=answer,
            confidence=answer.confidence_overall,
            debug_info=debug_info,
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception(f"Answer API failed for query_hash='{query_hash}'")
        raise HTTPException(status_code=500, detail="Internal Server Error")
