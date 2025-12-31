"""
RAG API: Answer routes.

Implements §9.1 of the Canonical Blueprint.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Mapping
from typing import Any

from cortex.audit import log_audit_event
from cortex.context import tenant_id_ctx, user_id_ctx
from cortex.domain_models.rag import Answer
from cortex.observability import trace_operation
from cortex.rag_api.models import AnswerRequest, AnswerResponse
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError

logger = logging.getLogger(__name__)
router = APIRouter()


def _debug_enabled() -> bool:
    return os.getenv("ENVIRONMENT") == "dev"


def get_answer_graph(request: Request) -> Any:
    """Get the answer graph from application state."""
    graphs = getattr(getattr(request.app, "state", None), "graphs", None)
    if not isinstance(graphs, dict):
        logger.error("Answer graphs not configured on app.state")
        raise HTTPException(status_code=500, detail="Graph not initialized")

    graph = graphs.get("answer")
    if not graph:
        logger.error("Answer graph not found in app.state.graphs")
        raise HTTPException(status_code=500, detail="Graph not initialized")
    if not hasattr(graph, "ainvoke"):
        logger.error("Answer graph does not support async invocation")
        raise HTTPException(status_code=500, detail="Graph not initialized")
    return graph


@router.post("/answer", response_model=AnswerResponse)
@trace_operation("api_answer")
async def answer_api(
    request: AnswerRequest,
    http_request: Request,
    graph: Any = Depends(get_answer_graph),
):
    correlation_id = getattr(http_request.state, "correlation_id", None)
    if not isinstance(request.query, str):
        raise HTTPException(status_code=400, detail="Invalid query")
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Invalid query")

    tenant_id = tenant_id_ctx.get()
    user_id = user_id_ctx.get()
    if not tenant_id or not user_id:
        logger.warning(
            "Missing tenant/user context for answer",
            extra={"correlation_id": correlation_id},
        )
        raise HTTPException(status_code=401, detail="Unauthorized")

    query_hash = hashlib.sha256(query.encode()).hexdigest()

    try:
        initial_state = {
            "query": query,
            "tenant_id": tenant_id,
            "user_id": user_id,
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

        if not isinstance(final_state, Mapping):
            logger.error(
                "Answer graph returned invalid state",
                extra={"correlation_id": correlation_id, "query_hash": query_hash},
            )
            raise HTTPException(status_code=500, detail="Answer workflow failed")

        if final_state.get("error"):
            logger.error(
                "Answer graph execution error",
                extra={"correlation_id": correlation_id, "query_hash": query_hash},
            )
            raise HTTPException(status_code=500, detail="Answer workflow failed")

        answer_dict = final_state.get("answer")
        if answer_dict is None:
            logger.error(
                "No answer generated",
                extra={"correlation_id": correlation_id, "query_hash": query_hash},
            )
            raise HTTPException(status_code=500, detail="Answer workflow failed")

        answer = Answer.model_validate(answer_dict)

        input_str = request.model_dump_json()
        input_hash = hashlib.sha256(input_str.encode()).hexdigest()
        log_audit_event(
            tenant_id=tenant_id,
            user_or_agent=user_id,
            action="answer_question",
            input_hash=input_hash,
            risk_level="low",
            correlation_id=correlation_id,
            metadata={
                "query_hash": query_hash,
                "confidence": answer.confidence_overall,
            },
        )

        debug_info = None
        if request.debug:
            if not _debug_enabled():
                logger.warning(
                    "Debug mode requested in non-dev environment",
                    extra={"user_id": user_id},
                )
            else:
                classification = final_state.get("classification")
                if hasattr(classification, "type") and hasattr(classification, "flags"):
                    classification_info = {
                        "type": classification.type,
                        "flags": list(classification.flags or []),
                    }
                elif classification is not None:
                    classification_info = str(classification)
                else:
                    classification_info = None

                retrieval_results = final_state.get("retrieval_results")
                retrieval_count = 0
                reranker = None
                if retrieval_results is not None:
                    results_list = getattr(retrieval_results, "results", None) or []
                    retrieval_count = len(results_list)
                    reranker = getattr(retrieval_results, "reranker", None)

                debug_info = {
                    "classification": classification_info,
                    "retrieval": {
                        "result_count": retrieval_count,
                        "reranker": reranker,
                    },
                }

        return AnswerResponse(
            correlation_id=correlation_id,
            answer=answer,
            confidence=answer.confidence_overall,
            debug_info=debug_info,
        )

    except HTTPException:
        raise
    except ValidationError:
        logger.exception("Answer response validation failed")
        raise HTTPException(status_code=500, detail="Answer workflow failed")
    except Exception:
        logger.exception(
            "Answer API failed",
            extra={"correlation_id": correlation_id, "query_hash": query_hash},
        )
        raise HTTPException(status_code=500, detail="Internal Server Error")
