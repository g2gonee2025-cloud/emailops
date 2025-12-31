"""
Summarize Thread API Routes.

Implements §9.5 of the Canonical Blueprint.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from collections.abc import Mapping

from cortex.audit import log_audit_event
from cortex.context import tenant_id_ctx, user_id_ctx
from cortex.domain_models.rag import ThreadSummary
from cortex.observability import trace_operation
from cortex.orchestration.graphs import build_summarize_graph
from cortex.rag_api.models import SummarizeThreadRequest, SummarizeThreadResponse
from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

logger = logging.getLogger(__name__)
router = APIRouter()

# Lazily compile graph to avoid startup cost (fallback if app.state.graphs not available)

_summarize_graph = None
_graph_lock = asyncio.Lock()


async def get_summarize_graph(http_request: Request = None):
    """Get pre-compiled graph from app.state or lazy load as fallback in thread-safe way."""
    if http_request and hasattr(http_request.app.state, "graphs"):
        cached = http_request.app.state.graphs.get("summarize")
        if cached:
            return cached

    global _summarize_graph
    if _summarize_graph is None:
        async with _graph_lock:
            if _summarize_graph is None:
                logger.info(
                    "Lazily initializing summarize graph (prefer app.state.graphs)."
                )

                def _compile_graph():
                    return build_summarize_graph().compile()

                _summarize_graph = await asyncio.to_thread(_compile_graph)
    return _summarize_graph


@router.post("/summarize", response_model=SummarizeThreadResponse)
@trace_operation("api_summarize")
async def summarize_thread_endpoint(
    request: SummarizeThreadRequest, http_request: Request
) -> SummarizeThreadResponse:
    correlation_id = getattr(http_request.state, "correlation_id", None)
    tenant_id = tenant_id_ctx.get()
    user_id = user_id_ctx.get()

    if not tenant_id or not user_id:
        logger.warning(
            "Missing tenant/user context for summarize",
            extra={"correlation_id": correlation_id},
        )
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        thread_uuid = uuid.UUID(str(request.thread_id))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid thread_id")

    try:
        from cortex.db.models import Conversation
        from cortex.db.session import SessionLocal, set_session_tenant

        with SessionLocal() as session:
            set_session_tenant(session, tenant_id)
            exists = (
                session.query(Conversation.conversation_id)
                .filter(
                    Conversation.tenant_id == tenant_id,
                    Conversation.conversation_id == thread_uuid,
                )
                .first()
            )
            if not exists:
                raise HTTPException(status_code=404, detail="Thread not found")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to validate thread access")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    try:
        initial_state = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "thread_id": str(thread_uuid),
            "max_length": request.max_length,
            "iteration_count": 0,
            "error": None,
            "correlation_id": correlation_id,
        }

        graph = await get_summarize_graph(http_request)
        final_state = await graph.ainvoke(initial_state)

        if not isinstance(final_state, Mapping):
            logger.error(
                "Summarize graph returned invalid state",
                extra={"correlation_id": correlation_id},
            )
            raise HTTPException(status_code=500, detail="Summarization failed")

        if final_state.get("error"):
            logger.error(
                "Summarize workflow error",
                extra={"correlation_id": correlation_id},
            )
            raise HTTPException(status_code=500, detail="Summarization failed")

        summary = final_state.get("summary")
        if summary is None:
            raise HTTPException(status_code=500, detail="Summarization failed")

        if not isinstance(summary, ThreadSummary):
            try:
                summary = ThreadSummary.model_validate(summary)
            except ValidationError:
                logger.error(
                    "Summarize workflow returned invalid summary",
                    extra={"correlation_id": correlation_id},
                    exc_info=True,
                )
                raise HTTPException(status_code=500, detail="Summarization failed")

        input_hash = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        log_audit_event(
            tenant_id=tenant_id,
            user_or_agent=user_id,
            action="summarize_thread",
            input_hash=input_hash,
            risk_level="low",
            correlation_id=correlation_id,
            metadata={"thread_id": request.thread_id},
        )

        return SummarizeThreadResponse(
            correlation_id=correlation_id,
            summary=summary,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Summarize API failed")
        raise HTTPException(status_code=500, detail="Internal Server Error")
