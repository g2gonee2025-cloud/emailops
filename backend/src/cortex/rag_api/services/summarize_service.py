"""
Service layer for summarization tasks.
"""
import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, Optional

from cortex.audit import log_audit_event
from cortex.context import tenant_id_ctx, user_id_ctx
from cortex.orchestration.graphs import build_summarize_graph
from fastapi import Request

logger = logging.getLogger(__name__)

_summarize_graph = None
_graph_lock = asyncio.Lock()


async def get_summarize_graph(http_request: Optional[Request] = None) -> Any:
    """Get pre-compiled graph from app.state or lazy load as fallback in thread-safe way."""
    if http_request and hasattr(http_request.app.state, "graphs"):
        cached = http_request.app.state.graphs.get("summarize")
        if cached:
            return cached

    global _summarize_graph
    if _summarize_graph is None:
        async with _graph_lock:
            if _summarize_graph is None:
                logger.info("Lazily Initializing Summarize Graph (prefer app.state.graphs)...")
                _summarize_graph = build_summarize_graph().compile()
    return _summarize_graph


async def summarize_thread(
    thread_id: str,
    max_length: Optional[int],
    correlation_id: Optional[str],
    http_request: Request,
) -> str:
    """
    Orchestrates the summarization of a thread.

    Args:
        thread_id: The ID of the thread to summarize.
        max_length: The maximum length of the summary.
        correlation_id: The correlation ID for the request.
        http_request: The FastAPI request object.

    Returns:
        The generated summary.

    Raises:
        ValueError: If the summarization workflow fails or no summary is generated.
    """
    tenant_id = tenant_id_ctx.get()
    user_id = user_id_ctx.get()

    initial_state = {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "thread_id": thread_id,
        "max_length": max_length,
        "iteration_count": 0,
        "error": None,
        "correlation_id": correlation_id,
    }

    graph = await get_summarize_graph(http_request)
    final_state = await graph.ainvoke(initial_state)

    if error := final_state.get("error"):
        logger.error("Graph execution error: %s", error)
        raise ValueError("Summarization workflow failed")

    summary = final_state.get("summary")
    if not summary:
        raise ValueError("No summary generated")

    try:
        input_data = {"thread_id": thread_id, "max_length": max_length}
        input_json = json.dumps(input_data, sort_keys=True)
        input_hash = hashlib.sha256(input_json.encode()).hexdigest()
        log_audit_event(
            tenant_id=tenant_id,
            user_or_agent=user_id,
            action="summarize_thread",
            input_hash=input_hash,
            risk_level="low",
            correlation_id=correlation_id,
            metadata={"thread_id": thread_id},
        )
    except Exception as audit_err:
        logger.error("Audit logging failed during summarization: %s", audit_err)
        # Non-blocking error: we still return the summary

    return summary
