"""
RAG API Services.

This module implements the business logic for the RAG API endpoints.
It abstracts the core functionality away from the HTTP transport layer.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from cortex.audit import log_audit_event
from cortex.context import tenant_id_ctx, user_id_ctx
from cortex.rag_api.models import DraftEmailRequest

logger = logging.getLogger(__name__)


async def draft_email_service(
    graph: Any,
    request: DraftEmailRequest,
    correlation_id: str | None,
) -> dict:
    """
    Service layer for drafting an email.

    This function encapsulates the core business logic for the draft email functionality,
    including state initialization, graph invocation, and audit logging.
    """
    tenant_id = tenant_id_ctx.get()
    user_id = user_id_ctx.get()

    # Initialize state for the graph
    initial_state = {
        "tenant_id": tenant_id,
        "user_id": user_id,
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

    # Invoke the RAG graph
    final_state = await graph.ainvoke(initial_state)

    # Error handling from within the graph
    if error := final_state.get("error"):
        # The service layer should not raise HTTPExceptions.
        # It should return a result or raise a domain-specific exception.
        # For now, we'll re-raise a generic exception that the API layer can catch.
        raise RuntimeError(f"Draft generation failed: {error}")

    draft = final_state.get("draft")
    if not draft:
        raise RuntimeError("No draft generated")

    # Audit logging
    try:
        # PII Hardening: Hash the user's instruction for the audit log
        instruction_hash = hashlib.sha256(request.instruction.encode()).hexdigest()
        input_str = request.model_dump_json()
        input_hash = hashlib.sha256(input_str.encode()).hexdigest()

        log_audit_event(
            tenant_id=tenant_id,
            user_or_agent=user_id,
            action="draft_email",
            input_hash=input_hash,
            risk_level="medium",
            correlation_id=correlation_id,
            metadata={
                "query_hash": instruction_hash,  # Log hash instead of raw query
                "thread_id": str(request.thread_id) if request.thread_id else None,
                "iterations": final_state.get("iteration_count", 0),
            },
        )
    except Exception as audit_err:
        # If audit logging fails, log the error but don't fail the request
        logger.error(f"Audit logging failed: {audit_err}")

    return {
        "draft": draft,
        "iterations": final_state.get("iteration_count", 0),
    }
