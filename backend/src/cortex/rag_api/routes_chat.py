"""
Chat API Routes.

Implements §9.6 of the Canonical Blueprint.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Literal

from cortex.audit import log_audit_event
from cortex.common.exceptions import ProviderError
from cortex.config.loader import get_config
from cortex.context import tenant_id_ctx, user_id_ctx
from cortex.domain_models.rag import Answer, RetrievalDiagnostics, ThreadSummary
from cortex.llm.client import complete_messages
from cortex.observability import trace_operation  # P2 Fix: Enable tracing
from cortex.orchestration.nodes import (
    _extract_evidence_from_answer,
    node_assemble_context,
)
from cortex.prompts import (
    SYSTEM_ANSWER_QUESTION,
    USER_ANSWER_QUESTION,
    construct_prompt_messages,
)
from cortex.rag_api.models import ChatMessage, ChatRequest, ChatResponse
from cortex.retrieval.hybrid_search import KBSearchInput, tool_kb_search_hybrid
from cortex.retrieval.query_classifier import (
    tool_classify_query,
)
from cortex.retrieval.results import SearchResults
from cortex.safety.guardrails_client import validate_with_repair
from cortex.security.defenses import sanitize_user_input
from cortex.security.dependencies import get_current_user
from cortex.security.validators import sanitize_retrieved_content
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatActionDecision(BaseModel):
    """LLM routing decision for chat requests."""

    action: Literal["answer", "search", "summarize"]
    reason: str


def _complete_json_messages(
    messages: list[dict[str, str]],
    model_cls: type[BaseModel],
    correlation_id: str | None,
) -> BaseModel:
    schema = model_cls.model_json_schema()
    schema_json = json.dumps(schema, indent=2)
    instructions = (
        "Respond with a single valid JSON object that conforms to this JSON Schema:\n"
        f"{schema_json}\n\n"
        "Do not include markdown. Return ONLY the JSON object."
    )
    payload = [dict(m) for m in messages if isinstance(m, dict)]
    inserted = False
    for message in payload:
        if message.get("role") == "system" and isinstance(message.get("content"), str):
            message["content"] = f"{message['content']}\n\n{instructions}"
            inserted = True
            break
    if not inserted:
        payload.insert(0, {"role": "system", "content": instructions})

    kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }
    try:
        raw_output = complete_messages(payload, **kwargs)
    except ProviderError as exc:
        msg = str(exc).lower()
        if "response_format" in msg or "json_object" in msg:
            kwargs.pop("response_format", None)
            raw_output = complete_messages(payload, **kwargs)
        else:
            raise

    return validate_with_repair(
        raw_output,
        model_cls,
        correlation_id or "chat-route",
    )


def get_summarize_graph(http_request: Request):
    """Get pre-compiled graph from app.state."""
    graphs = getattr(http_request.app.state, "graphs", None)
    if isinstance(graphs, dict):
        cached = graphs.get("summarize")
        if cached:
            return cached
    raise HTTPException(status_code=500, detail="Summarize graph not initialized")


def _trim_history(messages: list[ChatMessage], max_history: int) -> list[ChatMessage]:
    if max_history <= 0:
        return []
    if len(messages) <= max_history:
        return messages
    return messages[-max_history:]


def _format_history(messages: list[ChatMessage], sanitize: bool = False) -> str:
    lines: list[str] = []
    for msg in messages:
        content = sanitize_user_input(msg.content) if sanitize else msg.content
        lines.append(f"{msg.role}: {content}")
    return "\n".join(lines)


def _latest_user_message(messages: list[ChatMessage]) -> str | None:
    for msg in reversed(messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content.strip()
    return None


async def _decide_action(
    request: ChatRequest,
    history: list[ChatMessage],
    latest_user: str,
    correlation_id: str | None = None,
) -> ChatActionDecision:
    safe_history = _format_history(history, sanitize=True)
    safe_latest_user = sanitize_user_input(latest_user)
    system_prompt = (
        "You are a routing assistant for EmailOps. Decide the next action.\n"
        "Choose one of: answer, search, summarize.\n"
        "- Use summarize when the user requests a summary of an email thread or the conversation.\n"
        "- Use search when you need to look up documents or knowledge base results.\n"
        "- Otherwise answer directly using available context.\n"
        "Return JSON with fields: action, reason."
    )
    user_prompt = (
        f"Thread ID: {request.thread_id or 'none'}\n"
        f"Latest user message: <user_input>{safe_latest_user}</user_input>\n\n"
        "Conversation history:\n"
        f"{safe_history}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    decision = await run_in_threadpool(
        _complete_json_messages,
        messages,
        ChatActionDecision,
        correlation_id,
    )
    return decision


def _build_retrieval_diagnostics(
    retrieval_results: SearchResults | None,
) -> list[RetrievalDiagnostics]:
    diagnostics: list[RetrievalDiagnostics] = []
    if retrieval_results and retrieval_results.results:
        for idx, item in enumerate(retrieval_results.results[:10]):
            lex = item.lexical_score
            vec = item.vector_score
            if lex is None and item.metadata:
                lex = item.metadata.get("lexical_score", 0.0)
            if vec is None and item.metadata:
                vec = item.metadata.get("vector_score", 0.0)
            diagnostics.append(
                RetrievalDiagnostics(
                    lexical_score=lex or 0.0,
                    vector_score=vec or 0.0,
                    fused_rank=idx,
                    reranker=retrieval_results.reranker,
                )
            )
    return diagnostics


async def _run_search(query: str, k: int, classification: Any) -> SearchResults:
    tool_input = KBSearchInput(
        tenant_id=tenant_id_ctx.get("default"),
        user_id=user_id_ctx.get("anonymous"),
        query=query,
        k=k,
        classification=classification,
    )
    result_wrapper = await tool_kb_search_hybrid(tool_input)
    if hasattr(result_wrapper, "is_err") and result_wrapper.is_err():
        tool_error = result_wrapper.unwrap_err()
        logger.error("Chat search tool failed: %s", tool_error)
        raise HTTPException(status_code=500, detail="Search failed")
    if hasattr(result_wrapper, "unwrap"):
        return result_wrapper.unwrap()
    return result_wrapper


@router.post("/chat", response_model=ChatResponse)
@trace_operation("api_chat")  # P2 Fix: Enable request tracing
async def chat_endpoint(
    request: ChatRequest,
    http_request: Request,
    current_user: dict = Depends(get_current_user),
) -> ChatResponse:
    """
    Conversational chat endpoint with tool routing.

    Blueprint §9.6:
    * POST /api/v1/chat
    * Request: ChatRequest
    * Response: ChatResponse
    """
    correlation_id = getattr(http_request.state, "correlation_id", None)

    latest_user, history = _validate_and_prepare_request(request)
    debug_allowed = _is_debug_allowed(current_user)

    try:
        decision = await _decide_action(request, history, latest_user, correlation_id)

        if decision.action == "summarize":
            response = await _handle_summarize(
                request,
                history,
                correlation_id,
                decision,
                http_request,
                debug_allowed,
            )
        elif decision.action == "search":
            response = await _handle_search(
                request,
                latest_user,
                correlation_id,
                decision,
                debug_allowed,
            )
        else:
            response = await _handle_answer(
                request,
                history,
                latest_user,
                correlation_id,
                decision,
                debug_allowed,
            )

        _log_chat_audit(
            tenant_id_ctx.get("default"),
            user_id_ctx.get("anonymous"),
            request,
            response,
            correlation_id,
        )

        return response
    except HTTPException:
        raise
    except Exception:
        logger.exception("Chat API failed")
        raise HTTPException(status_code=500, detail="Internal Server Error")


def _validate_and_prepare_request(
    request: ChatRequest,
) -> tuple[str, list[ChatMessage]]:
    """Validate request and prepare history."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages are required")

    latest_user = _latest_user_message(request.messages)
    if not latest_user:
        raise HTTPException(status_code=400, detail="No user message provided")

    latest_user = sanitize_user_input(latest_user)

    config = get_config()
    max_history = (
        request.max_history
        if request.max_history is not None
        else config.unified.max_chat_history
    )
    history = _trim_history(request.messages, max_history)
    return latest_user, history


def _is_debug_allowed(current_user: dict | None) -> bool:
    if not current_user or not isinstance(current_user, dict):
        return False
    roles: list[str] = []
    role_value = current_user.get("role")
    if isinstance(role_value, str):
        roles.append(role_value)
    roles_value = current_user.get("roles")
    if isinstance(roles_value, list):
        roles.extend([role for role in roles_value if isinstance(role, str)])
    return "admin" in roles


def _log_chat_audit(
    tenant_id: str,
    user_id: str,
    request: ChatRequest,
    response: ChatResponse,
    correlation_id: str | None,
) -> None:
    """Log audit event for chat."""
    try:
        # PII Leak Fix: Avoid logging the full request.
        # Create a hash of the latest user message as a pseudo-identifier.
        latest_user_message = _latest_user_message(request.messages) or ""
        input_hash = hashlib.sha256(latest_user_message.encode()).hexdigest()

        log_audit_event(
            tenant_id=tenant_id,
            user_or_agent=user_id,
            action="chat",
            input_hash=input_hash,
            risk_level="low",
            correlation_id=correlation_id,
            metadata={
                "action": response.action,
                "thread_id": request.thread_id,
            },
        )
    except Exception as audit_err:
        logger.error(f"Audit logging failed: {audit_err}")


async def _handle_summarize(
    request: ChatRequest,
    history: list[ChatMessage],
    correlation_id: str | None,
    decision: ChatActionDecision,
    http_request: Request,
    debug_allowed: bool,
) -> ChatResponse:
    if request.thread_id:
        max_length = request.max_length if request.max_length is not None else 500
        initial_state = {
            "tenant_id": tenant_id_ctx.get(),
            "user_id": user_id_ctx.get(),
            "thread_id": request.thread_id,
            "max_length": max_length,
            "iteration_count": 0,
            "error": None,
            "correlation_id": correlation_id,
        }
        graph = get_summarize_graph(http_request)
        final_state = await graph.ainvoke(initial_state)
        if final_state.get("error"):
            raise HTTPException(status_code=500, detail=final_state["error"])
        summary = final_state.get("summary")
        if isinstance(summary, ThreadSummary):
            summary_obj = summary
        elif isinstance(summary, dict):
            summary_obj = ThreadSummary.model_validate(summary)
        else:
            summary_obj = None
        if not summary_obj or not summary_obj.summary_markdown:
            raise HTTPException(status_code=500, detail="No summary generated")
        reply = summary_obj.summary_markdown
    else:
        safe_history = _format_history(history, sanitize=True)
        summary_prompt = (
            "<conversation_history>\n" f"{safe_history}\n" "</conversation_history>"
        )
        summary_messages = [
            {
                "role": "system",
                "content": "Summarize the following conversation in markdown.",
            },
            {"role": "user", "content": summary_prompt},
        ]
        summary_text = await run_in_threadpool(complete_messages, summary_messages)
        summary_obj = ThreadSummary(summary_markdown=summary_text, key_points=[])
        reply = summary_text

    return ChatResponse(
        correlation_id=correlation_id,
        action="summarize",
        reply=reply,
        summary=summary_obj,
        debug_info=(
            {"reason": decision.reason} if request.debug and debug_allowed else None
        ),
    )


async def _handle_search(
    request: ChatRequest,
    latest_user: str,
    correlation_id: str | None,
    decision: ChatActionDecision,
    debug_allowed: bool,
) -> ChatResponse:
    classification = await run_in_threadpool(
        tool_classify_query,
        latest_user,
        True,  # use_llm
    )
    safe_k = request.k if request.k is not None else 10
    results = await _run_search(latest_user, safe_k, classification)
    results_list = results.results or []
    results_dicts = [r.model_dump() for r in results_list]
    snippets = "\n".join(
        f"{idx + 1}. <snippet>{sanitize_retrieved_content(item.highlights[0]) if item.highlights else ''}</snippet>"
        for idx, item in enumerate(results_list[:5])
    )
    safe_latest_user = sanitize_user_input(latest_user)
    reply_prompt = (
        f"User request: <user_input>{safe_latest_user}</user_input>\n\n"
        f"Search results:\n{snippets}"
    )
    reply_messages = [
        {
            "role": "system",
            "content": (
                "You are responding to a search request. Provide a concise response "
                "grounded in the search results."
            ),
        },
        {"role": "user", "content": reply_prompt},
    ]
    reply = await run_in_threadpool(complete_messages, reply_messages)
    return ChatResponse(
        correlation_id=correlation_id,
        action="search",
        reply=reply,
        search_results=results_dicts,
        debug_info=(
            {"reason": decision.reason} if request.debug and debug_allowed else None
        ),
    )


async def _handle_answer(
    request: ChatRequest,
    history: list[ChatMessage],
    latest_user: str,
    correlation_id: str | None,
    decision: ChatActionDecision,
    debug_allowed: bool,
) -> ChatResponse:
    classification = await run_in_threadpool(
        tool_classify_query,
        latest_user,
        True,  # use_llm
    )
    safe_k = request.k if request.k is not None else 10
    results = await _run_search(latest_user, safe_k, classification)
    context_state = {
        "retrieval_results": results,
    }
    assembled_context = node_assemble_context(context_state).get(
        "assembled_context", ""
    )
    if not assembled_context:
        answer_text = (
            "I could not find any relevant information to answer your question."
        )
        answer = Answer(
            query=latest_user,
            answer_markdown=answer_text,
            evidence=[],
            confidence_overall=0.0,
            safety={},
            retrieval_diagnostics=[],
        )
    else:
        # Sanitize all user-controllable inputs
        safe_history = _format_history(history, sanitize=True)
        safe_context = sanitize_user_input(assembled_context)
        safe_query = sanitize_user_input(latest_user)

        # Construct the prompt using the secure message-based approach
        messages = construct_prompt_messages(
            system_prompt_template=SYSTEM_ANSWER_QUESTION,
            user_prompt_template=f"Conversation history:\n{safe_history}\n\n"
            + USER_ANSWER_QUESTION,
            context=safe_context,
            query=safe_query,
        )

        answer_text = await run_in_threadpool(complete_messages, messages)

        evidence = await run_in_threadpool(
            _extract_evidence_from_answer, answer_text, results
        )
        confidence = min(0.95, 0.5 + 0.1 * len(evidence)) if evidence else 0.6
        answer = Answer(
            query=latest_user,
            answer_markdown=answer_text,
            evidence=evidence,
            confidence_overall=confidence,
            safety={},
            retrieval_diagnostics=_build_retrieval_diagnostics(results),
        )

    return ChatResponse(
        correlation_id=correlation_id,
        action="answer",
        reply=answer.answer_markdown,
        answer=answer,
        debug_info=(
            {"reason": decision.reason} if request.debug and debug_allowed else None
        ),
    )
