"""
Chat API Routes.

Implements §9.6 of the Canonical Blueprint.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, List, Literal, Optional

from cortex.audit import log_audit_event
from cortex.config.loader import get_config
from cortex.context import tenant_id_ctx, user_id_ctx
from cortex.domain_models.rag import Answer, RetrievalDiagnostics, ThreadSummary
from cortex.llm.client import complete_json, complete_text
from cortex.orchestration.graphs import build_summarize_graph
from cortex.orchestration.nodes import (
    _extract_evidence_from_answer,
    node_assemble_context,
)
from cortex.prompts import PROMPT_ANSWER_QUESTION
from cortex.rag_api.models import ChatMessage, ChatRequest, ChatResponse
from cortex.retrieval.hybrid_search import KBSearchInput, tool_kb_search_hybrid
from cortex.retrieval.query_classifier import (
    QueryClassificationInput,
    tool_classify_query,
)
from cortex.retrieval.results import SearchResults
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

_summarize_graph = None


class ChatActionDecision(BaseModel):
    """LLM routing decision for chat requests."""

    action: Literal["answer", "search", "summarize"]
    reason: str


def get_summarize_graph():
    global _summarize_graph
    if _summarize_graph is None:
        logger.info("Initializing Summarize Graph...")
        _summarize_graph = build_summarize_graph().compile()
    return _summarize_graph


def _trim_history(messages: List[ChatMessage], max_history: int) -> List[ChatMessage]:
    if max_history <= 0:
        return []
    if len(messages) <= max_history:
        return messages
    return messages[-max_history:]


def _format_history(messages: List[ChatMessage]) -> str:
    return "\n".join(f"{msg.role}: {msg.content}" for msg in messages)


def _latest_user_message(messages: List[ChatMessage]) -> Optional[str]:
    for msg in reversed(messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content.strip()
    return None


def _decide_action(
    request: ChatRequest, history: List[ChatMessage], latest_user: str
) -> ChatActionDecision:
    prompt = (
        "You are a routing assistant for EmailOps. Decide the next action.\n"
        "Choose one of: answer, search, summarize.\n"
        "- Use summarize when the user requests a summary of an email thread or the conversation.\n"
        "- Use search when you need to look up documents or knowledge base results.\n"
        "- Otherwise answer directly using available context.\n\n"
        f"Thread ID: {request.thread_id or 'none'}\n"
        f"Latest user message: {latest_user}\n\n"
        "Conversation history:\n"
        f"{_format_history(history)}\n\n"
        "Return JSON with fields: action, reason."
    )
    raw = complete_json(prompt=prompt, schema=ChatActionDecision.model_json_schema())
    return ChatActionDecision.model_validate(raw)


def _build_retrieval_diagnostics(
    retrieval_results: Optional[SearchResults],
) -> List[RetrievalDiagnostics]:
    diagnostics: List[RetrievalDiagnostics] = []
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
        tenant_id=tenant_id_ctx.get(),
        user_id=user_id_ctx.get(),
        query=query,
        k=k,
        classification=classification,
    )
    return await run_in_threadpool(tool_kb_search_hybrid, tool_input)


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, http_request: Request) -> ChatResponse:
    """
    Conversational chat endpoint with tool routing.

    Blueprint §9.6:
    * POST /api/v1/chat
    * Request: ChatRequest
    * Response: ChatResponse
    """
    correlation_id = getattr(http_request.state, "correlation_id", None)

    if not request.messages:
        raise HTTPException(status_code=400, detail="messages are required")

    latest_user = _latest_user_message(request.messages)
    if not latest_user:
        raise HTTPException(status_code=400, detail="No user message provided")

    config = get_config()
    max_history = (
        request.max_history
        if request.max_history is not None
        else config.unified.max_chat_history
    )
    history = _trim_history(request.messages, max_history)

    try:
        decision = _decide_action(request, history, latest_user)

        if decision.action == "summarize":
            if request.thread_id:
                initial_state = {
                    "tenant_id": tenant_id_ctx.get(),
                    "user_id": user_id_ctx.get(),
                    "thread_id": request.thread_id,
                    "max_length": request.max_length,
                    "iteration_count": 0,
                    "error": None,
                    "correlation_id": correlation_id,
                }
                final_state = await get_summarize_graph().ainvoke(initial_state)
                if final_state.get("error"):
                    raise HTTPException(status_code=500, detail=final_state["error"])
                summary = final_state.get("summary")
                if not summary:
                    raise HTTPException(status_code=500, detail="No summary generated")
                reply = summary.summary_markdown
            else:
                summary_prompt = (
                    "Summarize the following conversation in markdown.\n\n"
                    f"{_format_history(history)}"
                )
                summary_text = complete_text(summary_prompt)
                summary = ThreadSummary(summary_markdown=summary_text, key_points=[])
                reply = summary_text

            response = ChatResponse(
                correlation_id=correlation_id,
                action="summarize",
                reply=reply,
                summary=summary,
                debug_info={"reason": decision.reason} if request.debug else None,
            )
        elif decision.action == "search":
            classification = tool_classify_query(
                QueryClassificationInput(query=latest_user, use_llm=True)
            )
            results = await _run_search(latest_user, request.k, classification)
            results_dicts = (
                [r.model_dump() for r in results.results] if results.results else []
            )
            snippets = "\n".join(
                f"{idx + 1}. {item.highlights[0] if item.highlights else ''}"
                for idx, item in enumerate(results.results[:5])
            )
            reply_prompt = (
                "You are responding to a search request. Provide a concise response "
                "grounded in the search results.\n\n"
                f"User request: {latest_user}\n\n"
                f"Search results:\n{snippets}"
            )
            reply = complete_text(reply_prompt)
            response = ChatResponse(
                correlation_id=correlation_id,
                action="search",
                reply=reply,
                search_results=results_dicts,
                debug_info={"reason": decision.reason} if request.debug else None,
            )
        else:
            classification = tool_classify_query(
                QueryClassificationInput(query=latest_user, use_llm=True)
            )
            results = await _run_search(latest_user, request.k, classification)
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
                prompt = (
                    PROMPT_ANSWER_QUESTION
                    + "\n\nConversation history:\n"
                    + _format_history(history)
                    + "\n\nContext:\n"
                    + assembled_context
                    + f"\n\nQuestion: {latest_user}"
                )
                answer_text = complete_text(prompt)
                evidence = _extract_evidence_from_answer(answer_text, results)
                confidence = min(0.95, 0.5 + 0.1 * len(evidence)) if evidence else 0.6
                answer = Answer(
                    query=latest_user,
                    answer_markdown=answer_text,
                    evidence=evidence,
                    confidence_overall=confidence,
                    safety={},
                    retrieval_diagnostics=_build_retrieval_diagnostics(results),
                )

            response = ChatResponse(
                correlation_id=correlation_id,
                action="answer",
                reply=answer.answer_markdown,
                answer=answer,
                debug_info={"reason": decision.reason} if request.debug else None,
            )

        try:
            input_str = request.model_dump_json()
            input_hash = hashlib.sha256(input_str.encode()).hexdigest()
            log_audit_event(
                tenant_id=tenant_id_ctx.get(),
                user_or_agent=user_id_ctx.get(),
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

        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Chat API failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
