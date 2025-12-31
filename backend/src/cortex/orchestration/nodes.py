"""
Graph Nodes.

Implements §10.2 of the Canonical Blueprint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Optional

from cortex.common.exceptions import SecurityError
from cortex.config.loader import EmailOpsConfig, get_config
from cortex.domain_models.facts_ledger import CriticReview, FactsLedger
from cortex.domain_models.rag import (
    Answer,
    DraftCritique,
    DraftValidationScores,
    EmailDraft,
    EvidenceItem,
    NextAction,
    RetrievalDiagnostics,
    ThreadContext,
    ThreadMessage,
    ThreadParticipant,
    ThreadSummary,
    ToneStyle,
)
from cortex.llm.client import complete_json, complete_messages, complete_text
from cortex.observability import get_logger, trace_operation
from cortex.orchestration.redacted import Redacted
from cortex.prompts import (
    SYSTEM_ANSWER_QUESTION,
    SYSTEM_CRITIQUE_EMAIL,
    SYSTEM_DRAFT_EMAIL_AUDIT,
    SYSTEM_DRAFT_EMAIL_IMPROVE,
    SYSTEM_DRAFT_EMAIL_INITIAL,
    SYSTEM_SUMMARIZE_ANALYST,
    SYSTEM_SUMMARIZE_CRITIC,
    SYSTEM_SUMMARIZE_FINAL,
    SYSTEM_SUMMARIZE_IMPROVER,
    USER_ANSWER_QUESTION,
    USER_CRITIQUE_EMAIL,
    USER_DRAFT_EMAIL_AUDIT,
    USER_DRAFT_EMAIL_IMPROVE,
    USER_DRAFT_EMAIL_INITIAL,
    USER_SUMMARIZE_ANALYST,
    USER_SUMMARIZE_CRITIC,
    USER_SUMMARIZE_FINAL,
    USER_SUMMARIZE_IMPROVER,
    construct_prompt_messages,
    get_prompt,
)
from cortex.retrieval.hybrid_search import (
    KBSearchInput,
    tool_kb_search_hybrid,
)
from cortex.retrieval.query_classifier import (
    QueryClassification,
    QueryClassificationInput,
    tool_classify_query,
)
from cortex.retrieval.results import SearchResults
from cortex.safety.guardrails_client import validate_with_repair
from cortex.safety.policy_enforcer import check_action
from cortex.security.defenses import sanitize_user_input
from cortex.security.injection_defense import validate_for_injection
from cortex.security.validators import sanitize_retrieved_content, validate_file_result
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, or_, select
from sqlalchemy.orm import aliased

logger = logging.getLogger(__name__)


class DraftGenerationOutput(BaseModel):
    """Simplified schema for LLM draft generation."""

    to: list[str] = Field(description="List of recipient email addresses")
    cc: list[str] = Field(
        default_factory=list, description="List of CC email addresses"
    )
    subject: str = Field(description="Email subject line")
    body_markdown: str = Field(description="Email body in Markdown format")
    next_actions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of next actions (description, owner, etc.)",
    )


def _complete_with_guardrails(
    messages: list[dict[str, str]],
    model_cls: type[BaseModel],
    correlation_id: str | None,
):
    """Run structured completion with guardrails repair fallback."""
    schema = model_cls.model_json_schema()

    # The `complete_json` function is deprecated. We should be moving towards
    # a pattern where the model is guided to produce JSON via system prompts
    # and we parse the result from `complete_messages`.
    # For now, we adapt to what `complete_json` expects.
    if not messages:
        raise ValueError("Cannot perform completion with empty messages.")

    # Reconstruct a single "prompt" string for the deprecated function.
    # This is a temporary measure during the security transition.
    # The `system` message provides the instructions and JSON schema info.
    # The `user` message provides the data.
    system_content = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
    reconstructed_prompt = f"{system_content}\n\n{user_content}"

    raw = complete_json(prompt=reconstructed_prompt, schema=schema)
    try:
        return model_cls.model_validate(raw)
    except ValidationError:
        raw_payload = json.dumps(raw, ensure_ascii=False)
        return validate_with_repair(
            raw_payload,
            model_cls,
            correlation_id or str(uuid.uuid4()),
        )


# -----------------------------------------------------------------------------
# Helper Tools (§10.2)
# -----------------------------------------------------------------------------


def _extract_patterns(
    text: str,
    pattern: re.Pattern,
    seen: set[str],
    mentions: list[str],
    group: int = 1,
    min_len: int = 0,
) -> None:
    """Helper to extract patterns and deduplicate."""
    for match in pattern.finditer(text):
        ref = match.group(group).strip()
        lower_ref = ref.lower()
        if (min_len == 0 or len(ref) > min_len) and lower_ref not in seen:
            # Additional check for quoted refs (Pattern 2 logic)
            if pattern.pattern.startswith("[\"']") and not (
                "_" in ref or any(c.isupper() for c in ref)
            ):
                continue

            seen.add(lower_ref)
            mentions.append(ref)


def extract_document_mentions(text: str) -> list[str]:
    """
    Extract filenames or document references from text.

    Blueprint §10.3.1:
    * Finds patterns like:
      - Explicit file references: "document.pdf", "report.docx"
      - Attachment references: "see attached", "as per the attachment"
      - Quoted filenames: "File: report.pdf"

    Args:
        text: The text to search for document mentions

    Returns:
        List of extracted document/filename mentions
    """
    if not text:
        return []

    mentions: list[str] = []
    seen: set[str] = set()

    # Pattern 1: Common document extensions
    _extract_patterns(
        text,
        re.compile(
            r"\b([\w\-_.]+\.(?:pdf|docx?|xlsx?|pptx?|txt|csv|eml|msg|png|jpg|jpeg|gif))\b",
            re.IGNORECASE,
        ),
        seen,
        mentions,
    )

    # Pattern 2: Quoted file references
    quoted_pattern = re.compile(r'["\']([A-Z][A-Za-z0-9\s_\-]{2,30})["\']')
    for match in quoted_pattern.finditer(text):
        try:
            ref = match.group(1).strip()
            lower_ref = ref.lower()
            # Only include if it looks like a document name (has capital or underscore)
            if lower_ref not in seen and ("_" in ref or any(c.isupper() for c in ref)):
                seen.add(lower_ref)
                mentions.append(ref)
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(
                "Skipping malformed quoted reference during extraction: %s",
                e,
                exc_info=True,
            )
            continue
        except Exception:
            logger.exception("Unexpected error while extracting quoted references")
            raise

    # Pattern 3: Explicit attachment references
    _extract_patterns(
        text,
        re.compile(
            r"(?:attached|attachment|enclosed|enclosure)[:\s]+([A-Za-z0-9\s_\-]+?)(?:\.|,|\s|$)",
            re.IGNORECASE,
        ),
        seen,
        mentions,
        min_len=2,
    )

    return mentions


def _extract_entity_mentions(text: str) -> list[str]:
    """Extract candidate entity mentions from query text."""
    if not text:
        return []

    candidates: set[str] = set()

    quoted = re.findall(r'["\']([^"\']{2,100})["\']', text)
    candidates.update(s.strip() for s in quoted if s.strip())

    capitalized_phrases = re.findall(r"\b(?:[A-Z][\w&]*(?:\s+[A-Z][\w&]*)+)\b", text)
    candidates.update(s.strip() for s in capitalized_phrases if s.strip())

    single_caps = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    candidates.update(s.strip() for s in single_caps if s.strip())

    stop_words = {
        "What",
        "When",
        "Where",
        "Why",
        "Who",
        "How",
        "Can",
        "Could",
        "Should",
        "Would",
        "Does",
        "Do",
        "Did",
        "Is",
        "Are",
        "The",
        "A",
        "An",
        "And",
        "Or",
        "To",
        "From",
        "For",
        "Of",
        "In",
        "On",
        "Please",
        "Tell",
        "Explain",
    }

    return [
        candidate
        for candidate in candidates
        if candidate and candidate not in stop_words
    ]


async def _safe_stat_mb(path: Path) -> float:
    """Safely get file size in MB."""
    try:
        # Use to_thread to avoid blocking the event loop with sync file I/O
        if await asyncio.to_thread(path.exists):
            stat_result = await asyncio.to_thread(path.stat)
            return stat_result.st_size / (1024 * 1024)
        return 0.0
    except Exception:
        return 0.0


async def _select_attachments_from_mentions(
    context_snippets: list[dict[str, Any]],
    mentions: list[str],
    *,
    max_attachments: int = 3,
) -> list[dict[str, Any]]:
    """Select attachments that were explicitly mentioned/extracted from text."""
    cfg = get_config()
    base_dir = (
        Path(cfg.directories.export_root).resolve()
        if cfg.directories.export_root
        else Path.cwd()
    )
    allowed_patterns = cfg.file_patterns.allowed_file_patterns
    attach_max_mb = float(cfg.limits.max_attachment_text_chars / (1024 * 1024))
    limit = max_attachments

    if not mentions:
        return []

    wanted = {m.strip().lower() for m in mentions if m and isinstance(m, str)}
    out: list[dict[str, Any]] = []

    for c in context_snippets or []:
        try:
            path_str = str(c.get("path") or "")
            if not path_str or not Path(path_str).suffix:
                continue

            name = str(c.get("attachment_name") or Path(path_str).name).lower()

            if name and any(w in name for w in wanted):
                p = Path(path_str)
                result = validate_file_result(
                    str(p), base_directory=base_dir, must_exist=True
                )
                if not result.is_ok():
                    continue

                validated_path = result.value
                if await _safe_stat_mb(validated_path) > attach_max_mb:
                    continue

                if not any(
                    validated_path.match(pattern) for pattern in allowed_patterns
                ):
                    continue

                out.append(
                    {"path": str(validated_path), "filename": validated_path.name}
                )
                if len(out) >= int(limit):
                    break
        except Exception:
            continue
    return out


async def _select_all_available_attachments(
    context_snippets: list[dict[str, Any]], *, max_attachments: int = 3
) -> list[dict[str, Any]]:
    """Select any valid attachments found in context (heuristic fallback)."""
    cfg = get_config()
    base_dir = (
        Path(cfg.directories.export_root).resolve()
        if cfg.directories.export_root
        else Path.cwd()
    )
    allowed_patterns = cfg.file_patterns.allowed_file_patterns
    attach_max_mb = float(cfg.limits.max_attachment_text_chars / (1024 * 1024))

    selected: list[dict[str, Any]] = []
    seen_paths = set()

    for c in context_snippets or []:
        try:
            path_str = str(c.get("path") or "")
            if not path_str:
                continue

            p = Path(path_str)
            if not p.suffix:
                continue

            if str(p) in seen_paths:
                continue

            result = validate_file_result(
                str(p), base_directory=base_dir, must_exist=True
            )
            if not result.is_ok():
                continue

            validated_path = result.value
            if not any(validated_path.match(pattern) for pattern in allowed_patterns):
                continue

            if await _safe_stat_mb(validated_path) > attach_max_mb:
                continue

            selected.append(
                {"path": str(validated_path), "filename": validated_path.name}
            )
            seen_paths.add(str(validated_path))

            if len(selected) >= int(max_attachments):
                break
        except Exception:
            continue
    return selected


@trace_operation("tool_email_get_thread")
def tool_email_get_thread(
    thread_id: uuid.UUID | str,
    tenant_id: str,
    include_attachments: bool = False,
) -> ThreadContext | None:
    """
    Fetch thread context from database.

    Blueprint §10.2:
    * tool_email_get_thread(thread_id: UUID, tenant_id: str) -> ThreadContext
    * Retrieves conversation and builds ThreadContext from messages JSONB
    * Builds ThreadContext with participants and messages

    Args:
        thread_id: UUID of the conversation to fetch
        tenant_id: Tenant ID for RLS
        include_attachments: Whether to include attachment info

    Returns:
        ThreadContext with full thread data, or None if not found
    """
    from cortex.db.models import Conversation
    from cortex.db.session import SessionLocal
    from sqlalchemy.exc import SQLAlchemyError

    # Ensure thread_id is a UUID
    try:
        thread_uuid = uuid.UUID(str(thread_id))
    except ValueError:
        logger.warning("Invalid thread_id format: %s", thread_id)
        return None

    try:
        with SessionLocal() as session:
            from cortex.db.session import set_session_tenant

            set_session_tenant(session, tenant_id)

            # Fetch conversation (which contains messages as JSONB)
            conversation = (
                session.query(Conversation)
                .filter(
                    Conversation.conversation_id == thread_uuid,
                    Conversation.tenant_id == tenant_id,
                )
                .first()
            )

            if not conversation:
                logger.warning("Conversation not found: %s", thread_uuid)
                return None

            # Build participants from conversation.participants JSONB
            participants_dict: dict[str, ThreadParticipant] = {}
            if conversation.participants:
                for p in conversation.participants:
                    email = p.get("smtp", p.get("email", ""))
                    if email and email not in participants_dict:
                        participants_dict[email] = ThreadParticipant(
                            email=email,
                            name=p.get("name"),
                            role=p.get("role", "participant"),
                        )

            # Build thread messages from conversation.messages JSONB
            thread_messages: list[ThreadMessage] = []
            if conversation.messages:
                for msg in conversation.messages:
                    msg_id = msg.get("message_id", str(uuid.uuid4()))
                    from_addr = msg.get("from", msg.get("sender", ""))
                    to_addrs = msg.get("to", [])
                    cc_addrs = msg.get("cc", [])
                    subject = msg.get("subject", conversation.subject or "")
                    body = msg.get("body", msg.get("text", ""))
                    sent_at = msg.get("sent_at", msg.get("date"))

                    # Parse sent_at if string
                    if isinstance(sent_at, str):
                        try:
                            from datetime import datetime

                            sent_at = datetime.fromisoformat(
                                sent_at.replace("Z", "+00:00")
                            )
                        except ValueError:
                            sent_at = None

                    thread_messages.append(
                        ThreadMessage(
                            message_id=msg_id,
                            sent_at=sent_at,
                            recv_at=sent_at,  # Use same as sent_at
                            from_addr=from_addr,
                            to_addrs=(
                                to_addrs if isinstance(to_addrs, list) else [to_addrs]
                            ),
                            cc_addrs=(
                                cc_addrs if isinstance(cc_addrs, list) else [cc_addrs]
                            ),
                            subject=subject,
                            body_markdown=body,
                            is_inbound=False,  # Would need more logic to determine
                        )
                    )

            # If no structured messages, create one from summary or chunks
            if not thread_messages:
                # Fallback: create a placeholder message from summary
                thread_messages.append(
                    ThreadMessage(
                        message_id=str(thread_uuid),
                        sent_at=conversation.latest_date,
                        recv_at=conversation.latest_date,
                        from_addr="",
                        to_addrs=[],
                        cc_addrs=[],
                        subject=conversation.subject or "",
                        body_markdown=conversation.summary_text or "",
                        is_inbound=False,
                    )
                )

            return ThreadContext(
                thread_id=thread_uuid,
                subject=conversation.subject or "",
                participants=list(participants_dict.values()),
                messages=thread_messages,
            )
    except SQLAlchemyError as e:
        logger.error(
            "Database error while fetching thread %s for tenant %s: %s",
            thread_uuid,
            tenant_id,
            e,
            exc_info=False,  # Keep log clean, stack is in Sentry
        )
        return None
    except (TypeError, KeyError, AttributeError) as e:
        logger.warning(
            "Data integrity error parsing thread %s for tenant %s: %s",
            thread_uuid,
            tenant_id,
            e,
            exc_info=True,
        )
        return None
    except Exception:
        logger.exception(
            "Unexpected error fetching thread %s for tenant %s",
            thread_uuid,
            tenant_id,
        )
        return None


def _extract_evidence_from_answer(
    answer_text: str, retrieval_results: SearchResults | None
) -> list[EvidenceItem]:
    """
    Extract evidence items by matching citations in answer text to retrieval results.

    Looks for patterns like:
    - [Source 1], [Source 2], etc.
    - (Source ID: xxx)
    - References to chunk/message IDs
    """
    import re

    if not retrieval_results or not retrieval_results.results:
        return []

    evidence = []
    seen_ids = set()

    # Find all source references in the answer
    # Pattern: [Source N] or (Source N) or Source N
    source_refs = re.findall(r"\[?Source\s*(\d+)\]?", answer_text, re.IGNORECASE)
    referenced_indices = {int(ref) - 1 for ref in source_refs}  # Convert to 0-based

    for i, result in enumerate(retrieval_results.results):
        # Include if explicitly referenced, or if in top 3 results
        if i in referenced_indices or i < 3:
            chunk_id = result.chunk_id or result.message_id or f"result_{i}"

            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            evidence.append(
                EvidenceItem(
                    chunk_id=chunk_id,
                    text=result.highlights[0] if result.highlights else "",
                    relevance_score=result.score,
                    source_type="email" if result.message_id else "attachment",
                )
            )

    return evidence[:5]  # Limit to top 5 evidence items


from cortex.audit import log_audit_event


def node_handle_error(state: dict[str, Any]) -> dict[str, Any]:
    """
    Handle errors in graph execution.

    Blueprint §10.5:
    * records error details into audit_log
    * logs via observability.get_logger
    * sets state.error
    """
    error_detail = state.get("error", "Unknown error")
    obs_logger = get_logger(__name__)

    # Sanitize error for external audit logging to prevent PII leakage
    error_for_audit = "An internal error occurred during graph execution."

    obs_logger.error(
        f"Graph execution error: {error_detail}",
        extra={"tenant_id": state.get("tenant_id")},
        exc_info=True,
    )

    # Record sanitized error to audit_log
    try:
        query = state.get("query", "")
        # Basic sanitization for query in case it's part of the error
        safe_query_snippet = (
            f"{query[:50]}..." if isinstance(query, str) and len(query) > 50 else query
        )

        log_audit_event(
            tenant_id=state.get("tenant_id", "unknown"),
            user_or_agent=state.get("user_id", "system"),
            action="graph_error",
            risk_level="medium",
            metadata={
                "error": error_for_audit,
                "query_snippet": safe_query_snippet,
                "graph_type": state.get("_graph_type", "unknown"),
            },
        )
    except Exception as e:
        obs_logger.warning(f"Failed to log audit event: {e}")

    return {"error": error_for_audit}


def node_assemble_context(state: dict[str, Any]) -> dict[str, Any]:
    """
    Assemble context from retrieval results.

    Blueprint §10.1.1:
    * Redundant cleaning optimization
    * Proactive injection defense
    """
    results = state.get("retrieval_results")
    if not results or not results.results:
        return {"assembled_context": Redacted("")}

    context_parts = []

    for i, item in enumerate(results.results):
        # 1. Redundant cleaning optimization
        text = "\n".join(item.highlights)

        # 2. Proactive injection defense
        safe_text = sanitize_retrieved_content(text)

        # Format with metadata for citation
        # We use a simple index or ID reference for the LLM
        source_ref = f"Source {i + 1} (ID: {item.chunk_id or item.message_id})"
        context_parts.append(f"[{source_ref}]\n{safe_text}")

    return {"assembled_context": Redacted("\n\n".join(context_parts))}


def node_query_graph(state: dict[str, Any]) -> dict[str, Any]:
    """
    Query the knowledge graph for entities mentioned in the prompt.

    Blueprint §10.1:
    * Inject relational facts into answer context.
    """
    query_obj = state.get("query", "")
    try:
        query = query_obj.get_secret_value()
    except AttributeError:
        query = query_obj
    tenant_id = state.get("tenant_id")

    if not query or not tenant_id:
        return {"graph_context": Redacted("")}
    mentions = _extract_entity_mentions(query)
    if not mentions:
        return {"graph_context": Redacted("")}

    try:
        nodes, edges = _fetch_graph_entities(tenant_id, mentions)
        if not nodes:
            return {"graph_context": Redacted("")}

        context_lines = _build_graph_context_lines(nodes, edges)
        if not context_lines:
            return {"graph_context": Redacted("")}

        graph_context = "Knowledge Graph Facts:\n" + "\n".join(
            f"- {line}" for line in context_lines
        )
        return {"graph_context": Redacted(graph_context)}
    except Exception as e:
        logger.error(f"Graph query failed: {e}")
        return {"graph_context": Redacted("")}


def _fetch_graph_entities(
    tenant_id: str, mentions: list[str]
) -> tuple[list[Any], list[Any]]:
    """Fetch entity nodes and edges from the knowledge graph."""
    from cortex.db.models import EntityEdge, EntityNode
    from cortex.db.session import SessionLocal, set_session_tenant
    from sqlalchemy.dialects.postgresql import any_

    with SessionLocal() as session:
        set_session_tenant(session, tenant_id)

        # Perf: Use ILIKE ANY for efficient multi-pattern matching
        patterns = set(mentions)
        patterns.update(f"%{m}%" for m in mentions if len(m) >= 4)

        if not patterns:
            return [], []

        nodes = (
            session.execute(
                select(EntityNode).where(
                    EntityNode.tenant_id == tenant_id,
                    EntityNode.name.ilike(any_(list(patterns))),
                )
            )
            .scalars()
            .all()
        )

        if not nodes:
            return [], []

        node_ids = [node.node_id for node in nodes]

        source_node = aliased(EntityNode)
        target_node = aliased(EntityNode)
        edges = session.execute(
            select(EntityEdge, source_node, target_node)
            .join(source_node, EntityEdge.source_id == source_node.node_id)
            .join(target_node, EntityEdge.target_id == target_node.node_id)
            .where(
                EntityEdge.tenant_id == tenant_id,
                or_(
                    EntityEdge.source_id.in_(node_ids),
                    EntityEdge.target_id.in_(node_ids),
                ),
            )
        ).all()

        return list(nodes), list(edges)


def _build_graph_context_lines(nodes: list[Any], edges: list[Any]) -> list[str]:
    """Build context lines from graph nodes and edges."""
    context_lines: list[str] = []

    for node in nodes:
        if node.description:
            context_lines.append(
                sanitize_retrieved_content(
                    f"Entity: {node.name} ({node.type}) - {node.description}"
                )
            )
        else:
            context_lines.append(
                sanitize_retrieved_content(f"Entity: {node.name} ({node.type})")
            )

    for edge, source, target in edges:
        relation = edge.relation.replace("_", " ").lower()
        description = f" {edge.description}" if edge.description else ""
        fact = f"{source.name} {relation} {target.name}.{description}"
        context_lines.append(sanitize_retrieved_content(fact.strip()))

    return context_lines


def node_classify_query(state: dict[str, Any]) -> dict[str, Any]:
    """
    Classify the user query.

    Blueprint §8.1:
    * Determine intent (navigational, semantic, drafting)
    """
    query_obj = state.get("query", "")
    try:
        query = query_obj.get_secret_value()
    except AttributeError:
        query = query_obj

    try:
        validate_for_injection(query)
    except SecurityError:
        logger.error("Potential injection attack detected in query.")
        return {"error": "Invalid input detected."}

    try:
        # args = QueryClassificationInput(query=query, use_llm=True)
        classification = tool_classify_query(query=query, use_llm=True)
        return {"classification": classification}
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return {
            "classification": QueryClassification(
                query=query, type="semantic", flags=["classification_failed"]
            )
        }


async def node_retrieve_context(state: dict[str, Any]) -> dict[str, Any]:
    """
    Retrieve context based on query and classification.

    Blueprint §8.3:
    * Call Hybrid Search
    """
    query_obj = state.get("query", "")
    query = (
        query_obj.get_secret_value()
        if hasattr(query_obj, "get_secret_value")
        else query_obj
    )

    classification = state.get("classification")
    tenant_id = state.get("tenant_id")
    user_id = state.get("user_id")
    k = state.get("k", 10)

    try:
        args = KBSearchInput(
            tenant_id=tenant_id,
            user_id=user_id,
            query=query,
            classification=classification,
            k=k,
        )
        # tool_kb_search_hybrid is async, await it directly
        result = await tool_kb_search_hybrid(args)

        # Handle Result type (Ok/Err)
        if result.is_ok():
            return {"retrieval_results": result.unwrap()}
        else:
            return {"error": f"Retrieval failed: {result.unwrap_err()}"}
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {"error": f"Retrieval failed: {e!s}"}


def node_generate_answer(state: dict[str, Any]) -> dict[str, Any]:
    """
    Generate answer using LLM and context.

    Blueprint §3.6:
    * Generate response with citations
    """
    query_obj = state.get("query", "")
    try:
        query = query_obj.get_secret_value()
    except AttributeError:
        query = query_obj

    context = state.get("assembled_context", "")
    try:
        context = context.get_secret_value()
    except AttributeError:
        pass

    graph_context = state.get("graph_context", "")
    try:
        graph_context = graph_context.get_secret_value()
    except AttributeError:
        pass

    combined_context = "\n\n".join(
        part for part in [context, graph_context] if part
    ).strip()

    if not combined_context:
        return {
            "answer": Answer(
                query=query,
                answer_markdown="I could not find any relevant information to answer your question.",
                evidence=[],
                confidence_overall=0.0,
                safety={},
                retrieval_diagnostics=[],
            )
        }

    try:
        # We want the LLM to generate the answer text.
        # Constructing the full Answer object via JSON might be too complex for the LLM
        # to get the EvidenceItem UUIDs right without very specific prompting.
        # For now, let's get the text answer and construct a basic Answer object.

        messages = construct_prompt_messages(
            system_prompt_template=SYSTEM_ANSWER_QUESTION,
            user_prompt_template=USER_ANSWER_QUESTION,
            context=sanitize_user_input(combined_context),
            query=sanitize_user_input(query),
        )

        answer_text = complete_messages(messages)

        retrieval_results: SearchResults | None = state.get("retrieval_results")

        # Extract evidence from answer text and retrieval results
        evidence = _extract_evidence_from_answer(answer_text, retrieval_results)

        # Calculate confidence based on evidence quality
        confidence = min(0.95, 0.5 + 0.1 * len(evidence)) if evidence else 0.6

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

        return {
            "answer": Answer(
                query=query,
                answer_markdown=answer_text,
                evidence=evidence,
                confidence_overall=confidence,
                safety={},
                retrieval_diagnostics=diagnostics,
            )
        }
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {"error": f"Generation failed: {e!s}"}


def node_prepare_draft_query(state: dict[str, Any]) -> dict[str, Any]:
    """
    Prepare query for drafting.

    Blueprint §10.3:
    * Combine explicit query and thread context
    """
    explicit_query = state.get("explicit_query", "")
    try:
        validate_for_injection(explicit_query)
    except SecurityError:
        logger.error("Potential injection attack detected in explicit_query.")
        return {"error": "Invalid input detected."}

    # If we had thread context, we would combine it here.
    # For now, just use explicit query as the search query.
    return {"query": explicit_query}


def node_draft_email_initial(state: dict[str, Any]) -> dict[str, Any]:
    """
    Generate initial email draft.

    Blueprint §10.3:
    * Draft based on context and query
    """
    from cortex.config.loader import get_config

    query = state.get("explicit_query", "")
    context = state.get("assembled_context", "")

    # Get sender info from config
    config = get_config()
    sender_name = config.email.sender_locked_name or "User"
    sender_email = config.email.sender_locked_email or "user@example.com"

    try:
        messages = construct_prompt_messages(
            system_prompt_template=SYSTEM_DRAFT_EMAIL_INITIAL,
            user_prompt_template=USER_DRAFT_EMAIL_INITIAL,
            mode=sanitize_user_input(state.get("mode", "fresh")),
            thread_context=sanitize_user_input(state.get("thread_context", "")),
            query=sanitize_user_input(query),
            context=sanitize_user_input(context) or "[no retrieved context]",
            sender_name=sender_name,  # Assumed safe from config
            sender_email=sender_email,  # Assumed safe from config
            to=sanitize_user_input(", ".join(state.get("to") or [])),
            cc=sanitize_user_input(", ".join(state.get("cc") or [])),
            subject=sanitize_user_input(state.get("subject") or ""),
        )

        draft_structured = _complete_with_guardrails(
            messages,
            DraftGenerationOutput,
            state.get("correlation_id"),
        )

        draft = EmailDraft(
            to=draft_structured.to,
            cc=draft_structured.cc,
            subject=draft_structured.subject or "No Subject",
            body_markdown=draft_structured.body_markdown,
            tone_style=ToneStyle(
                persona_id="default", tone=state.get("tone", "professional")
            ),  # P1 Fix: Use state.tone
            val_scores=DraftValidationScores(
                factuality=0.0,
                citation_coverage=0.0,
                tone_fit=0.0,
                safety=0.0,
                overall=0.0,
                thresholds={},
            ),
            next_actions=[
                NextAction(description=na.get("description", str(na)))
                for na in draft_structured.next_actions
            ],
        )
        return {"draft": draft}
    except Exception as e:
        logger.error(f"Drafting failed: {e}")
        return {"error": f"Drafting failed: {e!s}"}


def node_critique_draft(state: dict[str, Any]) -> dict[str, Any]:
    """
    Critique the draft.

    Blueprint §10.3:
    * Review for tone, clarity, etc.
    """
    draft = state.get("draft")
    if not draft:
        return {"error": "No draft to critique"}

    try:
        messages = construct_prompt_messages(
            system_prompt_template=SYSTEM_CRITIQUE_EMAIL,
            user_prompt_template=USER_CRITIQUE_EMAIL,
            draft_subject=sanitize_user_input(draft.subject),
            draft_body=sanitize_user_input(draft.body_markdown),
            context=sanitize_user_input(state.get("assembled_context", "")),
        )

        critique = _complete_with_guardrails(
            messages,
            DraftCritique,
            state.get("correlation_id"),
        )
        return {"critique": critique}
    except Exception as e:
        logger.error(f"Critique failed: {e}")
        return {"error": f"Critique failed: {e!s}"}


def node_improve_draft(state: dict[str, Any]) -> dict[str, Any]:
    """
    Improve draft based on critique.

    Blueprint §10.3:
    * Refine based on feedback
    """
    draft = state.get("draft")
    critique = state.get("critique")
    iteration = state.get("iteration_count", 0)

    if not draft or not critique:
        return {}

    try:
        messages = construct_prompt_messages(
            system_prompt_template=SYSTEM_DRAFT_EMAIL_IMPROVE,
            user_prompt_template=USER_DRAFT_EMAIL_IMPROVE,
            original_draft=sanitize_user_input(draft.body_markdown),
            critique=critique.model_dump_json(indent=2),  # Assumed safe from system
            context=sanitize_user_input(state.get("assembled_context", "")),
        )

        draft_structured = _complete_with_guardrails(
            messages,
            DraftGenerationOutput,
            state.get("correlation_id"),
        )

        # Construct full EmailDraft with defaults, preserving original metadata where possible
        new_draft = EmailDraft(
            to=draft_structured.to or draft.to,
            cc=draft_structured.cc or draft.cc,
            subject=draft_structured.subject or draft.subject,
            body_markdown=draft_structured.body_markdown,
            tone_style=draft.tone_style,
            val_scores=DraftValidationScores(
                factuality=0.0,
                citation_coverage=0.0,
                tone_fit=0.0,
                safety=0.0,
                overall=0.0,
                thresholds={},
            ),
            next_actions=[
                NextAction(description=na.get("description", str(na)))
                for na in draft_structured.next_actions
            ],
        )

        return {"draft": new_draft, "iteration_count": iteration + 1}
    except Exception as e:
        logger.error(f"Improvement failed: {e}")
        return {"error": f"Improvement failed: {e!s}"}


async def node_select_attachments(state: dict[str, Any]) -> dict[str, Any]:
    """
    Select attachments to include in the email.

    P0-CRITICAL Implementation:
    1. Check if user explicitly asked for attachments
    2. Check if draft mentions attachments
    3. Validate files exist and are safe
    """
    draft = state.get("draft")
    if not draft or not isinstance(draft, EmailDraft):
        return {}

    # Gather context (snippets from retrieval)
    # The 'retrieval_results' in state has 'results' list, needing normalization
    retrieval_results = state.get("retrieval_results")
    snippets = []
    if retrieval_results and retrieval_results.results:
        for r in retrieval_results.results:
            snippets.append(
                {
                    "path": r.metadata.get("path"),
                    "attachment_name": r.metadata.get("filename"),
                    "doc_type": r.metadata.get("type", "unknown"),
                    "id": r.chunk_id,
                }
            )

    selected: list[dict[str, str]] = []

    # Strategy 1: Explicit mentions in the generated draft body
    body_mentions = extract_document_mentions(draft.body_markdown)
    if body_mentions:
        selected.extend(
            await _select_attachments_from_mentions(snippets, body_mentions)
        )

    # Strategy 2: If user explicitly asked in query "attach the report", etc.
    # (We rely on retrieval having found it and put it in snippets)
    query = state.get("query", "")
    query_mentions = extract_document_mentions(query)
    if query_mentions:
        # Avoid duplicates
        current_names = {s["filename"] for s in selected}
        from_query = await _select_attachments_from_mentions(snippets, query_mentions)
        for f in from_query:
            if f["filename"] not in current_names:
                selected.append(f)

    # Strategy 3: Heuristic - if query has "attach" but no specific file named,
    # grab most relevant attachment from context
    if "attach" in query.lower() and not selected:
        selected.extend(
            await _select_all_available_attachments(snippets, max_attachments=1)
        )

    # Update draft in state
    draft.attachments = selected
    return {"draft": draft}


def node_audit_draft(state: dict[str, Any]) -> dict[str, Any]:
    """
    Audit draft for policy compliance and quality rubric.

    Blueprint §10.3:
    * Policy Check (Blocker)
    * Quality/Safety Rubric (Scoring)
    """
    from cortex.context import claims_ctx

    draft = state.get("draft")
    if not draft:
        return {}

    # 1. Hard Policy Check
    recipients = list(dict.fromkeys((draft.to or []) + (draft.cc or [])))
    claims = claims_ctx.get({}) or {}

    metadata = {
        "recipients": recipients,
        "subject": draft.subject,
        "content": draft.body_markdown,
        "attachments": state.get("attachments", []),
        "check_external": True,
        "role": claims.get("role"),
        "roles_verified": bool(claims),
    }

    decision = check_action("draft_email", metadata)

    if decision.decision == "deny":
        return {
            "error": f"Draft rejected by policy: {decision.reason}",
            "policy_decision": decision.model_dump(),
        }

    # 2. LLM Rubric Audit
    try:
        messages = construct_prompt_messages(
            system_prompt_template=SYSTEM_DRAFT_EMAIL_AUDIT,
            user_prompt_template=USER_DRAFT_EMAIL_AUDIT,
            subject=sanitize_user_input(draft.subject),
            body=sanitize_user_input(draft.body_markdown),
            context=sanitize_user_input(state.get("assembled_context", "")),
        )

        scores = _complete_with_guardrails(
            messages,
            DraftValidationScores,
            state.get("correlation_id"),
        )

        # Merge scores into draft
        draft.val_scores = scores

        # If safety score is low, we might want to flag it even if policy passed
        if scores.safety < 0.7:
            logger.warning(f"Draft safety score low: {scores.safety}")

    except Exception as e:
        logger.error(f"Audit LLM failed: {e}")
        # Robustness: Fallback to neutral scores so pipeline continues
        draft.val_scores = DraftValidationScores(
            factuality=0.5,
            citation_coverage=0.5,
            tone_fit=0.5,
            safety=0.5,
            overall=0.5,
            feedback=f"Audit service unavailable (Error: {e!s}). Validation skipped.",
        )
        # Non-blocking failure for rubric, but policy passed

    result = {
        "draft": draft,
        "policy_decision": decision.model_dump(),
    }

    if decision.decision == "require_approval":
        result["approval_required"] = True

    return result


def node_load_thread(state: dict[str, Any]) -> dict[str, Any]:
    """
    Load thread context from DB.

    Blueprint §10.4:
    * Fetch messages for thread using tool_email_get_thread
    """
    thread_id = state.get("thread_id")
    tenant_id = state.get("tenant_id")

    if not thread_id:
        return {"error": "No thread_id provided"}

    # Use the standardized tool
    thread_context = tool_email_get_thread(thread_id, tenant_id)

    if not thread_context:
        return {"error": "Thread not found"}

    # Format as text for summarization
    thread_text = ""
    for msg in thread_context.messages:
        thread_text += f"From: {msg.from_addr}\n"
        thread_text += f"Date: {msg.sent_at}\n"
        thread_text += f"Subject: {msg.subject}\n\n"
        thread_text += f"{msg.body_markdown}\n\n---\n\n"

    return {
        "thread_context": thread_text,
        "_thread_context_obj": thread_context,  # Keep structured version too
    }


def node_summarize_analyst(state: dict[str, Any]) -> dict[str, Any]:
    """
    Analyst: Extract facts ledger.

    Blueprint §10.4:
    * Identify asks, commitments, dates
    """
    thread_context = state.get("thread_context")
    if not thread_context:
        return {}

    try:
        messages = construct_prompt_messages(
            system_prompt_template=SYSTEM_SUMMARIZE_ANALYST,
            user_prompt_template=USER_SUMMARIZE_ANALYST,
            thread_context=sanitize_user_input(thread_context),
        )
        facts = _complete_with_guardrails(
            messages,
            FactsLedger,
            state.get("correlation_id"),
        )
        return {"facts_ledger": facts}
    except Exception as e:
        logger.error(f"Analyst failed: {e}")
        return {"error": f"Analyst failed: {e!s}"}


def node_summarize_critic(state: dict[str, Any]) -> dict[str, Any]:
    """
    Critic: Review facts ledger.

    Blueprint §10.4:
    * Identify gaps
    """
    facts = state.get("facts_ledger")
    if not facts:
        return {}

    try:
        messages = construct_prompt_messages(
            system_prompt_template=SYSTEM_SUMMARIZE_CRITIC,
            user_prompt_template=USER_SUMMARIZE_CRITIC,
            facts_ledger_json=facts.model_dump_json(indent=2),
        )
        critique = _complete_with_guardrails(
            messages,
            CriticReview,
            state.get("correlation_id"),
        )
        return {"critique": critique}
    except Exception as e:
        logger.error(f"Critic failed: {e}")
        return {"error": f"Critic failed: {e!s}"}


def node_summarize_improver(state: dict[str, Any]) -> dict[str, Any]:
    """
    Improver: Refine facts ledger.

    Blueprint §10.4:
    * Fix gaps based on critique
    """
    facts = state.get("facts_ledger")
    critique = state.get("critique")
    thread_context = state.get("thread_context")

    if not facts or not critique:
        return {}

    try:
        messages = construct_prompt_messages(
            system_prompt_template=SYSTEM_SUMMARIZE_IMPROVER,
            user_prompt_template=USER_SUMMARIZE_IMPROVER,
            thread_context=sanitize_user_input(thread_context or ""),
            ledger=facts.model_dump_json(),
            critique=critique.model_dump_json(),
        )

        new_facts = _complete_with_guardrails(
            messages,
            FactsLedger,
            state.get("correlation_id"),
        )

        # Merge new refinement with existing facts to prevent data loss
        if facts and isinstance(facts, FactsLedger):
            merged_facts = facts.merge(new_facts)
            return {"facts_ledger": merged_facts}

        return {"facts_ledger": new_facts}
    except Exception as e:
        logger.error(f"Improver failed: {e}")
        return {"error": f"Improver failed: {e!s}"}


def node_summarize_final(state: dict[str, Any]) -> dict[str, Any]:
    """
    Finalize summary.

    Blueprint §10.4:
    * Generate final markdown summary
    """
    facts = state.get("facts_ledger")
    thread_id = state.get("thread_id")

    if not facts:
        return {"error": "No facts ledger to finalize"}

    try:
        max_len = state.get("max_length", 500)
        messages = construct_prompt_messages(
            system_prompt_template=SYSTEM_SUMMARIZE_FINAL,
            user_prompt_template=USER_SUMMARIZE_FINAL,
            max_len=max_len,
            ledger=facts.model_dump_json(),
        )

        summary_text = complete_messages(messages)

        # Prepare participant analysis by merging DB context with LLM inference
        thread_context_obj = state.get("_thread_context_obj")
        final_participants = _merge_participants(facts, thread_context_obj)

        summary = ThreadSummary(
            thread_id=thread_id,
            summary_markdown=summary_text,
            facts_ledger=facts,
            participants=final_participants,
            quality_scores={},
        )
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Finalization failed: {e}")
        return {"error": f"Finalization failed: {e!s}"}


def _merge_participants(facts: Any, thread_context_obj: Any) -> list[Any]:
    """Merge DB participant context with LLM-inferred insights."""
    if not thread_context_obj:
        return facts.participants

    from cortex.domain_models.facts_ledger import ParticipantAnalysis

    # Create lookup maps from LLM ledger
    ledger_by_email = {p.email.lower(): p for p in facts.participants if p.email}
    ledger_by_name = {p.name.lower(): p for p in facts.participants if p.name}

    return [
        _create_merged_participant(
            p, ledger_by_email, ledger_by_name, ParticipantAnalysis
        )
        for p in thread_context_obj.participants
    ]


def _create_merged_participant(
    p: Any,
    ledger_by_email: dict[str, Any],
    ledger_by_name: dict[str, Any],
    ParticipantAnalysis: type,
) -> Any:
    """Create a merged participant with DB info and LLM insights."""
    email_key = p.email.lower()

    # Start with DB info
    analysis = ParticipantAnalysis(
        name=p.name or p.email.split("@")[0],
        email=p.email,
        role="other" if p.role in ["recipient", "cc"] else "internal",
        tone="neutral",
        stance="Unknown",
    )

    # Find matching LLM insight
    match = ledger_by_email.get(email_key) or (
        ledger_by_name.get(p.name.lower()) if p.name else None
    )

    # Overlay LLM insights if found
    if match:
        analysis.role = match.role or analysis.role
        analysis.tone = match.tone or analysis.tone
        analysis.stance = match.stance or analysis.stance
        if email_key in ledger_by_email:
            analysis.name = match.name or analysis.name

    return analysis
