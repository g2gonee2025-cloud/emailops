"""
Graph Nodes.

Implements §10.2 of the Canonical Blueprint.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from cortex.config.loader import EmailOpsConfig
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
from cortex.llm.client import complete_json, complete_text
from cortex.observability import get_logger, trace_operation
from cortex.prompts import (
    PROMPT_ANSWER_QUESTION,
    PROMPT_SUMMARIZE_ANALYST,
    PROMPT_SUMMARIZE_CRITIC,
    PROMPT_SUMMARIZE_FINAL,
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
from cortex.safety import strip_injection_patterns
from cortex.security.validators import sanitize_retrieved_content, validate_file_result
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, or_, select
from sqlalchemy.orm import aliased

logger = logging.getLogger(__name__)


class DraftGenerationOutput(BaseModel):
    """Simplified schema for LLM draft generation."""

    to: List[str] = Field(description="List of recipient email addresses")
    cc: List[str] = Field(
        default_factory=list, description="List of CC email addresses"
    )
    subject: str = Field(description="Email subject line")
    body_markdown: str = Field(description="Email body in Markdown format")
    next_actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of next actions (description, owner, etc.)",
    )


def _complete_with_guardrails(
    prompt: str,
    model_cls: Type[BaseModel],
    correlation_id: Optional[str],
):
    """Run structured completion with guardrails repair fallback."""
    schema = model_cls.model_json_schema()
    raw = complete_json(prompt=prompt, schema=schema)
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
    mentions: List[str],
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


def extract_document_mentions(text: str) -> List[str]:
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

    mentions: List[str] = []
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


def _extract_entity_mentions(text: str) -> List[str]:
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


def _safe_stat_mb(path: Path) -> float:
    """Safely get file size in MB."""
    try:
        return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
    except Exception:
        return 0.0


def _select_attachments_from_mentions(
    context_snippets: List[Dict[str, Any]],
    mentions: List[str],
    *,
    max_attachments: int = 3,
) -> List[Dict[str, Any]]:
    """Select attachments that were explicitly mentioned/extracted from text."""
    cfg = get_config()
    base_dir = Path(cfg.directories.export_root) if cfg.directories.export_root else Path.cwd()
    allowed_patterns = cfg.file_patterns.allowed_file_patterns
    attach_max_mb = float(cfg.limits.max_attachment_text_chars / (1024 * 1024))
    limit = max_attachments

    if not mentions:
        return []

    wanted = {m.strip().lower() for m in mentions if m and isinstance(m, str)}
    out: List[Dict[str, Any]] = []

    for c in context_snippets or []:
        try:
            path_str = str(c.get("path") or "")
            if not path_str or not Path(path_str).suffix:
                continue

            name = str(c.get("attachment_name") or Path(path_str).name).lower()

            if name and any(w in name for w in wanted):
                p = Path(path_str)
                result = validate_file_result(str(p), base_directory=base_dir, must_exist=True)
                if not result.is_ok():
                    continue

                validated_path = result.value
                if _safe_stat_mb(validated_path) > attach_max_mb:
                    continue

                if not any(validated_path.match(pattern) for pattern in allowed_patterns):
                    continue

                out.append({"path": str(validated_path), "filename": validated_path.name})
                if len(out) >= int(limit):
                    break
        except Exception:
            continue
    return out


def _select_all_available_attachments(
    context_snippets: List[Dict[str, Any]], *, max_attachments: int = 3
) -> List[Dict[str, Any]]:
    """Select any valid attachments found in context (heuristic fallback)."""
    cfg = get_config()
    base_dir = Path(cfg.directories.export_root) if cfg.directories.export_root else Path.cwd()
    allowed_patterns = cfg.file_patterns.allowed_file_patterns
    attach_max_mb = float(cfg.limits.max_attachment_text_chars / (1024 * 1024))

    selected: List[Dict[str, Any]] = []
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

            result = validate_file_result(str(p), base_directory=base_dir, must_exist=True)
            if not result.is_ok():
                continue

            validated_path = result.value
            if not any(validated_path.match(pattern) for pattern in allowed_patterns):
                continue

            if _safe_stat_mb(validated_path) > attach_max_mb:
                continue

            selected.append({"path": str(validated_path), "filename": validated_path.name})
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
) -> Optional[ThreadContext]:
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

    # Ensure thread_id is a UUID
    if isinstance(thread_id, str):
        try:
            thread_id = uuid.UUID(thread_id)
        except ValueError:
            logger.error(f"Invalid thread_id format: {thread_id}")
            return None

    try:
        with SessionLocal() as session:
            from cortex.db.session import set_session_tenant

            set_session_tenant(session, tenant_id)

            # Fetch conversation (which contains messages as JSONB)
            conversation = (
                session.query(Conversation)
                .filter(
                    Conversation.conversation_id == thread_id,
                    Conversation.tenant_id == tenant_id,
                )
                .first()
            )

            if not conversation:
                logger.warning(f"Conversation not found: {thread_id}")
                return None

            # Build participants from conversation.participants JSONB
            participants_dict: Dict[str, ThreadParticipant] = {}
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
            thread_messages: List[ThreadMessage] = []
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
                        except Exception:
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
                        message_id=str(thread_id),
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
                thread_id=thread_id,
                subject=conversation.subject or "",
                participants=list(participants_dict.values()),
                messages=thread_messages,
            )

    except Exception as e:
        logger.error(f"Failed to fetch thread {thread_id}: {e}", exc_info=True)
        return None


def _extract_evidence_from_answer(
    answer_text: str, retrieval_results: SearchResults | None
) -> List[EvidenceItem]:
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


def node_handle_error(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle errors in graph execution.

    Blueprint §10.5:
    * records error details into audit_log
    * logs via observability.get_logger
    * sets state.error
    """
    error = state.get("error", "Unknown error")
    obs_logger = get_logger(__name__)

    obs_logger.error(
        f"Graph execution error: {error}", extra={"tenant_id": state.get("tenant_id")}
    )

    # Record error to audit_log
    try:
        log_audit_event(
            tenant_id=state.get("tenant_id", "unknown"),
            user_or_agent=state.get("user_id", "system"),
            action="graph_error",
            risk_level="medium",
            metadata={
                "error": str(error),
                "query": state.get("query", ""),
                "graph_type": state.get("_graph_type", "unknown"),
            },
        )
    except Exception as e:
        obs_logger.warning(f"Failed to log audit event: {e}")

    return {"error": str(error)}


def node_assemble_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assemble context from retrieval results.

    Blueprint §10.1.1:
    * Redundant cleaning optimization
    * Proactive injection defense
    """
    results = state.get("retrieval_results")
    if not results or not results.results:
        return {"assembled_context": ""}

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

    return {"assembled_context": "\n\n".join(context_parts)}


def node_query_graph(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Query the knowledge graph for entities mentioned in the prompt.

    Blueprint §10.1:
    * Inject relational facts into answer context.
    """
    query = state.get("query", "")
    tenant_id = state.get("tenant_id")

    if not query or not tenant_id:
        return {"graph_context": ""}

    mentions = _extract_entity_mentions(query)
    if not mentions:
        return {"graph_context": ""}

    try:
        nodes, edges = _fetch_graph_entities(tenant_id, mentions)
        if not nodes:
            return {"graph_context": ""}

        context_lines = _build_graph_context_lines(nodes, edges)
        if not context_lines:
            return {"graph_context": ""}

        graph_context = "Knowledge Graph Facts:\n" + "\n".join(
            f"- {line}" for line in context_lines
        )
        return {"graph_context": graph_context}
    except Exception as e:
        logger.error(f"Graph query failed: {e}")
        return {"graph_context": ""}


def _fetch_graph_entities(
    tenant_id: str, mentions: List[str]
) -> tuple[List[Any], List[Any]]:
    """Fetch entity nodes and edges from the knowledge graph."""
    from cortex.db.models import EntityEdge, EntityNode
    from cortex.db.session import SessionLocal, set_session_tenant

    with SessionLocal() as session:
        set_session_tenant(session, tenant_id)

        match_conditions = [
            func.lower(EntityNode.name) == mention.lower() for mention in mentions
        ]

        for mention in mentions:
            if len(mention) >= 4:
                match_conditions.append(
                    func.lower(EntityNode.name).like(f"%{mention.lower()}%")
                )

        if not match_conditions:
            return [], []

        nodes = (
            session.execute(
                select(EntityNode).where(
                    EntityNode.tenant_id == tenant_id,
                    or_(*match_conditions),
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


def _build_graph_context_lines(nodes: List[Any], edges: List[Any]) -> List[str]:
    """Build context lines from graph nodes and edges."""
    context_lines: List[str] = []

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


def node_classify_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify the user query.

    Blueprint §8.1:
    * Determine intent (navigational, semantic, drafting)
    """
    query = strip_injection_patterns(state.get("query", ""))

    # Update state with sanitized query
    state["query"] = query

    try:
        args = QueryClassificationInput(query=query, use_llm=True)
        classification = tool_classify_query(args)
        return {"classification": classification}
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return {
            "classification": QueryClassification(
                query=query, type="semantic", flags=["classification_failed"]
            )
        }


def node_retrieve_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve context based on query and classification.

    Blueprint §8.3:
    * Call Hybrid Search
    """
    query = state.get("query", "")
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
        results = tool_kb_search_hybrid(args)
        return {"retrieval_results": results}
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {"error": f"Retrieval failed: {str(e)}"}


def node_generate_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate answer using LLM and context.

    Blueprint §3.6:
    * Generate response with citations
    """
    query = state.get("query", "")
    context = state.get("assembled_context", "")
    graph_context = state.get("graph_context", "")
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

        prompt = (
            PROMPT_ANSWER_QUESTION
            + f"\n\nContext:\n{combined_context}\n\nQuestion: {query}"
        )

        answer_text = complete_text(prompt)

        retrieval_results: Optional[SearchResults] = state.get("retrieval_results")

        # Extract evidence from answer text and retrieval results
        evidence = _extract_evidence_from_answer(answer_text, retrieval_results)

        # Calculate confidence based on evidence quality
        confidence = min(0.95, 0.5 + 0.1 * len(evidence)) if evidence else 0.6

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
        return {"error": f"Generation failed: {str(e)}"}


def node_prepare_draft_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare query for drafting.

    Blueprint §10.3:
    * Combine explicit query and thread context
    """
    explicit_query = strip_injection_patterns(state.get("explicit_query", ""))

    # Update state with sanitized query
    state["explicit_query"] = explicit_query

    # If we had thread context, we would combine it here.
    # For now, just use explicit query as the search query.
    return {"query": explicit_query}


def node_draft_email_initial(state: Dict[str, Any]) -> Dict[str, Any]:
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
        prompt = get_prompt(
            "DRAFT_EMAIL_INITIAL",
            mode=state.get("mode", "fresh"),
            thread_context=state.get("thread_context", ""),
            query=query,
            context=context or "[no retrieved context]",
            sender_name=sender_name,
            sender_email=sender_email,
            to=", ".join(state.get("to") or []),
            cc=", ".join(state.get("cc") or []),
            subject=state.get("subject") or "",
        )

        draft_structured = _complete_with_guardrails(
            prompt,
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
        return {"error": f"Drafting failed: {str(e)}"}


def node_critique_draft(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Critique the draft.

    Blueprint §10.3:
    * Review for tone, clarity, etc.
    """
    draft = state.get("draft")
    if not draft:
        return {"error": "No draft to critique"}

    try:
        prompt = get_prompt(
            "DRAFT_EMAIL_CRITIQUE",
            draft_subject=draft.subject,
            draft_body=draft.body_markdown,
            context=state.get("assembled_context", ""),
        )

        critique = _complete_with_guardrails(
            prompt,
            DraftCritique,
            state.get("correlation_id"),
        )
        return {"critique": critique}
    except Exception as e:
        logger.error(f"Critique failed: {e}")
        return {"error": f"Critique failed: {str(e)}"}


def node_improve_draft(state: Dict[str, Any]) -> Dict[str, Any]:
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
        prompt = get_prompt(
            "DRAFT_EMAIL_IMPROVE",
            original_draft=draft.body_markdown,
            critique=critique.model_dump_json(indent=2),
            context=state.get("assembled_context", ""),
        )

        draft_structured = _complete_with_guardrails(
            prompt,
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
        return {"error": f"Improvement failed: {str(e)}"}


def node_select_attachments(state: Dict[str, Any]) -> Dict[str, Any]:
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

    selected: List[Dict[str, str]] = []

    # Strategy 1: Explicit mentions in the generated draft body
    body_mentions = extract_document_mentions(draft.body_markdown)
    if body_mentions:
        selected.extend(_select_attachments_from_mentions(snippets, body_mentions))

    # Strategy 2: If user explicitly asked in query "attach the report", etc.
    # (We rely on retrieval having found it and put it in snippets)
    query = state.get("query", "")
    query_mentions = extract_document_mentions(query)
    if query_mentions:
        # Avoid duplicates
        current_names = {s["filename"] for s in selected}
        from_query = _select_attachments_from_mentions(snippets, query_mentions)
        for f in from_query:
            if f["filename"] not in current_names:
                selected.append(f)

    # Strategy 3: Heuristic - if query has "attach" but no specific file named,
    # grab most relevant attachment from context
    if "attach" in query.lower() and not selected:
        selected.extend(_select_all_available_attachments(snippets, max_attachments=1))

    # Update draft in state
    draft.attachments = selected
    return {"draft": draft}


def node_audit_draft(state: Dict[str, Any]) -> Dict[str, Any]:
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
    }

    decision = check_action("draft_email", metadata)

    if decision.decision == "deny":
        return {
            "error": f"Draft rejected by policy: {decision.reason}",
            "policy_decision": decision.model_dump(),
        }

    # 2. LLM Rubric Audit
    try:
        prompt = get_prompt(
            "DRAFT_EMAIL_AUDIT",
            subject=draft.subject,
            body=draft.body_markdown,
            context=state.get("assembled_context", ""),
        )

        scores = _complete_with_guardrails(
            prompt,
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
            feedback=f"Audit service unavailable (Error: {str(e)}). Validation skipped.",
        )
        # Non-blocking failure for rubric, but policy passed

    result = {
        "draft": draft,
        "policy_decision": decision.model_dump(),
    }

    if decision.decision == "require_approval":
        result["approval_required"] = True

    return result


def node_load_thread(state: Dict[str, Any]) -> Dict[str, Any]:
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


def node_summarize_analyst(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyst: Extract facts ledger.

    Blueprint §10.4:
    * Identify asks, commitments, dates
    """
    thread_context = state.get("thread_context")
    if not thread_context:
        return {}

    try:
        prompt = PROMPT_SUMMARIZE_ANALYST + f"\n\nThread:\n{thread_context}"
        facts = _complete_with_guardrails(
            prompt,
            FactsLedger,
            state.get("correlation_id"),
        )
        return {"facts_ledger": facts}
    except Exception as e:
        logger.error(f"Analyst failed: {e}")
        return {"error": f"Analyst failed: {str(e)}"}


def node_summarize_critic(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Critic: Review facts ledger.

    Blueprint §10.4:
    * Identify gaps
    """
    facts = state.get("facts_ledger")
    if not facts:
        return {}

    try:
        prompt = (
            PROMPT_SUMMARIZE_CRITIC
            + f"\n\nFacts Ledger:\n{facts.model_dump_json(indent=2)}"
        )
        critique = _complete_with_guardrails(
            prompt,
            CriticReview,
            state.get("correlation_id"),
        )
        return {"critique": critique}
    except Exception as e:
        logger.error(f"Critic failed: {e}")
        return {"error": f"Critic failed: {str(e)}"}


def node_summarize_improver(state: Dict[str, Any]) -> Dict[str, Any]:
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
        prompt = get_prompt(
            "SUMMARIZE_IMPROVER",
            thread_context=thread_context or "",
            ledger=facts.model_dump_json(),
            critique=critique.model_dump_json(),
        )

        new_facts = _complete_with_guardrails(
            prompt,
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
        return {"error": f"Improver failed: {str(e)}"}


def node_summarize_final(state: Dict[str, Any]) -> Dict[str, Any]:
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
        prompt = (
            PROMPT_SUMMARIZE_FINAL
            + f"\n\nConstraint: Keep summary under {max_len} words."
            + f"\n\nFacts Ledger:\n{facts.model_dump_json()}"
        )

        summary_text = complete_text(prompt)

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
        return {"error": f"Finalization failed: {str(e)}"}


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
