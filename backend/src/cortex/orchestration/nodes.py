"""
Graph Nodes.

Implements §10.2 of the Canonical Blueprint.
"""
from __future__ import annotations

import logging
import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from cortex.observability import get_logger, trace_operation
from cortex.safety.injection_defense import strip_injection_patterns
from cortex.safety.policy_enforcer import check_action
from cortex.orchestration.states import AnswerQuestionState
from cortex.retrieval.query_classifier import QueryClassification, tool_classify_query
from cortex.retrieval.hybrid_search import tool_kb_search_hybrid, KBSearchInput, SearchResults
from cortex.llm.client import complete_json, complete_text
from cortex.prompts import (
    PROMPT_QUERY_CLASSIFY, PROMPT_ANSWER_QUESTION,
    PROMPT_DRAFT_EMAIL_INITIAL, PROMPT_DRAFT_EMAIL_IMPROVE,
    PROMPT_CRITIQUE_EMAIL, PROMPT_DRAFT_EMAIL_AUDIT,
    PROMPT_SUMMARIZE_ANALYST, PROMPT_SUMMARIZE_CRITIC,
    PROMPT_SUMMARIZE_IMPROVER, PROMPT_SUMMARIZE_FINAL
)
from cortex.models.rag import (
    Answer, EvidenceItem, RetrievalDiagnostics,
    EmailDraft, DraftCritique, DraftValidationScores, ToneStyle,
    ThreadSummary, ThreadContext, ThreadMessage, ThreadParticipant,
    NextAction
)
from cortex.models.facts_ledger import FactsLedger, CriticReview
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DraftGenerationOutput(BaseModel):
    """Simplified schema for LLM draft generation."""
    to: List[str] = Field(description="List of recipient email addresses")
    cc: List[str] = Field(default=[], description="List of CC email addresses")
    subject: str = Field(description="Email subject line")
    body_markdown: str = Field(description="Email body in Markdown format")
    next_actions: List[Dict[str, Any]] = Field(description="List of next actions (description, owner, etc.)")


# -----------------------------------------------------------------------------
# Helper Tools (§10.2)
# -----------------------------------------------------------------------------

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
    # Matches: report.pdf, Budget_2024.xlsx, meeting-notes.docx, etc.
    file_extension_pattern = re.compile(
        r'\b([\w\-_.]+\.(?:pdf|docx?|xlsx?|pptx?|txt|csv|eml|msg|png|jpg|jpeg|gif))\b',
        re.IGNORECASE
    )
    
    for match in file_extension_pattern.finditer(text):
        filename = match.group(1)
        lower_name = filename.lower()
        if lower_name not in seen:
            seen.add(lower_name)
            mentions.append(filename)
    
    # Pattern 2: Quoted file references
    # Matches: "Budget Report", 'Q4 Summary', etc.
    quoted_pattern = re.compile(r'["\']([A-Z][A-Za-z0-9\s_\-]{2,30})["\']')
    
    for match in quoted_pattern.finditer(text):
        ref = match.group(1).strip()
        lower_ref = ref.lower()
        # Only include if it looks like a document name (has capital or underscore)
        if lower_ref not in seen and ('_' in ref or any(c.isupper() for c in ref)):
            seen.add(lower_ref)
            mentions.append(ref)
    
    # Pattern 3: Explicit attachment references
    # Matches: "attached report", "the attached file", "see attachment: xyz"
    attachment_ref_pattern = re.compile(
        r'(?:attached|attachment|enclosed|enclosure)[:\s]+([A-Za-z0-9\s_\-]+?)(?:\.|,|\s|$)',
        re.IGNORECASE
    )
    
    for match in attachment_ref_pattern.finditer(text):
        ref = match.group(1).strip()
        if ref and len(ref) > 2:
            lower_ref = ref.lower()
            if lower_ref not in seen:
                seen.add(lower_ref)
                mentions.append(ref)
    
    return mentions


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
    * Retrieves all messages for a thread
    * Builds ThreadContext with participants and messages
    
    Args:
        thread_id: UUID of the thread to fetch
        tenant_id: Tenant ID for RLS
        include_attachments: Whether to include attachment info
        
    Returns:
        ThreadContext with full thread data, or None if not found
    """
    from cortex.db.session import SessionLocal
    from cortex.db.models import Thread, Message, Attachment
    
    # Ensure thread_id is a UUID
    if isinstance(thread_id, str):
        try:
            thread_id = uuid.UUID(thread_id)
        except ValueError:
            logger.error(f"Invalid thread_id format: {thread_id}")
            return None
    
    try:
        with SessionLocal() as session:
            # Fetch thread
            thread = session.query(Thread).filter(
                Thread.thread_id == thread_id,
                Thread.tenant_id == tenant_id
            ).first()
            
            if not thread:
                logger.warning(f"Thread not found: {thread_id}")
                return None
            
            # Fetch messages ordered by sent_at
            messages_query = session.query(Message).filter(
                Message.thread_id == thread_id,
                Message.tenant_id == tenant_id
            ).order_by(Message.sent_at.asc())
            
            messages = messages_query.all()
            
            if not messages:
                logger.warning(f"No messages found for thread: {thread_id}")
                return None
            
            # Build participants set
            participants_dict: Dict[str, ThreadParticipant] = {}
            
            # Build thread messages
            thread_messages: List[ThreadMessage] = []
            
            for msg in messages:
                # Track participants
                if msg.from_addr and msg.from_addr not in participants_dict:
                    participants_dict[msg.from_addr] = ThreadParticipant(
                        email=msg.from_addr,
                        name=None,  # Could parse from headers if available
                        role="sender"
                    )
                
                for addr in (msg.to_addrs or []):
                    if addr not in participants_dict:
                        participants_dict[addr] = ThreadParticipant(
                            email=addr,
                            name=None,
                            role="recipient"
                        )
                
                for addr in (msg.cc_addrs or []):
                    if addr not in participants_dict:
                        participants_dict[addr] = ThreadParticipant(
                            email=addr,
                            name=None,
                            role="cc"
                        )
                
                # Determine if inbound (not from our domain)
                # Simple heuristic: first message sender is "us", replies are "them"
                is_inbound = False
                if thread_messages:
                    # If sender differs from first message sender, it's inbound
                    first_sender = thread_messages[0].from_addr
                    is_inbound = msg.from_addr != first_sender
                
                # Build body markdown from plain text
                body_md = msg.body_plain or ""
                if not body_md and msg.body_html:
                    # Fallback: strip HTML tags (basic)
                    body_md = re.sub(r'<[^>]+>', '', msg.body_html)
                
                thread_messages.append(ThreadMessage(
                    message_id=msg.message_id,
                    sent_at=msg.sent_at,
                    recv_at=msg.recv_at,
                    from_addr=msg.from_addr,
                    to_addrs=msg.to_addrs or [],
                    cc_addrs=msg.cc_addrs or [],
                    subject=msg.subject or "",
                    body_markdown=body_md,
                    is_inbound=is_inbound
                ))
            
            return ThreadContext(
                thread_id=thread_id,
                subject=thread.original_subject or thread.subject_norm or "",
                participants=list(participants_dict.values()),
                messages=thread_messages
            )
            
    except Exception as e:
        logger.error(f"Failed to fetch thread {thread_id}: {e}", exc_info=True)
        return None


def _extract_evidence_from_answer(
    answer_text: str,
    retrieval_results: SearchResults | None
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
    source_refs = re.findall(r'\[?Source\s*(\d+)\]?', answer_text, re.IGNORECASE)
    referenced_indices = {int(ref) - 1 for ref in source_refs}  # Convert to 0-based
    
    for i, result in enumerate(retrieval_results.results):
        # Include if explicitly referenced, or if in top 3 results
        if i in referenced_indices or i < 3:
            chunk_id = result.chunk_id or result.message_id or f"result_{i}"
            
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            
            evidence.append(EvidenceItem(
                chunk_id=chunk_id,
                text=result.highlights[0] if result.highlights else "",
                relevance_score=result.score,
                source_type="email" if result.message_id else "attachment"
            ))
    
    return evidence[:5]  # Limit to top 5 evidence items


def node_handle_error(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle errors in graph execution.
    
    Blueprint §10.5:
    * records error details into audit_log
    * logs via observability.get_logger
    * sets state.error
    """
    from cortex.audit import log_audit_event
    
    error = state.get("error", "Unknown error")
    obs_logger = get_logger(__name__)
    
    obs_logger.error(f"Graph execution error: {error}", extra={"tenant_id": state.get("tenant_id")})
    
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
                "graph_type": state.get("_graph_type", "unknown")
            }
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
        safe_text = strip_injection_patterns(text)
        
        # Format with metadata for citation
        # We use a simple index or ID reference for the LLM
        source_ref = f"Source {i+1} (ID: {item.chunk_id or item.message_id})"
        context_parts.append(f"[{source_ref}]\n{safe_text}")
        
    return {"assembled_context": "\n\n".join(context_parts)}


def node_classify_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify the user query.
    
    Blueprint §8.1:
    * Determine intent (navigational, semantic, drafting)
    """
    query = state.get("query", "")
    
    try:
        # Use complete_json to get structured classification
        classification_dict = complete_json(
            prompt=PROMPT_QUERY_CLASSIFY + f"\nQuery: {query}",
            schema=QueryClassification.model_json_schema()
        )
        classification = QueryClassification(**classification_dict)
        return {"classification": classification}
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        # Fallback to semantic if classification fails
        return {
            "classification": QueryClassification(
                query=query,
                type="semantic",
                flags=["classification_failed"]
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
    
    try:
        args = KBSearchInput(
            tenant_id=tenant_id,
            user_id=user_id,
            query=query,
            classification=classification
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
    
    if not context:
        return {
            "answer": Answer(
                query=query,
                answer_markdown="I could not find any relevant information to answer your question.",
                evidence=[],
                confidence_overall=0.0,
                safety={},
                retrieval_diagnostics=[]
            )
        }
        
    try:
        # We want the LLM to generate the answer text.
        # Constructing the full Answer object via JSON might be too complex for the LLM 
        # to get the EvidenceItem UUIDs right without very specific prompting.
        # For now, let's get the text answer and construct a basic Answer object.
        
        prompt = PROMPT_ANSWER_QUESTION + f"\n\nContext:\n{context}\n\nQuestion: {query}"
        
        answer_text = complete_text(prompt)
        
        # Extract evidence from answer text and retrieval results
        evidence = _extract_evidence_from_answer(
            answer_text, 
            state.get("retrieval_results")
        )
        
        # Calculate confidence based on evidence quality
        confidence = min(0.95, 0.5 + 0.1 * len(evidence)) if evidence else 0.6
        
        return {
            "answer": Answer(
                query=query,
                answer_markdown=answer_text,
                evidence=evidence,
                confidence_overall=confidence,
                safety={},
                retrieval_diagnostics=[]
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
    explicit_query = state.get("explicit_query", "")
    # If we had thread context, we would combine it here.
    # For now, just use explicit query as the search query.
    return {"query": explicit_query}


def node_draft_email_initial(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate initial email draft.
    
    Blueprint §10.3:
    * Draft based on context and query
    """
    query = state.get("explicit_query", "")
    context = state.get("assembled_context", "")
    
    try:
        prompt = PROMPT_DRAFT_EMAIL_INITIAL + f"\n\nContext:\n{context}\n\nInstructions: {query}"
        
        # Use simplified schema for generation
        draft_dict = complete_json(
            prompt=prompt,
            schema=DraftGenerationOutput.model_json_schema()
        )
        
        # Construct full EmailDraft with defaults
        draft = EmailDraft(
            to=draft_dict.get("to", []),
            cc=draft_dict.get("cc", []),
            subject=draft_dict.get("subject", "No Subject"),
            body_markdown=draft_dict.get("body_markdown", ""),
            tone_style=ToneStyle(persona_id="default", tone="professional"),
            val_scores=DraftValidationScores(
                factuality=0.0, citation_coverage=0.0, tone_fit=0.0, safety=0.0, overall=0.0, thresholds={}
            ),
            next_actions=[
                NextAction(description=na.get("description", str(na))) 
                for na in draft_dict.get("next_actions", [])
            ]
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
        prompt = PROMPT_CRITIQUE_EMAIL + f"\n\nDraft Subject: {draft.subject}\nDraft Body:\n{draft.body_markdown}"
        
        critique_dict = complete_json(
            prompt=prompt,
            schema=DraftCritique.model_json_schema()
        )
        critique = DraftCritique(**critique_dict)
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
        prompt = PROMPT_DRAFT_EMAIL_IMPROVE + f"\n\nOriginal Draft:\n{draft.body_markdown}\n\nCritique:\n{critique.overall_comment}\nIssues: {critique.issues}"
        
        # Use simplified schema for generation
        draft_dict = complete_json(
            prompt=prompt,
            schema=DraftGenerationOutput.model_json_schema()
        )
        
        # Construct full EmailDraft with defaults, preserving original metadata where possible
        new_draft = EmailDraft(
            to=draft_dict.get("to", draft.to),
            cc=draft_dict.get("cc", draft.cc),
            subject=draft_dict.get("subject", draft.subject),
            body_markdown=draft_dict.get("body_markdown", ""),
            tone_style=draft.tone_style,
            val_scores=DraftValidationScores(
                factuality=0.0, citation_coverage=0.0, tone_fit=0.0, safety=0.0, overall=0.0, thresholds={}
            ),
            next_actions=[
                NextAction(description=na.get("description", str(na))) 
                for na in draft_dict.get("next_actions", [])
            ]
        )
        
        return {
            "draft": new_draft,
            "iteration_count": iteration + 1
        }
    except Exception as e:
        logger.error(f"Improvement failed: {e}")
        return {"error": f"Improvement failed: {str(e)}"}


def node_audit_draft(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Audit draft for policy compliance.
    
    Blueprint §10.3:
    * Final check before returning
    """
    draft = state.get("draft")
    if not draft:
        return {}
        
    recipients = list(dict.fromkeys((draft.to or []) + (draft.cc or [])))

    metadata = {
        "recipients": recipients,
        "subject": draft.subject,
        "content": draft.body_markdown,
        "attachments": state.get("attachments", []),
        "check_external": True,
    }

    decision = check_action("draft_email", metadata)
    decision_payload = decision.model_dump()

    if decision.decision == "deny":
        return {
            "error": f"Draft rejected by policy: {decision.reason}",
            "policy_decision": decision_payload,
        }

    result = {
        "draft": draft,
        "policy_decision": decision_payload,
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
        "_thread_context_obj": thread_context  # Keep structured version too
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
        
        # We want a structured facts ledger
        facts_dict = complete_json(
            prompt=prompt,
            schema=FactsLedger.model_json_schema()
        )
        facts = FactsLedger(**facts_dict)
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
        prompt = PROMPT_SUMMARIZE_CRITIC + f"\n\nFacts Ledger:\n{facts.model_dump_json(indent=2)}"
        
        critique_dict = complete_json(
            prompt=prompt,
            schema=CriticReview.model_json_schema()
        )
        critique = CriticReview(**critique_dict)
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
        prompt = PROMPT_SUMMARIZE_IMPROVER + f"\n\nThread:\n{thread_context}\n\nFacts Ledger:\n{facts.model_dump_json()}\n\nCritique:\n{critique.model_dump_json()}"
        
        new_facts_dict = complete_json(
            prompt=prompt,
            schema=FactsLedger.model_json_schema()
        )
        new_facts = FactsLedger(**new_facts_dict)
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
        prompt = PROMPT_SUMMARIZE_FINAL + f"\n\nFacts Ledger:\n{facts.model_dump_json()}"
        
        summary_text = complete_text(prompt)
        
        summary = ThreadSummary(
            thread_id=thread_id,
            summary_markdown=summary_text,
            facts_ledger=facts,
            quality_scores={}
        )
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Finalization failed: {e}")
        return {"error": f"Finalization failed: {str(e)}"}
