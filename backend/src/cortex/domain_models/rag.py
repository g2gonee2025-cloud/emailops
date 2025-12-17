"""
RAG Domain Models.

Implements §10 of the Canonical Blueprint - data models for orchestration.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from cortex.domain_models.facts_ledger import FactsLedger


class ThreadParticipant(BaseModel):
    """Participant in an email thread."""

    email: str
    name: Optional[str] = None
    role: Literal["sender", "recipient", "cc"] = "recipient"


class ThreadMessage(BaseModel):
    """Single message in a thread."""

    message_id: Optional[UUID] = None
    sent_at: Optional[datetime] = None
    recv_at: Optional[datetime] = None
    from_addr: str = ""
    to_addrs: List[str] = Field(default_factory=list)
    cc_addrs: List[str] = Field(default_factory=list)
    subject: str = ""
    body_markdown: str = ""
    is_inbound: bool = False


class ThreadContext(BaseModel):
    """Full context for an email thread."""

    thread_id: Optional[UUID] = None
    subject: str = ""
    participants: List[ThreadParticipant] = Field(default_factory=list)
    messages: List[ThreadMessage] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """Evidence item from retrieval results."""

    chunk_id: str
    text: str = ""
    relevance_score: float = 0.0
    source_type: Literal["email", "attachment", "other"] = "email"


class RetrievalDiagnostics(BaseModel):
    """Diagnostics for retrieval results."""

    lexical_score: float = 0.0
    vector_score: float = 0.0
    fused_rank: int = 0
    reranker: Optional[str] = None


class Answer(BaseModel):
    """Generated answer with evidence."""

    query: str
    answer_markdown: str = ""
    evidence: List[EvidenceItem] = Field(default_factory=list)
    confidence_overall: float = 0.0
    safety: Dict[str, Any] = Field(default_factory=dict)
    retrieval_diagnostics: List[RetrievalDiagnostics] = Field(default_factory=list)


class ToneStyle(BaseModel):
    """Tone and style configuration for emails."""

    persona_id: str = "default"
    tone: str = "professional"


class DraftValidationScores(BaseModel):
    """Validation scores for a draft."""

    factuality: float = 0.0
    citation_coverage: float = 0.0
    tone_fit: float = 0.0
    safety: float = 0.0
    overall: float = 0.0
    thresholds: Dict[str, float] = Field(default_factory=dict)


class NextAction(BaseModel):
    """Next action item extracted from email."""

    description: str
    owner: Optional[str] = None
    due_date: Optional[str] = None


class EmailDraft(BaseModel):
    """Generated email draft."""

    to: List[str] = Field(default_factory=list)
    cc: List[str] = Field(default_factory=list)
    subject: str = ""
    body_markdown: str = ""
    tone_style: ToneStyle = Field(default_factory=ToneStyle)
    val_scores: DraftValidationScores = Field(default_factory=DraftValidationScores)
    next_actions: List[NextAction] = Field(default_factory=list)


class DraftCritique(BaseModel):
    """Critique of a draft email."""

    overall_rating: float = 0.0
    tone_feedback: str = ""
    clarity_feedback: str = ""
    completeness_feedback: str = ""
    suggestions: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)


class ThreadSummary(BaseModel):
    """Summary of an email thread."""

    type: Literal["thread_summary"] = "thread_summary"
    thread_id: Optional[UUID] = None
    summary_markdown: str = ""
    facts_ledger: FactsLedger = Field(default_factory=FactsLedger)
    quality_scores: Dict[str, Any] = Field(default_factory=dict)
    key_points: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    participants_mentioned: List[str] = Field(default_factory=list)
