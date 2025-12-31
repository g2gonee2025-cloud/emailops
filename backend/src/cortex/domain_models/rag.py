"""
RAG Domain Models.

Implements §10 of the Canonical Blueprint - data models for orchestration.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Literal
from uuid import UUID

from cortex.common.models import SecureBaseModel
from cortex.domain_models.facts_ledger import FactsLedger, ParticipantAnalysis
from pydantic import BaseModel, ConfigDict, Field


class ThreadParticipant(SecureBaseModel):
    """Participant in an email thread."""

    _PII_FIELDS: ClassVar[set[str]] = {"email", "name"}

    email: str
    name: str | None = None
    role: Literal["sender", "recipient", "cc"] = "recipient"


class ThreadMessage(SecureBaseModel):
    """Single message in a thread."""

    _PII_FIELDS: ClassVar[set[str]] = {
        "from_addr",
        "to_addrs",
        "cc_addrs",
        "subject",
        "body_markdown",
    }

    message_id: UUID | None = None
    sent_at: datetime | None = None
    recv_at: datetime | None = None
    from_addr: str = ""
    to_addrs: list[str] = Field(default_factory=list)
    cc_addrs: list[str] = Field(default_factory=list)
    subject: str = ""
    body_markdown: str = ""
    is_inbound: bool = False


class ThreadContext(SecureBaseModel):
    """Full context for an email thread."""

    thread_id: UUID | None = None
    subject: str = ""
    participants: list[ThreadParticipant] = Field(default_factory=list)
    messages: list[ThreadMessage] = Field(default_factory=list)


class EvidenceItem(SecureBaseModel):
    """Evidence item from retrieval results."""

    _PII_FIELDS: ClassVar[set[str]] = {"text"}

    chunk_id: str
    text: str = ""
    relevance_score: float = 0.0
    source_type: Literal["email", "attachment", "other"] = "email"


class RetrievalDiagnostics(SecureBaseModel):
    """Diagnostics for retrieval results."""

    lexical_score: float = 0.0
    vector_score: float = 0.0
    fused_rank: int = 0
    reranker: str | None = None


class Answer(SecureBaseModel):
    """Generated answer with evidence."""

    _PII_FIELDS: ClassVar[set[str]] = {"query", "answer_markdown"}

    query: str
    answer_markdown: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)
    confidence_overall: float = 0.0
    safety: dict[str, Any] = Field(default_factory=dict)
    retrieval_diagnostics: list[RetrievalDiagnostics] = Field(default_factory=list)


class ToneStyle(SecureBaseModel):
    """Tone and style configuration for emails."""

    persona_id: str = "default"
    tone: str = "professional"


class DraftValidationScores(SecureBaseModel):
    """Validation scores for a draft."""

    _PII_FIELDS: ClassVar[set[str]] = {"feedback"}

    factuality: float = 0.0
    citation_coverage: float = 0.0
    tone_fit: float = 0.0
    safety: float = 0.0
    overall: float = 0.0
    thresholds: dict[str, float] = Field(default_factory=dict)
    feedback: str | None = None


class NextAction(SecureBaseModel):
    """Next action item extracted from email."""

    _PII_FIELDS: ClassVar[set[str]] = {"description", "owner"}


class AttachmentRef(SecureBaseModel):
    """Reference to an attachment for inclusion."""

    _PII_FIELDS: ClassVar[set[str]] = {"path", "filename"}

    path: str
    filename: str

    description: str
    owner: str | None = None
    due_date: str | None = None


class EmailDraft(SecureBaseModel):
    """Generated email draft."""

    _PII_FIELDS: ClassVar[set[str]] = {
        "to",
        "cc",
        "subject",
        "body_markdown",
        "attachments",
    }

    to: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)
    subject: str = ""
    body_markdown: str = ""
    tone_style: ToneStyle = Field(default_factory=ToneStyle)
    val_scores: DraftValidationScores = Field(default_factory=DraftValidationScores)
    next_actions: list[NextAction] = Field(default_factory=list)
    attachments: list[AttachmentRef] = Field(
        default_factory=list,
        description="List of attachments with 'path' and 'filename'",
    )


class Issue(SecureBaseModel):
    """Critique issue of a draft email."""

    _PII_FIELDS: ClassVar[set[str]] = {"description"}

    description: str
    severity: str = "medium"
    category: str = "general"


class DraftCritique(SecureBaseModel):
    """Critique of a draft email."""

    overall_rating: float = 0.0
    tone_feedback: str = ""
    clarity_feedback: str = ""
    completeness_feedback: str = ""
    suggestions: list[str] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)


class ThreadSummary(SecureBaseModel):
    """Summary of an email thread."""

    model_config = ConfigDict(populate_by_name=True)

    _PII_FIELDS: ClassVar[set[str]] = {
        "summary_markdown",
        "key_points",
        "action_items",
        "participants",
    }

    summary_type: Literal["thread_summary"] = Field(
        default="thread_summary", alias="type"
    )
    thread_id: UUID | None = None
    summary_markdown: str = ""
    facts_ledger: FactsLedger = Field(default_factory=FactsLedger)
    quality_scores: dict[str, Any] = Field(default_factory=dict)
    key_points: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    participants: list[ParticipantAnalysis] = Field(default_factory=list)

    @property
    def type(self) -> str:
        return self.summary_type
