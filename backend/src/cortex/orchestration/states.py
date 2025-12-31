"""
Graph State Models.

Implements §3.6, §10.3, and §10.4 of the Canonical Blueprint.
"""

from __future__ import annotations

import uuid
from typing import Any

from cortex.domain_models.facts_ledger import CriticReview, FactsLedger
from cortex.domain_models.rag import Answer, DraftCritique, EmailDraft, ThreadSummary
from cortex.orchestration.redacted import Redacted
from cortex.retrieval.query_classifier import QueryClassification
from cortex.retrieval.results import SearchResults
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    field_validator,
)

# Default configuration constants
DEFAULT_RETRIEVAL_K = 10  # Number of chunks to retrieve
DEFAULT_SUMMARY_MAX_LENGTH = 500  # Max summary length in words


class GraphState(BaseModel):
    """Base class that exposes dict-like helpers for LangGraph nodes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Internal metadata
    _graph_type: str = PrivateAttr(default="unknown")

    def get(self, key: str, default: Any | None = None) -> Any | None:
        return getattr(self, key, default)

    def __repr__(self) -> str:
        """Return a redacted representation of the state."""
        field_reprs = []
        for field_name in self.__class__.model_fields:
            field_value = getattr(self, field_name, None)
            safe_value = _redact_state_value(field_value)
            field_reprs.append(f"{field_name}={safe_value!r}")
        return f"{self.__class__.__name__}({', '.join(field_reprs)})"


class AnswerQuestionState(GraphState):
    """
    State for graph_answer_question.

    Blueprint §3.6:
    * query: str
    * tenant_id: str
    * user_id: str
    * classification: Optional[QueryClassification]
    * retrieval_results: Optional[SearchResults]
    * assembled_context: Optional[str]
    * graph_context: Optional[str]
    * answer: Optional[Answer]
    * error: Optional[str]
    """

    query: SecretStr
    tenant_id: str
    user_id: str
    thread_id: str | None = None
    k: int = Field(
        default=DEFAULT_RETRIEVAL_K,
        ge=1,
        description="Number of chunks to retrieve",
    )
    debug: bool = False
    classification: QueryClassification | None = None
    retrieval_results: SearchResults | None = None
    assembled_context: Redacted | None = None
    graph_context: Redacted | None = None
    answer: Answer | None = None
    error: str | None = None
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def retrieval_k(self) -> int:
        return self.k


class DraftEmailState(GraphState):
    """
    State for graph_draft_email.

    Blueprint §10.3:
    * thread_id: Optional[UUID]
    * explicit_query: Optional[str]
    * draft_query: Optional[str]
    * ...
    """

    tenant_id: str
    user_id: str
    to: list[SecretStr] = Field(default_factory=list)
    cc: list[SecretStr] = Field(default_factory=list)
    subject: str | None = None
    tone: str | None = None
    reply_to_message_id: str | None = None
    thread_id: str | None = None
    thread_context: Redacted | None = None  # Loaded from DB if thread_id is present
    explicit_query: SecretStr | None = None
    draft_query: SecretStr | None = None
    retrieval_results: SearchResults | None = None
    assembled_context: Redacted | None = None
    draft: EmailDraft | None = None
    critique: DraftCritique | None = None
    iteration_count: int = 0
    error: str | None = None
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            return str(uuid.UUID(str(value)))
        except (ValueError, TypeError) as exc:
            raise ValueError("thread_id must be a valid UUID") from exc


class SummarizeThreadState(GraphState):
    """
    State for graph_summarize_thread.

    Blueprint §10.4:
    * thread_id: str (UUID string)
    * ...
    """

    tenant_id: str
    user_id: str
    thread_id: str
    max_length: int = Field(
        default=DEFAULT_SUMMARY_MAX_LENGTH,
        ge=50,
        le=2000,
        description="Max summary length in words",
    )
    thread_context: Redacted | None = None  # Raw text of thread
    facts_ledger: FactsLedger | None = None
    critique: CriticReview | None = None
    iteration_count: int = 0
    summary: ThreadSummary | None = None
    error: str | None = None
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, value: str) -> str:
        try:
            return str(uuid.UUID(str(value)))
        except (ValueError, TypeError) as exc:
            raise ValueError("thread_id must be a valid UUID") from exc


def _redact_state_value(value: Any) -> Any:
    if isinstance(value, (Redacted, SecretStr)):
        return "REDACTED"
    if isinstance(value, BaseModel):
        return f"<{value.__class__.__name__}>"
    if isinstance(value, dict):
        return {key: _redact_state_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_redact_state_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_state_value(item) for item in value)
    return value
