"""
Graph State Models.

Implements §3.6 and §10.1 of the Canonical Blueprint.
"""

from __future__ import annotations

import uuid
from typing import Any, List, Optional

from cortex.domain_models.facts_ledger import CriticReview, FactsLedger
from cortex.domain_models.rag import Answer, DraftCritique, EmailDraft, ThreadSummary
from cortex.orchestration.redacted import Redacted
from cortex.retrieval.query_classifier import QueryClassification
from cortex.retrieval.results import SearchResults
from pydantic import BaseModel, ConfigDict, Field, SecretStr

# Default configuration constants
DEFAULT_RETRIEVAL_K = 10  # Number of chunks to retrieve
DEFAULT_SUMMARY_MAX_LENGTH = 500  # Max summary length in words


class GraphState(BaseModel):
    """Base class that exposes dict-like helpers for LangGraph nodes."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Internal metadata
    _graph_type: str = "unknown"

    def get(self, key: str, default: Any | None = None) -> Any | None:
        return getattr(self, key, default)

    def __repr__(self) -> str:
        """Return a redacted representation of the state."""
        field_reprs = []
        for field_name, field_value in self:
            if isinstance(field_value, (Redacted, SecretStr)):
                field_reprs.append(f"{field_name}='REDACTED'")
            else:
                field_reprs.append(f"{field_name}={field_value!r}")
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
    thread_id: Optional[str] = None
    k: int = DEFAULT_RETRIEVAL_K
    debug: bool = False
    classification: Optional[QueryClassification] = None
    retrieval_results: Optional[SearchResults] = None
    assembled_context: Optional[Redacted] = None
    graph_context: Optional[Redacted] = None
    answer: Optional[Answer] = None
    error: Optional[str] = None
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


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
    to: List[SecretStr] = Field(default_factory=list)
    cc: List[SecretStr] = Field(default_factory=list)
    subject: Optional[str] = None
    tone: Optional[str] = None
    reply_to_message_id: Optional[str] = None
    thread_id: Optional[str] = None
    thread_context: Optional[Redacted] = None  # Loaded from DB if thread_id is present
    explicit_query: Optional[SecretStr] = None
    draft_query: Optional[SecretStr] = None
    retrieval_results: Optional[SearchResults] = None
    assembled_context: Optional[Redacted] = None
    draft: Optional[EmailDraft] = None
    critique: Optional[DraftCritique] = None
    iteration_count: int = 0
    error: Optional[str] = None
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


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
    max_length: int = DEFAULT_SUMMARY_MAX_LENGTH
    thread_context: Optional[Redacted] = None  # Raw text of thread
    facts_ledger: Optional[FactsLedger] = None
    critique: Optional[CriticReview] = None
    iteration_count: int = 0
    summary: Optional[ThreadSummary] = None
    error: Optional[str] = None
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
