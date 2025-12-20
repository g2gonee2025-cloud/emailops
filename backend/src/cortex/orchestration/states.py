"""
Graph State Models.

Implements §3.6 and §10.1 of the Canonical Blueprint.
"""

from __future__ import annotations

import uuid
from typing import Any, List, Optional

from cortex.domain_models.facts_ledger import CriticReview, FactsLedger
from cortex.domain_models.rag import Answer, DraftCritique, EmailDraft, ThreadSummary
from cortex.retrieval.query_classifier import QueryClassification
from cortex.retrieval.results import SearchResults
from pydantic import BaseModel, Field


class GraphState(BaseModel):
    """Base class that exposes dict-like helpers for LangGraph nodes."""

    def get(self, key: str, default: Any | None = None) -> Any | None:
        return getattr(self, key, default)


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
    * answer: Optional[Answer]
    * error: Optional[str]
    """

    query: str
    tenant_id: str
    user_id: str
    thread_id: Optional[str] = None
    k: int = 10
    debug: bool = False
    classification: Optional[QueryClassification] = None
    retrieval_results: Optional[SearchResults] = None
    assembled_context: Optional[str] = None
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
    to: List[str] = Field(default_factory=list)
    cc: List[str] = Field(default_factory=list)
    subject: Optional[str] = None
    tone: Optional[str] = None
    reply_to_message_id: Optional[str] = None
    thread_id: Optional[str] = None
    explicit_query: Optional[str] = None
    draft_query: Optional[str] = None
    retrieval_results: Optional[SearchResults] = None
    assembled_context: Optional[str] = None
    draft: Optional[EmailDraft] = None
    critique: Optional[DraftCritique] = None
    iteration_count: int = 0
    error: Optional[str] = None
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class SummarizeThreadState(GraphState):
    """
    State for graph_summarize_thread.

    Blueprint §10.4:
    * thread_id: UUID
    * ...
    """

    tenant_id: str
    user_id: str
    thread_id: str
    max_length: int = 500
    thread_context: Optional[str] = None  # Raw text of thread
    facts_ledger: Optional[FactsLedger] = None
    critique: Optional[CriticReview] = None
    iteration_count: int = 0
    summary: Optional[ThreadSummary] = None
    error: Optional[str] = None
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
