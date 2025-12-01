"""
Graph State Models.

Implements §3.6 and §10.1 of the Canonical Blueprint.
"""
from __future__ import annotations

from typing import Optional, Dict, Any

from pydantic import BaseModel

from cortex.models.rag import Answer, EmailDraft, ThreadSummary, DraftCritique
from cortex.models.facts_ledger import FactsLedger, CriticReview
from cortex.retrieval.query_classifier import QueryClassification
from cortex.retrieval.hybrid_search import SearchResults


class AnswerQuestionState(BaseModel):
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
    classification: Optional[QueryClassification] = None
    retrieval_results: Optional[SearchResults] = None
    assembled_context: Optional[str] = None
    answer: Optional[Answer] = None
    error: Optional[str] = None


class DraftEmailState(BaseModel):
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
    thread_id: Optional[str] = None
    explicit_query: Optional[str] = None
    draft_query: Optional[str] = None
    retrieval_results: Optional[SearchResults] = None
    assembled_context: Optional[str] = None
    draft: Optional[EmailDraft] = None
    critique: Optional[DraftCritique] = None
    iteration_count: int = 0
    error: Optional[str] = None


class SummarizeThreadState(BaseModel):
    """
    State for graph_summarize_thread.
    
    Blueprint §10.4:
    * thread_id: UUID
    * ...
    """
    tenant_id: str
    user_id: str
    thread_id: str
    thread_context: Optional[str] = None # Raw text of thread
    facts_ledger: Optional[FactsLedger] = None
    critique: Optional[CriticReview] = None
    iteration_count: int = 0
    summary: Optional[ThreadSummary] = None
    error: Optional[str] = None