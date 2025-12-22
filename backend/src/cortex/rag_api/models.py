"""
API Request/Response Models.

Implements §9 of the Canonical Blueprint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from cortex.domain_models.rag import Answer, EmailDraft, ThreadSummary
from pydantic import BaseModel, ConfigDict, Field


# -----------------------------------------------------------------------------
# Search API Models (§9.2)
# -----------------------------------------------------------------------------
class SearchRequest(BaseModel):
    """Search request payload."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., description="Search query text")
    k: int = Field(default=10, ge=1, le=500, description="Number of results")
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant ID (auto-filled from context)"
    )
    user_id: Optional[str] = Field(
        default=None, description="User ID (auto-filled from context)"
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict, description="Optional filters"
    )


class SearchResponse(BaseModel):
    """Search response payload."""

    correlation_id: Optional[str] = None
    results: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int = 0
    query_time_ms: float = 0.0


# -----------------------------------------------------------------------------
# Answer API Models (§9.3)
# -----------------------------------------------------------------------------
class AnswerRequest(BaseModel):
    """Question-answering request payload."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., description="Question to answer")
    thread_id: Optional[str] = Field(
        default=None, description="Optional thread context"
    )
    k: int = Field(default=10, description="Number of context chunks")
    debug: bool = Field(default=False, description="Enable debug info")
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None


class AnswerResponse(BaseModel):
    """Question-answering response payload."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: Optional[str] = None
    answer: Answer
    confidence: float = 0.0
    debug_info: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# Draft Email API Models (§9.4)
# -----------------------------------------------------------------------------
class DraftEmailRequest(BaseModel):
    """Email drafting request payload."""

    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(..., description="Drafting instruction")
    thread_id: Optional[str] = Field(default=None, description="Thread context")
    reply_to_message_id: Optional[str] = Field(
        default=None, description="Message to reply to"
    )
    tone: str = Field(default="professional", description="Email tone")
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None


class DraftEmailResponse(BaseModel):
    """Email drafting response payload."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: Optional[str] = None
    draft: EmailDraft
    confidence: float = 0.0
    iterations: int = 0


# -----------------------------------------------------------------------------
# Summarize Thread API Models (§9.5)
# -----------------------------------------------------------------------------
class SummarizeThreadRequest(BaseModel):
    """Thread summarization request payload."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(..., description="Thread to summarize")
    max_length: int = Field(default=500, description="Max summary length in words")
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None


class SummarizeThreadResponse(BaseModel):
    """Thread summarization response payload."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: Optional[str] = None
    summary: ThreadSummary


# -----------------------------------------------------------------------------
# Chat API Models (§9.6)
# -----------------------------------------------------------------------------
class ChatMessage(BaseModel):
    """Chat message payload."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Chat request payload."""

    messages: List[ChatMessage]
    thread_id: Optional[str] = Field(
        default=None, description="Optional thread context"
    )
    k: int = Field(default=10, description="Number of context chunks")
    max_length: int = Field(default=500, description="Max summary length in words")
    max_history: Optional[int] = Field(
        default=None, description="Max chat history entries"
    )
    debug: bool = Field(default=False, description="Enable debug info")
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response payload."""

    correlation_id: Optional[str] = None
    action: Literal["answer", "search", "summarize"]
    reply: str
    answer: Optional[Answer] = None
    summary: Optional[ThreadSummary] = None
    search_results: Optional[List[Dict[str, Any]]] = None
    debug_info: Optional[Dict[str, Any]] = None
