"""
API Request/Response Models.

Implements §9 of the Canonical Blueprint.
"""

from __future__ import annotations

from typing import Any, Literal

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
    tenant_id: str | None = Field(
        default=None, description="Tenant ID (auto-filled from context)"
    )
    user_id: str | None = Field(
        default=None, description="User ID (auto-filled from context)"
    )
    filters: dict[str, Any] = Field(
        default_factory=dict, description="Optional filters"
    )
    fusion_method: str | None = Field(
        default="rrf", description="Fusion method (rrf or weighted_sum)"
    )


class SearchResponse(BaseModel):
    """Search response payload."""

    correlation_id: str | None = None
    results: list[dict[str, Any]] = Field(default_factory=list)
    total_count: int = Field(0, ge=0)
    query_time_ms: float = Field(0.0, ge=0.0)


# -----------------------------------------------------------------------------
# Answer API Models (§9.3)
# -----------------------------------------------------------------------------
class AnswerRequest(BaseModel):
    """Question-answering request payload."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ..., description="Question to answer", min_length=1, max_length=2000
    )
    thread_id: str | None = Field(default=None, description="Optional thread context")
    k: int = Field(default=10, ge=1, le=20, description="Number of context chunks")
    debug: bool = Field(default=False, description="Enable debug info")
    tenant_id: str | None = None
    user_id: str | None = None


class AnswerResponse(BaseModel):
    """Question-answering response payload."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: str | None = None
    answer: Answer
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score (0-1)"
    )
    debug_info: dict[str, Any] | None = None


# -----------------------------------------------------------------------------
# Draft Email API Models (§9.4)
# -----------------------------------------------------------------------------
class DraftEmailRequest(BaseModel):
    """Email drafting request payload."""

    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(..., description="Drafting instruction")
    thread_id: str | None = Field(default=None, description="Thread context")
    reply_to_message_id: str | None = Field(
        default=None, description="Message to reply to"
    )
    tone: str = Field(default="professional", description="Email tone")
    tenant_id: str | None = None
    user_id: str | None = None


class DraftEmailResponse(BaseModel):
    """Email drafting response payload."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: str | None = None
    draft: EmailDraft
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    iterations: int = Field(0, ge=0)


# -----------------------------------------------------------------------------
# Summarize Thread API Models (§9.5)
# -----------------------------------------------------------------------------
class SummarizeThreadRequest(BaseModel):
    """Thread summarization request payload."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(..., description="Thread to summarize")
    max_length: int = Field(
        default=500, ge=50, le=2000, description="Max summary length in words"
    )
    tenant_id: str | None = None
    user_id: str | None = None


class SummarizeThreadResponse(BaseModel):
    """Thread summarization response payload."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: str | None = None
    summary: ThreadSummary


# -----------------------------------------------------------------------------
# Chat API Models (§9.6)
# -----------------------------------------------------------------------------
class ChatMessage(BaseModel):
    """Chat message payload."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(..., max_length=10000)


class ChatRequest(BaseModel):
    """Chat request payload."""

    messages: list[ChatMessage]
    thread_id: str | None = Field(default=None, description="Optional thread context")
    k: int = Field(default=10, ge=1, le=20, description="Number of context chunks")
    max_length: int = Field(
        default=500, ge=50, le=2000, description="Max summary length in words"
    )
    max_history: int | None = Field(
        default=None, ge=0, le=50, description="Max chat history entries"
    )


class ChatResponse(BaseModel):
    """Chat response payload."""

    correlation_id: str | None = None
    action: Literal["answer", "search", "summarize"]
    reply: str
    answer: Answer | None = None
    summary: ThreadSummary | None = None
    search_results: list[dict[str, Any]] | None = None
    debug_info: dict[str, Any] | None = None
