"""
Search Result Models.

Moved from hybrid_search.py to avoid circular dependencies.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SearchResultItem(BaseModel):
    """
    Search result item.

    Blueprint §8.4 (adapted for Conversation schema):
    * chunk_id: Optional[UUID]
    * score: float
    * conversation_id: str (was thread_id)
    * attachment_id: Optional[UUID]
    * highlights: List[str]
    * snippet: str
    * content: Optional[str]
    * source: Optional[str]
    * filename: Optional[str]
    * metadata: Dict[str, Any]
    """

    chunk_id: Optional[str]
    score: float
    conversation_id: str = ""  # Primary key for Conversation schema
    attachment_id: Optional[str] = None
    is_attachment: bool = False
    highlights: List[str] = Field(default_factory=list)
    snippet: str = ""
    content: Optional[str] = None
    source: Optional[str] = None
    filename: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    lexical_score: Optional[float] = None
    vector_score: Optional[float] = None
    fusion_score: Optional[float] = None
    rerank_score: Optional[float] = None
    content_hash: Optional[str] = None

    # Backward compatibility aliases
    @property
    def thread_id(self) -> str:
        """Alias for conversation_id (backward compat)."""
        return self.conversation_id

    @property
    def message_id(self) -> str:
        """Backward compat - empty since we use conversation model."""
        return ""


class SearchResults(BaseModel):
    """
    Search results container.

    Blueprint §8.4:
    * type: Literal["search_results"]
    * query: str
    * reranker: Optional[str]
    * results: List[SearchResultItem]
    """

    type: Literal["search_results"] = "search_results"
    query: str
    reranker: Optional[str] = None
    results: List[SearchResultItem]
