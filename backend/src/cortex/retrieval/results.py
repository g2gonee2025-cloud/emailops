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
    # Optional conversation identifier; None indicates "unassigned"
    conversation_id: Optional[str] = None
    attachment_id: Optional[str] = None
    is_attachment: bool = False
    highlights: list[str] = Field(default_factory=list)
    # Empty snippet is valid for metadata-only results
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
        # TODO: This should be Optional[str] to match conversation_id.
        # This requires updating consumers to handle `None`.
        """Alias for conversation_id (backward compat)."""
        return self.conversation_id

    @property
    def message_id(self) -> str:
        """Backward compat - empty since we use conversation model."""
        return ""

    @classmethod
    def from_fts_result(cls, res: Any) -> "SearchResultItem":
        """Factory from FTS result (ChunkFTSResult or similar)."""
        # Mapping depends on the exact shape of FTS result
        return cls(
            chunk_id=getattr(res, "chunk_id", None),
            score=getattr(res, "score", 0.0),
            conversation_id=getattr(res, "conversation_id", ""),
            snippet=getattr(res, "snippet", ""),
            content=getattr(res, "text", getattr(res, "content", None)),
            metadata=getattr(res, "metadata", {}),
            lexical_score=getattr(res, "lexical_score", 0.0),
        )


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
