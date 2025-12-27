"""
Search Result Models.

Moved from hybrid_search.py to avoid circular dependencies.
"""

from __future__ import annotations

from typing import Any, Literal

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

    chunk_id: str | None
    score: float
    # Optional conversation identifier; None indicates "unassigned"
    conversation_id: str | None = None
    attachment_id: str | None = None
    is_attachment: bool = False
    highlights: list[str] = Field(default_factory=list)
    # Empty snippet is valid for metadata-only results
    snippet: str = ""
    content: str | None = None
    source: str | None = None
    filename: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    lexical_score: float | None = None
    vector_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    content_hash: str | None = None

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
    def from_fts_result(cls, res: Any) -> SearchResultItem:
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
    reranker: str | None = None
    results: list[SearchResultItem]
