"""
Search Result Models.

Moved from hybrid_search.py to avoid circular dependencies.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SearchResultItem(BaseModel):
    """
    Search result item.

    Blueprint §8.4 (adapted for Conversation schema):
    * chunk_id: Optional[UUID]
    * score: float
    * conversation_id: Optional[str] (was thread_id)
    * attachment_id: Optional[UUID]
    * highlights: list[str]
    * snippet: str
    * content: Optional[str]
    * source: Optional[str]
    * filename: Optional[str]
    * metadata: dict[str, Any]
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

    def __repr__(self) -> str:
        """Build a string representation with redacted content for secure logging."""
        fields = {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "conversation_id": self.conversation_id,
            "attachment_id": self.attachment_id,
            "is_attachment": self.is_attachment,
            "highlights_count": len(self.highlights),
            "snippet": "[REDACTED]" if self.snippet else "",
            "content": "[REDACTED]" if self.content else None,
            "source": self.source,
            "filename": self.filename,
            "metadata_keys": len(self.metadata) if self.metadata else 0,
            "lexical_score": self.lexical_score,
            "vector_score": self.vector_score,
            "fusion_score": self.fusion_score,
            "rerank_score": self.rerank_score,
            "content_hash": self.content_hash,
        }
        field_strings = [f"{key}={value!r}" for key, value in fields.items()]
        return f"SearchResultItem({', '.join(field_strings)})"

    # Backward compatibility aliases
    @property
    def thread_id(self) -> str | None:
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
            conversation_id=getattr(res, "conversation_id", None),
            snippet=getattr(res, "snippet", ""),
            content=getattr(res, "text", getattr(res, "content", None)),
            metadata=getattr(res, "metadata", {}),
            lexical_score=getattr(res, "lexical_score", None),
        )


class SearchResults(BaseModel):
    """
    Search results container.

    Blueprint §8.4:
    * type: Literal["search_results"]
    * query: str
    * reranker: Optional[str]
    * results: list[SearchResultItem]
    """

    model_config = ConfigDict(populate_by_name=True)

    result_type: Literal["search_results"] = Field(
        default="search_results", alias="type"
    )
    query: str
    reranker: str | None = None
    results: list[SearchResultItem]

    @property
    def type(self) -> str:
        return self.result_type
