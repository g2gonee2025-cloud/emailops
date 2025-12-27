"""Domain-level models for tools and CLI inputs."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from cortex.retrieval.hybrid_search import KBSearchInput as RetrievalKBSearchInput
from pydantic import BaseModel, Field


class KBSearchInput(BaseModel):
    """Input model for knowledge base search tooling."""

    query: str
    limit: int = Field(default=10, ge=1, description="Number of results")
    fusion_strategy: Literal["rrf", "weighted_sum"] = "rrf"
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)

    def to_tool_input(self) -> RetrievalKBSearchInput:
        """Convert to the retrieval-layer input model."""

        return RetrievalKBSearchInput(
            tenant_id=self.tenant_id if self.tenant_id is not None else "default",
            user_id=self.user_id if self.user_id is not None else "cli-user",
            query=self.query,
            k=self.limit,
            fusion_method=self.fusion_strategy,
            filters=self.filters,
        )
