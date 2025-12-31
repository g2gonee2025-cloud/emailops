"""Domain-level models for tools and CLI inputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MAX_KB_LIMIT = 200
FUSION_STRATEGY_MAP: dict[str, Literal["rrf", "weighted_sum"]] = {
    "rrf": "rrf",
    "weighted_sum": "weighted_sum",
}


class KBSearchInput(BaseModel):
    """Input model for knowledge base search tooling."""

    query: str
    limit: int = Field(
        default=10, ge=1, le=MAX_KB_LIMIT, description="Number of results"
    )
    fusion_strategy: Literal["rrf", "weighted_sum"] = "rrf"
    tenant_id: str | None = None
    user_id: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("query must be a string")
        query = value.strip()
        if not query:
            raise ValueError("query cannot be empty")
        return query

    @field_validator("tenant_id", "user_id", mode="before")
    @classmethod
    def _normalize_optional_ids(cls, value: Any) -> Any:
        if isinstance(value, str):
            trimmed = value.strip()
            return trimmed or None
        return value

    def to_tool_input(self) -> dict[str, Any]:
        """Convert to the retrieval-layer payload."""
        tenant_id = self.tenant_id or ""
        user_id = self.user_id or ""
        if not tenant_id:
            raise ValueError("tenant_id is required for KB search")
        if not user_id:
            raise ValueError("user_id is required for KB search")

        fusion_method = FUSION_STRATEGY_MAP.get(self.fusion_strategy)
        if fusion_method is None:
            raise ValueError(f"Unsupported fusion_strategy: {self.fusion_strategy}")

        return {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "query": self.query,
            "k": min(int(self.limit), MAX_KB_LIMIT),
            "fusion_method": fusion_method,
            "filters": dict(self.filters),
        }
