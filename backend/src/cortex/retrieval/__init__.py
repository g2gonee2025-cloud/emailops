"""
Retrieval module for Cortex.

Implements §8 of the Canonical Blueprint.
"""
from cortex.retrieval.query_classifier import (
    QueryClassification,
    tool_classify_query,
    classify_query_fast,
    is_navigational,
    is_drafting,
    requires_grounding_check,
)
from cortex.retrieval.hybrid_search import (
    KBSearchInput,
    SearchResultItem,
    SearchResults,
    tool_kb_search_hybrid,
    apply_recency_boost,
    deduplicate_by_hash,
    downweight_quoted_history,
    fuse_rrf,
)

__all__ = [
    # Query Classification (§8.2)
    "QueryClassification",
    "tool_classify_query",
    "classify_query_fast",
    "is_navigational",
    "is_drafting",
    "requires_grounding_check",
    # Hybrid Search (§8.3)
    "KBSearchInput",
    "SearchResultItem",
    "SearchResults",
    "tool_kb_search_hybrid",
    "apply_recency_boost",
    "deduplicate_by_hash",
    "downweight_quoted_history",
    "fuse_rrf",
]