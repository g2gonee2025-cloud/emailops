"""Tool wrapper for hybrid search."""

from __future__ import annotations

from cortex.domain.models import KBSearchInput
from cortex.retrieval.hybrid_search import KBSearchInput as HybridSearchInput
from cortex.retrieval.hybrid_search import (
    tool_kb_search_hybrid as retrieval_tool_kb_search_hybrid,
)
from cortex.retrieval.results import SearchResults

RetrievalKBSearchInput = HybridSearchInput


def tool_kb_search_hybrid(
    args: KBSearchInput | RetrievalKBSearchInput,
) -> SearchResults:
    """Run hybrid search using the retrieval tool."""

    if args is None:
        raise ValueError("tool_kb_search_hybrid: 'args' cannot be None")
    tool_input = (
        args if isinstance(args, RetrievalKBSearchInput) else args.to_tool_input()
    )
    return retrieval_tool_kb_search_hybrid(tool_input)
