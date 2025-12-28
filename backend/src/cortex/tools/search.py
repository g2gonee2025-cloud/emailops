"""Tool wrapper for hybrid search."""

from __future__ import annotations

from cortex.common.exceptions import RetrievalError
from cortex.common.types import Result
from cortex.domain.models import KBSearchInput
from cortex.retrieval.hybrid_search import KBSearchInput as HybridSearchInput
from cortex.retrieval.hybrid_search import (
    tool_kb_search_hybrid as retrieval_tool_kb_search_hybrid,
)
from cortex.retrieval.results import SearchResults

RetrievalKBSearchInput = HybridSearchInput


async def tool_kb_search_hybrid(
    args: KBSearchInput | RetrievalKBSearchInput,
) -> Result[SearchResults, RetrievalError]:
    """Run hybrid search using the retrieval tool (async)."""

    if args is None:
        raise ValueError("tool_kb_search_hybrid: 'args' cannot be None")
    tool_input = (
        args if isinstance(args, RetrievalKBSearchInput) else args.to_tool_input()
    )
    return await retrieval_tool_kb_search_hybrid(tool_input)
