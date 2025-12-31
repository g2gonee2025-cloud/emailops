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
from pydantic import ValidationError

RetrievalKBSearchInput = HybridSearchInput


async def tool_kb_search_hybrid(
    args: KBSearchInput | RetrievalKBSearchInput,
) -> Result[SearchResults, RetrievalError]:
    """Run hybrid search using the retrieval tool (async)."""

    if args is None:
        raise ValueError("tool_kb_search_hybrid: 'args' cannot be None")
    if isinstance(args, RetrievalKBSearchInput):
        tool_input = args
    else:
        payload = args.to_tool_input()
        try:
            tool_input = RetrievalKBSearchInput(**payload)
        except (ValidationError, TypeError) as exc:
            raise ValueError("tool_kb_search_hybrid: invalid input") from exc
    return await retrieval_tool_kb_search_hybrid(tool_input)
