"""Tool wrapper for hybrid search."""

from __future__ import annotations

from typing import Union

from cortex.domain.models import KBSearchInput
from cortex.retrieval.hybrid_search import (
    KBSearchInput as RetrievalKBSearchInput,
    tool_kb_search_hybrid as retrieval_tool_kb_search_hybrid,
)
from cortex.retrieval.results import SearchResults


def tool_kb_search_hybrid(
    args: Union[KBSearchInput, RetrievalKBSearchInput],
) -> SearchResults:
    """Run hybrid search using the retrieval tool."""

    tool_input = (
        args if isinstance(args, RetrievalKBSearchInput) else args.to_tool_input()
    )
    return retrieval_tool_kb_search_hybrid(tool_input)
