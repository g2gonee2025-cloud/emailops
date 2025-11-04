"""
Operations package - Direct Python APIs for EmailOps functionality.

Replaces CLI subprocess calls with direct function calls for GUI integration.
Provides ~95% latency reduction (2-3s → 50-100ms) and progress callbacks.

This package extracts business logic from CLI, providing:
- search_documents(): Direct search API with progress callbacks
- index_documents(): Direct indexing API with progress callbacks
- summarize_conversation(): Direct summarization API
"""

from .indexing import IndexingError, IndexingProgress, index_documents
from .search import SearchError, SearchProgress, search_documents
from .summarize import SummarizationError, SummarizationProgress, summarize_conversation

__all__ = [
    "IndexingError",
    "IndexingProgress",
    "SearchError",
    "SearchProgress",
    "SummarizationError",
    "SummarizationProgress",
    "index_documents",
    "search_documents",
    "summarize_conversation",
]
