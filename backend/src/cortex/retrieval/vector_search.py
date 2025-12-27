"""
Vector Search retrieval.

Implements §8.3 of the Canonical Blueprint.
Adapted for Conversation-based schema (conversation_id instead of thread_id/message_id).

Per pgvector official docs:
- HNSW index for vector type limited to 2000 dims
- We use halfvec cast for 3840-dim embeddings (supports up to 4000 dims)
- Query must cast to halfvec to use the HNSW index
"""

from __future__ import annotations

import logging

from cortex.config.loader import get_config
from cortex.retrieval.vector_store import PgvectorStore, QdrantVectorStore, VectorResult
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def search_chunks_vector(
    session: Session,
    embedding: list[float],
    tenant_id: str,
    limit: int = 50,
    *,
    ef_search: int | None = None,
    conversation_ids: list[str] | None = None,
    is_attachment: bool | None = None,
    thread_ids: list[str] | None = None,  # Backward compat alias
    file_types: list[str] | None = None,  # P1 Fix: Add file_types filter
) -> list[VectorResult]:
    """
    Perform vector search on chunks using a pluggable vector store.

    Blueprint §8.3 (adapted for Conversation schema):
    * Vector search (pgvector) over chunks.embedding
    * Uses cosine distance operator (<=>) for normalized embeddings
    """
    config = get_config()
    output_dim = config.embedding.output_dimensionality

    if limit <= 0:
        limit = 50  # Default if invalid

    # Handling backward compatibility: merge thread_ids into conversation_ids
    # Prefer conversation_ids as the canonical source
    if thread_ids:
        if conversation_ids is None:
            conversation_ids = []
        conversation_ids.extend(thread_ids)
        # Deduplicate if mixed
        conversation_ids = list(set(conversation_ids))

    if getattr(config, "qdrant", None) and config.qdrant.enabled:
        store = QdrantVectorStore(config.qdrant, output_dim)
        # Removed logging every query to reduce noise
    else:
        store = PgvectorStore(session, output_dim)

    return store.search(
        embedding,
        tenant_id,
        limit=limit,
        ef_search=ef_search,
        conversation_ids=conversation_ids,
        is_attachment=is_attachment,
        file_types=file_types,
    )


# -----------------------------------------------------------------------------
# Canonical Blueprint Alias (§8.3)
# -----------------------------------------------------------------------------
# Blueprint uses search_vector_chunks naming convention

search_vector_chunks = search_chunks_vector
"""Canonical alias for search_chunks_vector per Blueprint §8.3."""
