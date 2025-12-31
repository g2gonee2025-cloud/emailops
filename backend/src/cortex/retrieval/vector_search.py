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
from typing import Any

from cortex.config.loader import get_config
from cortex.retrieval.vector_store import PgvectorStore, QdrantVectorStore, VectorResult
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
_QDRANT_STORE_CACHE: dict[tuple[str, str | None, str, int], QdrantVectorStore] = {}


def search_chunks_vector(
    session: Session | None,
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
    embedding_config = getattr(config, "embedding", None)
    output_dim = getattr(embedding_config, "output_dimensionality", None)

    if limit <= 0:
        logger.debug("Vector search skipped for non-positive limit: %s", limit)
        return []

    if not output_dim:
        logger.error("Embedding output dimensionality missing; skipping vector search")
        return []
    if not embedding:
        logger.warning("Empty embedding supplied; skipping vector search")
        return []
    if len(embedding) != output_dim:
        logger.error(
            "Embedding dimension mismatch (expected %d, got %d); skipping search",
            output_dim,
            len(embedding),
        )
        return []

    # Handling backward compatibility: merge thread_ids into conversation_ids
    # Prefer conversation_ids as the canonical source
    normalized_conversation_ids = _normalize_conversation_ids(
        conversation_ids, thread_ids
    )
    if normalized_conversation_ids is not None and not normalized_conversation_ids:
        return []

    qdrant_config = getattr(config, "qdrant", None)
    if qdrant_config and getattr(qdrant_config, "enabled", False):
        store = _get_qdrant_store(qdrant_config, output_dim)
        # Removed logging every query to reduce noise
    else:
        if session is None:
            raise ValueError("Database session required for pgvector search")
        store = PgvectorStore(session, output_dim)

    return store.search(
        embedding,
        tenant_id,
        limit=limit,
        ef_search=ef_search,
        conversation_ids=normalized_conversation_ids,
        is_attachment=is_attachment,
        file_types=file_types,
    )


def _normalize_conversation_ids(
    conversation_ids: list[str] | None,
    thread_ids: list[str] | None,
) -> list[str] | None:
    if conversation_ids is None and thread_ids is None:
        return None
    merged: list[str] = []
    if conversation_ids:
        merged.extend(conversation_ids)
    if thread_ids:
        merged.extend(thread_ids)
    filtered = [str(conv_id) for conv_id in merged if conv_id]
    return list(dict.fromkeys(filtered))


def _get_qdrant_store(
    qdrant_config: Any,
    output_dim: int,
) -> QdrantVectorStore:
    key = (
        getattr(qdrant_config, "url", ""),
        getattr(qdrant_config, "api_key", None),
        getattr(qdrant_config, "collection_name", ""),
        output_dim,
    )
    store = _QDRANT_STORE_CACHE.get(key)
    if store is None:
        store = QdrantVectorStore(qdrant_config, output_dim)
        _QDRANT_STORE_CACHE[key] = store
    return store


# -----------------------------------------------------------------------------
# Canonical Blueprint Alias (§8.3)
# -----------------------------------------------------------------------------
# Blueprint uses search_vector_chunks naming convention

search_vector_chunks = search_chunks_vector
"""Canonical alias for search_chunks_vector per Blueprint §8.3."""
