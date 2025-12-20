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
from typing import List, Optional

from cortex.config.loader import get_config
from cortex.retrieval.vector_store import PgvectorStore, QdrantVectorStore, VectorResult
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def search_chunks_vector(
    session: Session,
    embedding: List[float],
    tenant_id: str,
    limit: int = 50,
    *,
    ef_search: Optional[int] = None,
    conversation_ids: Optional[List[str]] = None,
    is_attachment: Optional[bool] = None,
    thread_ids: Optional[List[str]] = None,  # Backward compat alias
) -> List[VectorResult]:
    """
    Perform vector search on chunks using a pluggable vector store.

    Blueprint §8.3 (adapted for Conversation schema):
    * Vector search (pgvector) over chunks.embedding
    * Uses cosine distance operator (<=>) for normalized embeddings
    """
    config = get_config()
    output_dim = config.embedding.output_dimensionality

    if getattr(config, "qdrant", None) and config.qdrant.enabled:
        store = QdrantVectorStore(config.qdrant, output_dim)
        logger.info(
            "Qdrant integration enabled – delegating vector search to Qdrant at %s",
            config.qdrant.url,
        )
    else:
        store = PgvectorStore(session, output_dim)

    return store.search(
        embedding,
        tenant_id,
        limit=limit,
        ef_search=ef_search,
        conversation_ids=conversation_ids,
        is_attachment=is_attachment,
        thread_ids=thread_ids,
    )


# -----------------------------------------------------------------------------
# Canonical Blueprint Alias (§8.3)
# -----------------------------------------------------------------------------
# Blueprint uses search_vector_chunks naming convention

search_vector_chunks = search_chunks_vector
"""Canonical alias for search_chunks_vector per Blueprint §8.3."""
