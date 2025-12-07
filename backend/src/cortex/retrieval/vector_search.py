"""
Vector Search retrieval.

Implements §8.3 of the Canonical Blueprint.

Per pgvector official docs:
- HNSW index for vector type limited to 2000 dims
- We use halfvec cast for 3072-dim embeddings (supports up to 4000 dims)
- Query must cast to halfvec to use the HNSW index
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from cortex.config.loader import get_config
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class VectorResult(BaseModel):
    """Vector search result."""

    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_type: Optional[str] = None


def search_chunks_vector(
    session: Session, embedding: List[float], tenant_id: str, limit: int = 50
) -> List[VectorResult]:
    """
    Perform vector search on chunks using HNSW index.

    Blueprint §8.3:
    * Vector search (pgvector) over chunks.embedding

    Per pgvector docs:
    * Uses halfvec cast for 3072-dim vectors to leverage HNSW index
    * Cosine distance operator (<=>) for normalized embeddings
    """
    config = get_config()
    output_dim = config.embedding.output_dimensionality

    # Convert embedding to string format for raw SQL
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

    # Use raw SQL to properly cast to halfvec for HNSW index utilization
    # Per pgvector docs: query must use same type as index (halfvec)
    stmt = text(
        """
        SELECT 
            chunk_id,
            text,
            metadata,
            chunk_type,
            thread_id,
            message_id,
            attachment_id,
            embedding::halfvec(:dim) <=> :query_vec::halfvec(:dim) AS distance
        FROM chunks
        WHERE tenant_id = :tenant_id
        ORDER BY embedding::halfvec(:dim) <=> :query_vec::halfvec(:dim)
        LIMIT :limit
    """
    )

    results = session.execute(
        stmt,
        {
            "tenant_id": tenant_id,
            "query_vec": embedding_str,
            "dim": output_dim,
            "limit": limit,
        },
    ).fetchall()

    out = []
    for row in results:
        # Convert distance to similarity score (0..1)
        # Cosine distance is 1 - cosine similarity
        score = 1.0 - row.distance if row.distance is not None else 0.0

        metadata = row.metadata if isinstance(row.metadata, dict) else {}
        metadata["thread_id"] = str(row.thread_id) if row.thread_id else ""
        metadata["message_id"] = str(row.message_id) if row.message_id else ""
        metadata["attachment_id"] = str(row.attachment_id) if row.attachment_id else ""
        if row.chunk_type:
            metadata["chunk_type"] = row.chunk_type

        out.append(
            VectorResult(
                chunk_id=str(row.chunk_id),
                score=score,
                text=row.text or "",
                metadata=metadata,
                chunk_type=row.chunk_type,
            )
        )

    return out


# -----------------------------------------------------------------------------
# Canonical Blueprint Alias (§8.3)
# -----------------------------------------------------------------------------
# Blueprint uses search_vector_chunks naming convention

search_vector_chunks = search_chunks_vector
"""Canonical alias for search_chunks_vector per Blueprint §8.3."""
