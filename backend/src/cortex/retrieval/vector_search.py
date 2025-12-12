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

import numpy as np
from cortex.common.exceptions import RetrievalError
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

    # Keep these first-class so callers don’t have to dig in JSON metadata.
    thread_id: str = ""
    message_id: str = ""
    attachment_id: str = ""

    # Raw distance can be useful for debugging/metrics.
    distance: Optional[float] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_type: Optional[str] = None


def search_chunks_vector(
    session: Session,
    embedding: List[float],
    tenant_id: str,
    limit: int = 50,
    *,
    ef_search: Optional[int] = None,
    thread_ids: Optional[List[str]] = None,
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

    if len(embedding) != output_dim:
        raise RetrievalError(
            "Embedding dimension mismatch",
            query="vector_search",
            context={"expected_dim": output_dim, "got": len(embedding)},
        )

    try:
        emb_array = np.asarray(embedding, dtype=float)
    except Exception:
        raise RetrievalError(
            "Embedding must contain numeric values",
            query="vector_search",
        )

    if not np.all(np.isfinite(emb_array)):
        raise RetrievalError(
            "Embedding contains non-finite values",
            query="vector_search",
        )

    # Convert embedding to pgvector text format for raw SQL
    # (pgvector docs use the same bracketed representation for literals)
    embedding_str = "[" + ",".join(repr(float(v)) for v in emb_array.tolist()) + "]"

    # Use raw SQL to properly cast to halfvec for HNSW index utilization
    # Per pgvector docs: query must use same type as index (halfvec)
    # We use CAST(:query_vec AS halfvec(:dim)) to avoid SQLAlchemy parsing issues with ::
    # Optional thread filter (often useful for navigational lookups)
    thread_filter_sql = ""
    params: Dict[str, Any] = {
        "tenant_id": tenant_id,
        "query_vec": embedding_str,
        "dim": output_dim,
        "limit": limit,
    }

    if thread_ids:
        thread_filter_sql = "AND c.thread_id = ANY(CAST(:thread_ids AS UUID[]))"
        params["thread_ids"] = thread_ids

    # Optional HNSW tuning for recall/speed tradeoff.
    # pgvector docs recommend setting hnsw.ef_search per query using SET LOCAL.
    # We do it inline via set_config(..., is_local=true) so it only affects this statement.
    hnsw_settings_cte = ""
    if ef_search is not None:
        params["ef_search"] = str(int(ef_search))
        hnsw_settings_cte = (
            "WITH settings AS (SELECT set_config('hnsw.ef_search', :ef_search, true))"
        )

    # Inject dimensions directly into SQL as Postgres type modifiers cannot be parameterized
    stmt = text(
        f"""
        {hnsw_settings_cte}
        SELECT
            c.chunk_id,
            c.text,
            c.metadata,
            c.chunk_type,
            c.thread_id,
            c.message_id,
            c.attachment_id,
            c.embedding::halfvec({output_dim}) <=> CAST(:query_vec AS halfvec({output_dim})) AS distance
        FROM chunks c
        {', settings' if ef_search is not None else ''}
        WHERE c.tenant_id = :tenant_id
        {thread_filter_sql}
        ORDER BY distance
        LIMIT :limit
    """
    )

    results = session.execute(stmt, params).fetchall()

    out = []
    for row in results:
        # pgvector’s cosine distance operator (<=>) is 1 - cosine_similarity.
        # Cosine similarity can be [-1, 1], so distance is typically [0, 2].
        # We map similarity to [0, 1] for easier blending with lexical scores.
        distance = float(row.distance) if row.distance is not None else None
        cosine_sim = 1.0 - distance if distance is not None else -1.0
        score = (cosine_sim + 1.0) / 2.0
        # Guard against minor numeric drift
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0

        thread_id = str(row.thread_id) if row.thread_id else ""
        message_id = str(row.message_id) if row.message_id else ""
        attachment_id = str(row.attachment_id) if row.attachment_id else ""

        metadata = dict(row.metadata) if isinstance(row.metadata, dict) else {}
        metadata.setdefault("thread_id", thread_id)
        metadata.setdefault("message_id", message_id)
        metadata.setdefault("attachment_id", attachment_id)
        if row.chunk_type:
            metadata.setdefault("chunk_type", row.chunk_type)

        out.append(
            VectorResult(
                chunk_id=str(row.chunk_id),
                score=score,
                text=row.text or "",
                thread_id=thread_id,
                message_id=message_id,
                attachment_id=attachment_id,
                distance=distance,
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
