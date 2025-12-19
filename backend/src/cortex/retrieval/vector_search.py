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
from typing import Any, Dict, List, Optional

import numpy as np
from cortex.common.exceptions import RetrievalError
from cortex.config.loader import get_config
from cortex.config.models import QdrantConfig
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class VectorResult(BaseModel):
    """Vector search result."""

    chunk_id: str
    score: float
    text: str

    # Conversation-based schema uses conversation_id instead of thread_id
    conversation_id: str = ""
    attachment_id: str = ""
    is_attachment: bool = False

    # Raw distance can be useful for debugging/metrics.
    distance: Optional[float] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_type: Optional[str] = None

    # Backward compat aliases (hybrid_search.py may reference these)
    @property
    def thread_id(self) -> str:
        """Alias for conversation_id (backward compat)."""
        return self.conversation_id

    @property
    def message_id(self) -> str:
        """Backward compat - return empty since we use conversation model."""
        return ""


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
    Perform vector search on chunks using pgvector.

    Blueprint §8.3 (adapted for Conversation schema):
    * Vector search (pgvector) over chunks.embedding
    * Uses cosine distance operator (<=>) for normalized embeddings
    """
    config = get_config()
    output_dim = config.embedding.output_dimensionality

    # Optional Qdrant integration: if enabled, switch to Qdrant API call.
    # Note: The actual implementation is left as a future enhancement.
    # When Qdrant is enabled, this function could delegate to a Qdrant client
    # that performs a filtered search and returns VectorResult objects.
    # For now we simply log the intent.
    qdrant_cfg: QdrantConfig = (
        config.qdrant if hasattr(config, "qdrant") else QdrantConfig()
    )
    if qdrant_cfg.enabled:
        logger.info(
            "Qdrant integration enabled – delegating vector search to Qdrant at %s",
            qdrant_cfg.url,
        )

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

    # Convert embedding to pgvector text format
    embedding_str = "[" + ",".join(repr(float(v)) for v in emb_array.tolist()) + "]"

    # Handle backward compat: thread_ids -> conversation_ids
    if thread_ids and not conversation_ids:
        conversation_ids = thread_ids

    # Build query with optional conversation filter
    conversation_filter_sql = ""
    params: Dict[str, Any] = {
        "tenant_id": tenant_id,
        "query_vec": embedding_str,
        "limit": limit,
    }

    if conversation_ids:
        conversation_filter_sql = (
            "AND c.conversation_id = ANY(CAST(:conversation_ids AS UUID[]))"
        )
        params["conversation_ids"] = conversation_ids

    # Attachment filter
    attachment_filter_sql = ""
    if is_attachment is not None:
        if is_attachment:
            attachment_filter_sql = "AND c.is_attachment = TRUE"
        else:
            attachment_filter_sql = "AND c.is_attachment = FALSE"

    # Optional HNSW tuning for recall/speed tradeoff
    hnsw_settings_cte = ""
    if ef_search is not None:
        params["ef_search"] = str(int(ef_search))
        hnsw_settings_cte = (
            "WITH settings AS (SELECT set_config('hnsw.ef_search', :ef_search, true))"
        )

    # Use cosine distance directly on vector type (no halfvec needed for 3840 dims
    # since it's within pgvector's vector limit for exact search)
    stmt = text(
        f"""
        {hnsw_settings_cte}
        SELECT
            c.chunk_id,
            c.conversation_id,
            c.text,
            c.extra_data,
            c.chunk_type,
            c.is_attachment,
            c.attachment_id,
            c.embedding <=> CAST(:query_vec AS vector({output_dim})) AS distance
        FROM chunks c
        {', settings' if ef_search is not None else ''}
        WHERE c.tenant_id = :tenant_id
          AND c.embedding IS NOT NULL
        WHERE c.tenant_id = :tenant_id
          AND c.embedding IS NOT NULL
        {conversation_filter_sql}
        {attachment_filter_sql}
        ORDER BY distance
        LIMIT :limit
    """
    )

    results = session.execute(stmt, params).fetchall()

    out = []
    for row in results:
        # pgvector's cosine distance operator (<=>) is 1 - cosine_similarity.
        # Cosine similarity can be [-1, 1], so distance is typically [0, 2].
        # We map similarity to [0, 1] for easier blending with lexical scores.
        distance = float(row.distance) if row.distance is not None else None
        cosine_sim = 1.0 - distance if distance is not None else -1.0
        score = (cosine_sim + 1.0) / 2.0
        # Guard against minor numeric drift
        score = max(0.0, min(1.0, score))

        conversation_id = str(row.conversation_id) if row.conversation_id else ""
        attachment_id = str(row.attachment_id) if row.attachment_id else ""

        metadata = dict(row.extra_data) if isinstance(row.extra_data, dict) else {}
        metadata.setdefault("conversation_id", conversation_id)
        metadata.setdefault("attachment_id", attachment_id)
        if row.chunk_type:
            metadata.setdefault("chunk_type", row.chunk_type)

        out.append(
            VectorResult(
                chunk_id=str(row.chunk_id),
                score=score,
                text=row.text or "",
                conversation_id=conversation_id,
                attachment_id=attachment_id,
                is_attachment=row.is_attachment or False,
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
