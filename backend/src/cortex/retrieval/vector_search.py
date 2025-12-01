"""
Vector Search retrieval.

Implements §8.3 of the Canonical Blueprint.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from cortex.db.models import Chunk

logger = logging.getLogger(__name__)


class VectorResult(BaseModel):
    """Vector search result."""
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


def search_chunks_vector(
    session: Session,
    embedding: List[float],
    tenant_id: str,
    limit: int = 50
) -> List[VectorResult]:
    """
    Perform vector search on chunks.
    
    Blueprint §8.3:
    * Vector search (pgvector) over chunks.embedding
    """
    # Blueprint §8.3: Vector search (pgvector) over chunks.embedding
    # Calculate cosine similarity score (1 - cosine_distance)
    
    distance_expr = Chunk.embedding.cosine_distance(embedding)
    
    stmt = select(Chunk, distance_expr.label("distance")).filter(
        Chunk.tenant_id == tenant_id
    ).order_by(
        distance_expr
    ).limit(limit)
    
    results = session.execute(stmt).all()
    
    out = []
    for chunk, distance in results:
        # Convert distance to similarity score (0..1)
        # Cosine distance is 1 - cosine similarity
        score = 1.0 - distance
        
        out.append(VectorResult(
            chunk_id=str(chunk.chunk_id),
            score=score,
            text=chunk.text,
            metadata=chunk.metadata_
        ))
        
    return out


# -----------------------------------------------------------------------------
# Canonical Blueprint Alias (§8.3)
# -----------------------------------------------------------------------------
# Blueprint uses search_vector_chunks naming convention

search_vector_chunks = search_chunks_vector
"""Canonical alias for search_chunks_vector per Blueprint §8.3."""