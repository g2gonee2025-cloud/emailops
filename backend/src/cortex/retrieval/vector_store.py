"""
Vector store abstractions.

Provides a pluggable interface for vector search backends such as pgvector
and Qdrant.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from cortex.common.exceptions import RetrievalError
from cortex.config.models import QdrantConfig
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _validate_embedding(embedding: List[float], expected_dim: int) -> np.ndarray:
    """Validate embedding dimensions and values. Returns numpy array."""
    if len(embedding) != expected_dim:
        raise RetrievalError(
            "Embedding dimension mismatch",
            query="vector_search",
            context={"expected_dim": expected_dim, "got": len(embedding)},
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

    return emb_array


def _process_pgvector_row(row: Any) -> VectorResult:
    """Process a pgvector result row into a VectorResult."""
    # pgvector's cosine distance operator (<=>)  is 1 - cosine_similarity.
    distance = float(row.distance) if row.distance is not None else None
    cosine_sim = 1.0 - distance if distance is not None else -1.0
    score = max(0.0, min(1.0, (cosine_sim + 1.0) / 2.0))

    conversation_id = str(row.conversation_id) if row.conversation_id else ""
    attachment_id = str(row.attachment_id) if row.attachment_id else ""

    metadata = dict(row.extra_data) if isinstance(row.extra_data, dict) else {}
    metadata.setdefault("conversation_id", conversation_id)
    metadata.setdefault("attachment_id", attachment_id)
    if row.chunk_type:
        metadata.setdefault("chunk_type", row.chunk_type)

    return VectorResult(
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


def _process_qdrant_point(point: Dict[str, Any]) -> VectorResult:
    """Process a Qdrant point into a VectorResult."""
    payload_data = point.get("payload") or {}
    if not isinstance(payload_data, dict):
        payload_data = {}

    chunk_id = str(payload_data.get("chunk_id") or point.get("id") or "")
    text = payload_data.get("text") or ""
    conversation_id = str(payload_data.get("conversation_id") or "")
    attachment_id = str(payload_data.get("attachment_id") or "")
    chunk_type = payload_data.get("chunk_type")

    metadata = payload_data.get("extra_data")
    if not isinstance(metadata, dict):
        metadata = payload_data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.setdefault("conversation_id", conversation_id)
    metadata.setdefault("attachment_id", attachment_id)
    if chunk_type:
        metadata.setdefault("chunk_type", chunk_type)

    score = point.get("score")
    try:
        score_value = float(score) if score is not None else 0.0
    except (TypeError, ValueError):
        score_value = 0.0
    score_value = max(0.0, min(1.0, score_value))

    return VectorResult(
        chunk_id=chunk_id,
        score=score_value,
        text=text,
        conversation_id=conversation_id,
        attachment_id=attachment_id,
        is_attachment=bool(payload_data.get("is_attachment", False)),
        distance=None,
        metadata=metadata,
        chunk_type=chunk_type,
    )


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


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def search(
        self,
        embedding: List[float],
        tenant_id: str,
        limit: int = 50,
        *,
        ef_search: Optional[int] = None,
        conversation_ids: Optional[List[str]] = None,
        is_attachment: Optional[bool] = None,
        file_types: Optional[List[str]] = None,  # P1 Fix: file_types filter
    ) -> List[VectorResult]:
        """Search for similar vectors."""


class PgvectorStore(VectorStore):
    """Vector store backed by Postgres/pgvector."""

    def __init__(self, session: Session, output_dim: int) -> None:
        self._session = session
        self._output_dim = output_dim

    def search(
        self,
        embedding: List[float],
        tenant_id: str,
        limit: int = 50,
        *,
        ef_search: Optional[int] = None,
        conversation_ids: Optional[List[str]] = None,
        is_attachment: Optional[bool] = None,
        file_types: Optional[List[str]] = None,  # P1 Fix: file_types filter
    ) -> List[VectorResult]:
        # P2 Fix: Cap limit to prevent resource exhaustion
        limit = min(limit, 500)
        emb_array = _validate_embedding(embedding, self._output_dim)

        # Convert embedding to pgvector text format
        embedding_str = "[" + ",".join(repr(float(v)) for v in emb_array.tolist()) + "]"

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
            attachment_filter_sql = (
                "AND c.is_attachment = TRUE"
                if is_attachment
                else "AND c.is_attachment = FALSE"
            )

        # P1 Fix: File types filter
        file_types_filter_sql = ""
        if file_types:
            file_types_filter_sql = "AND (c.extra_data->>'file_type' = ANY(:file_types) OR c.extra_data->>'source_type' = ANY(:file_types))"
            params["file_types"] = file_types

        # Optional HNSW tuning for recall/speed tradeoff
        hnsw_settings_cte = ""
        if ef_search is not None:
            params["ef_search"] = str(int(ef_search))
            hnsw_settings_cte = "WITH settings AS (SELECT set_config('hnsw.ef_search', :ef_search, true))"

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
                c.embedding <=> CAST(:query_vec AS halfvec({self._output_dim})) AS distance
            FROM chunks c
            {", settings" if ef_search is not None else ""}
            WHERE c.tenant_id = :tenant_id
              AND c.embedding IS NOT NULL
            {conversation_filter_sql}
            {attachment_filter_sql}
            {file_types_filter_sql}
            ORDER BY distance
            LIMIT :limit
        """
        )

        results = self._session.execute(stmt, params).fetchall()
        return [_process_pgvector_row(row) for row in results]


class QdrantVectorStore(VectorStore):
    """Vector store backed by Qdrant."""

    def __init__(self, config: QdrantConfig, output_dim: int) -> None:
        self._config = config
        self._output_dim = output_dim

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["api-key"] = self._config.api_key
        return headers

    def search(
        self,
        embedding: List[float],
        tenant_id: str,
        limit: int = 50,
        *,
        ef_search: Optional[int] = None,
        conversation_ids: Optional[List[str]] = None,
        is_attachment: Optional[bool] = None,
        file_types: Optional[List[str]] = None,  # P1 Fix: file_types filter
    ) -> List[VectorResult]:
        # P2 Fix: Cap limit to prevent resource exhaustion
        limit = min(limit, 500)
        emb_array = _validate_embedding(embedding, self._output_dim)

        must_filters: List[Dict[str, Any]] = [
            {"key": "tenant_id", "match": {"value": tenant_id}},
        ]
        if conversation_ids is not None:
            if conversation_ids:
                must_filters.append(
                    {"key": "conversation_id", "match": {"any": conversation_ids}}
                )
            else:
                return []
        if is_attachment is not None:
            must_filters.append(
                {"key": "is_attachment", "match": {"value": is_attachment}}
            )
        # P1 Fix: file_types filter for Qdrant
        if file_types:
            must_filters.append(
                {
                    "should": [
                        {"key": "file_type", "match": {"any": file_types}},
                        {"key": "source_type", "match": {"any": file_types}},
                    ]
                }
            )

        payload: Dict[str, Any] = {
            "vector": emb_array.tolist(),
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
            "filter": {"must": must_filters},
        }
        if ef_search is not None:
            payload["params"] = {"hnsw_ef": int(ef_search)}

        url = (
            f"{self._config.url.rstrip('/')}/collections/"
            f"{self._config.collection_name}/points/search"
        )

        try:
            response = requests.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=10,
            )
        except requests.RequestException as exc:
            raise RetrievalError(
                "Failed to query Qdrant",
                query="vector_search",
                context={"error": str(exc)},
            )

        if response.status_code != 200:
            raise RetrievalError(
                "Qdrant search failed",
                query="vector_search",
                context={
                    "status_code": response.status_code,
                    "detail": response.text,
                },
            )

        try:
            data = response.json()
        except requests.JSONDecodeError:
            raise RetrievalError(
                "Qdrant returned invalid JSON",
                query="vector_search",
                context={"response": response.text[:200]},
            )
        results = data.get("result", []) if isinstance(data, dict) else []

        out = [_process_qdrant_point(point) for point in results]
        logger.info("Qdrant search returned %d chunks", len(out))
        return out
