"""
Vector store abstractions.

Provides a pluggable interface for vector search backends such as pgvector
and Qdrant.
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any

import numpy as np
import requests
from cortex.common.exceptions import RetrievalError
from cortex.config.models import QdrantConfig
from pydantic import BaseModel, Field
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY, TEXT
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _normalize_output_dim(output_dim: int) -> int:
    """Validate output dimension to avoid invalid SQL constructs."""
    if isinstance(output_dim, bool) or not isinstance(output_dim, int):
        raise ValueError("output_dim must be a positive integer")
    if output_dim <= 0:
        raise ValueError("output_dim must be a positive integer")
    return output_dim


def _validate_embedding(embedding: list[float], expected_dim: int) -> np.ndarray:
    """Validate embedding dimensions and values. Returns numpy array."""
    if len(embedding) != expected_dim:
        raise RetrievalError(
            "Embedding dimension mismatch",
            query="vector_search",
            context={"expected_dim": expected_dim, "got": len(embedding)},
        )

    try:
        emb_array = np.asarray(embedding, dtype=float)
    except (TypeError, ValueError):
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


def _process_qdrant_point(point: dict[str, Any]) -> VectorResult:
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
    distance: float | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_type: str | None = None

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
        embedding: list[float],
        tenant_id: str,
        limit: int = 50,
        *,
        ef_search: int | None = None,
        conversation_ids: list[str] | None = None,
        is_attachment: bool | None = None,
        file_types: list[str] | None = None,  # P1 Fix: file_types filter
    ) -> list[VectorResult]:
        """Search for similar vectors."""


class PgvectorStore(VectorStore):
    """Vector store backed by Postgres/pgvector."""

    def __init__(self, session: Session, output_dim: int) -> None:
        self._session = session
        self._output_dim = _normalize_output_dim(output_dim)

    def search(
        self,
        embedding: list[float],
        tenant_id: str,
        limit: int = 50,
        *,
        ef_search: int | None = None,
        conversation_ids: list[str] | None = None,
        is_attachment: bool | None = None,
        file_types: list[str] | None = None,
    ) -> list[VectorResult]:
        """
        Search for similar vectors using pgvector.

        CRITICAL: This method has been refactored to prevent SQL injection.
        - All parameters are passed via bind variables.
        - Dynamic filters are handled with optional filters and bound arrays.

        PERFORMANCE:
        - The `file_types` filter on `extra_data` will be slow without an
          index. A GIN index is recommended on this column:
          `CREATE INDEX idx_chunks_extra_data_gin ON chunks USING GIN(extra_data);`
        """
        # P2 Fix: Cap limit to prevent resource exhaustion
        limit = min(limit, 500)
        emb_array = _validate_embedding(embedding, self._output_dim)

        conversation_id_list: list[uuid.UUID] | None = None
        if conversation_ids is not None:
            conversation_id_list = []
            for conv_id in conversation_ids:
                try:
                    conversation_id_list.append(uuid.UUID(str(conv_id)))
                except (TypeError, ValueError):
                    raise RetrievalError(
                        "Invalid conversation_id format",
                        query="vector_search",
                        context={"conversation_id": conv_id},
                    )

        file_types_param = file_types or None

        # Convert embedding to pgvector text format for the query
        embedding_str = "[" + ",".join(map(str, emb_array.tolist())) + "]"

        params: dict[str, Any] = {
            "tenant_id": tenant_id,
            "query_vec": embedding_str,
            "limit": limit,
            "conversation_ids": conversation_id_list,
            "is_attachment": is_attachment,
            "file_types": file_types_param,
        }

        # Optional HNSW tuning for recall/speed tradeoff
        ef_search_value = int(ef_search) if ef_search is not None else None

        # Fully parameterized query to prevent SQL injection
        stmt = text(
            f"""
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
            WHERE c.tenant_id = :tenant_id
              AND c.embedding IS NOT NULL
              AND (:conversation_ids IS NULL OR c.conversation_id = ANY(:conversation_ids))
              AND (:is_attachment IS NULL OR c.is_attachment = :is_attachment)
              AND (
                  :file_types IS NULL
                  OR (
                      c.extra_data->>'file_type' = ANY(:file_types)
                      OR c.extra_data->>'source_type' = ANY(:file_types)
                  )
              )
            ORDER BY distance
            LIMIT :limit
        """
        ).bindparams(
            bindparam(
                "conversation_ids",
                type_=ARRAY(PGUUID(as_uuid=True)),
            ),
            bindparam("file_types", type_=ARRAY(TEXT())),
        )

        transaction_context = nullcontext()
        if ef_search_value is not None and not self._session.in_transaction():
            transaction_context = self._session.begin()

        try:
            with transaction_context:
                if ef_search_value is not None:
                    # Set session-local configuration safely
                    self._session.execute(
                        text("SET LOCAL hnsw.ef_search = :ef_search"),
                        {"ef_search": ef_search_value},
                    )
                results = self._session.execute(stmt, params).fetchall()
        except SQLAlchemyError as exc:
            raise RetrievalError(
                "Postgres vector search failed",
                query="vector_search",
                context={"error": str(exc)},
            )

        return [_process_pgvector_row(row) for row in results]


class QdrantVectorStore(VectorStore):
    """Vector store backed by Qdrant."""

    def __init__(self, config: QdrantConfig, output_dim: int) -> None:
        self._config = config
        self._output_dim = _normalize_output_dim(output_dim)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["api-key"] = self._config.api_key
        return headers

    def search(
        self,
        embedding: list[float],
        tenant_id: str,
        limit: int = 50,
        *,
        ef_search: int | None = None,
        conversation_ids: list[str] | None = None,
        is_attachment: bool | None = None,
        file_types: list[str] | None = None,  # P1 Fix: file_types filter
    ) -> list[VectorResult]:
        # P2 Fix: Cap limit to prevent resource exhaustion
        limit = min(limit, 500)
        emb_array = _validate_embedding(embedding, self._output_dim)

        must_filters: list[dict[str, Any]] = [
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
                    ],
                    "min_should": 1,
                }
            )

        payload: dict[str, Any] = {
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
                    "reason": response.reason,
                },
            )

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError):
            raise RetrievalError(
                "Qdrant returned invalid JSON",
                query="vector_search",
                context={"status_code": response.status_code},
            )
        results = data.get("result", []) if isinstance(data, dict) else []

        out = [_process_qdrant_point(point) for point in results]
        logger.debug("Qdrant search returned %d chunks", len(out))
        return out
