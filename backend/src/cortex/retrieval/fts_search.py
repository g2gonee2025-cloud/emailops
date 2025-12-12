"""
Full-Text Search (FTS) retrieval.

Implements §8.2 of the Canonical Blueprint.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# FTS configuration
# -----------------------------------------------------------------------------

# Keep config centralized so it's easy to swap (for non-English tenants, etc.)
_FTS_CONFIG = "english"

# ts_headline defaults to HTML (<b>...</b>). PostgreSQL docs warn the output is
# not guaranteed to be safe for direct inclusion in web pages, so we use
# non-HTML markers here and still recommend HTML-escaping at render time.
#
# See: PostgreSQL docs §12.3.4 (Highlighting Results).
_HEADLINE_OPTIONS = (
    "MaxFragments=2, MaxWords=15, MinWords=5, ShortWord=3, "
    'StartSel=[[, StopSel=]], FragmentDelimiter=" ... "'
)


class FTSResult(BaseModel):
    """FTS search result."""

    message_id: str
    thread_id: str
    subject: str
    score: float
    snippet: str


def search_messages_fts(
    session: Session, query: str, tenant_id: str, limit: int = 50
) -> List[FTSResult]:
    """
    Perform FTS search on messages.

    Blueprint §8.2:
    * FTS search on messages.tsv_subject_body
    * Returns message-level hits
    """
    query = (query or "").strip()
    if not query:
        return []

    # Blueprint §8.2: FTS search on messages.tsv_subject_body
    # Use websearch_to_tsquery for robust query parsing.
    # Use ts_rank_cd for cover density ranking and normalization=32 to scale to 0..1.

    stmt = text(
        """
        WITH q AS (
            SELECT websearch_to_tsquery(:cfg, :query) AS tsq
        )
        SELECT
            m.message_id,
            m.thread_id,
            m.subject,
            ts_rank_cd(m.tsv_subject_body, q.tsq, 32) AS score,
            ts_headline(:cfg, m.body_plain, q.tsq, :headline_opts) AS snippet
        FROM messages m, q
        WHERE
            m.tenant_id = :tenant_id
            AND m.tsv_subject_body @@ q.tsq
        ORDER BY score DESC
        LIMIT :limit
    """
    )

    results = session.execute(
        stmt,
        {
            "query": query,
            "tenant_id": tenant_id,
            "limit": limit,
            "cfg": _FTS_CONFIG,
            "headline_opts": _HEADLINE_OPTIONS,
        },
    ).fetchall()

    return [
        FTSResult(
            message_id=row.message_id,
            thread_id=str(row.thread_id),
            subject=row.subject or "",
            score=row.score,
            snippet=row.snippet or "",
        )
        for row in results
    ]


class ChunkFTSResult(BaseModel):
    """Chunk FTS search result."""

    chunk_id: str
    thread_id: str
    message_id: str
    chunk_type: str
    score: float
    snippet: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


def search_chunks_fts(
    session: Session,
    query: str,
    tenant_id: str,
    limit: int = 50,
    thread_ids: List[str] | None = None,
) -> List[ChunkFTSResult]:
    """
    Perform FTS search on chunks.

    Blueprint §8.3:
    * FTS search on chunks.tsv_text
    * Returns chunk-level hits for hybrid fusion
    """
    query = (query or "").strip()
    if not query:
        return []

    thread_filter_sql = ""
    params: Dict[str, Any] = {
        "query": query,
        "tenant_id": tenant_id,
        "limit": limit,
        "cfg": _FTS_CONFIG,
        "headline_opts": _HEADLINE_OPTIONS,
    }

    if thread_ids:
        thread_filter_sql = "AND c.thread_id = ANY(CAST(:thread_ids AS UUID[]))"
        params["thread_ids"] = thread_ids

    stmt = text(
        f"""
        WITH q AS (
            SELECT websearch_to_tsquery(:cfg, :query) AS tsq
        )
        SELECT
            c.chunk_id,
            c.thread_id,
            c.message_id,
            c.chunk_type,
            c.text,
            c.metadata,
            ts_rank_cd(c.tsv_text, q.tsq, 32) AS score,
            ts_headline(:cfg, c.text, q.tsq, :headline_opts) AS snippet
        FROM chunks c, q
        WHERE
            c.tenant_id = :tenant_id
            AND c.tsv_text @@ q.tsq
            {thread_filter_sql}
        ORDER BY score DESC
        LIMIT :limit
    """
    )

    results = session.execute(stmt, params).fetchall()

    return [
        ChunkFTSResult(
            chunk_id=str(row.chunk_id),
            thread_id=str(row.thread_id),
            message_id=row.message_id or "",
            chunk_type=row.chunk_type or "other",
            score=row.score,
            snippet=row.snippet or "",
            text=row.text or "",
            metadata=row.metadata or {},
        )
        for row in results
    ]


# -----------------------------------------------------------------------------
# Canonical Blueprint Aliases (§8.2, §8.3)
# -----------------------------------------------------------------------------
# Blueprint uses search_fts_chunks, search_fts_messages naming convention

search_fts_chunks = search_chunks_fts
"""Canonical alias for search_chunks_fts per Blueprint §8.3."""

search_fts_messages = search_messages_fts
"""Canonical alias for search_messages_fts per Blueprint §8.2."""
