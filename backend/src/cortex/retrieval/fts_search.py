"""
Full-Text Search (FTS) retrieval.

Implements §8.2 of the Canonical Blueprint.
Adapted for Conversation-based schema (conversation_id instead of thread_id/message_id).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from cortex.intelligence.query_expansion import expand_for_fts

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
    """FTS search result (conversation-level)."""

    conversation_id: str
    subject: str
    score: float
    snippet: str


def search_conversations_fts(
    session: Session, query: str, tenant_id: str, limit: int = 50
) -> List[FTSResult]:
    """
    Perform FTS search on conversations.

    Blueprint §8.2 (adapted for Conversation schema):
    * FTS search on conversations subject AND summary_text
    * Returns conversation-level hits
    """
    query = (query or "").strip()
    if not query:
        return []

    # Expand query with synonyms for better recall.
    expanded_query = expand_for_fts(query)
    logger.debug(f"Original FTS query: '{query}', Expanded: '{expanded_query}'")

    # Search conversations by subject (weight A) and summary (weight B)
    stmt = text(
        """
        WITH q AS (
            SELECT to_tsquery(:cfg, :query) AS tsq
        )
        SELECT
            conv.conversation_id,
            conv.subject,
            ts_rank_cd(
                setweight(to_tsvector(:cfg, COALESCE(conv.subject, '')), 'A') ||
                setweight(to_tsvector(:cfg, COALESCE(conv.summary_text, '')), 'B'),
                q.tsq, 32
            ) AS score,
            ts_headline(:cfg,
                COALESCE(conv.subject, '') || ' | ' || COALESCE(conv.summary_text, ''),
                q.tsq, :headline_opts
            ) AS snippet
        FROM conversations conv, q
        WHERE
            conv.tenant_id = :tenant_id
            AND (
                setweight(to_tsvector(:cfg, COALESCE(conv.subject, '')), 'A') ||
                setweight(to_tsvector(:cfg, COALESCE(conv.summary_text, '')), 'B')
            ) @@ q.tsq
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
            conversation_id=str(row.conversation_id),
            subject=str(row.subject or ""),
            score=float(row.score or 0.0),
            snippet=str(row.snippet or ""),
        )
        for row in results
    ]


class ChunkFTSResult(BaseModel):
    """Chunk FTS search result."""

    chunk_id: str
    conversation_id: str
    chunk_type: str
    score: float
    snippet: str
    text: str
    is_attachment: bool = False
    attachment_id: str | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def search_chunks_fts(
    session: Session,
    query: str,
    tenant_id: str,
    limit: int = 50,
    conversation_ids: List[str] | None = None,
    is_attachment: bool | None = None,
    file_types: List[str] | None = None,  # P1 Fix: Add file_types filter
) -> List[ChunkFTSResult]:
    """
    Perform FTS search on chunks.

    Blueprint §8.3 (adapted for Conversation schema):
    * FTS search on chunks.tsv_text
    * Returns chunk-level hits for hybrid fusion
    * Supports file_types filtering (e.g., ['pdf', 'docx'])
    """
    query = (query or "").strip()
    if not query:
        return []

    # Expand query with synonyms for better recall.
    expanded_query = expand_for_fts(query)
    logger.debug(f"Original FTS query: '{query}', Expanded: '{expanded_query}'")

    conversation_filter_sql = ""
    params: Dict[str, Any] = {
        "query": expanded_query,
        "tenant_id": tenant_id,
        "limit": limit,
        "cfg": _FTS_CONFIG,
        "headline_opts": _HEADLINE_OPTIONS,
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

    # P1 Fix: File types filter (e.g., type:pdf returns only PDF chunks)
    file_types_filter_sql = ""
    if file_types:
        # Filter by chunk metadata.source containing file extension
        # OR by chunk_type for attachment chunks
        file_types_filter_sql = "AND (c.extra_data->>'file_type' = ANY(:file_types) OR c.extra_data->>'source_type' = ANY(:file_types))"
        params["file_types"] = file_types

    stmt = text(
        f"""
        WITH q AS (
            SELECT to_tsquery(:cfg, :query) AS tsq
        )
        SELECT
            c.chunk_id,
            c.conversation_id,
            c.chunk_type,
            c.text,
            c.is_attachment,
            c.attachment_id,
            c.extra_data,
            ts_rank_cd(c.tsv_text, q.tsq, 32) AS score,
            ts_headline(:cfg, c.text, q.tsq, :headline_opts) AS snippet
        FROM chunks c, q
        WHERE
            c.tenant_id = :tenant_id
            AND c.tsv_text @@ q.tsq
            {conversation_filter_sql}
            {attachment_filter_sql}
            {file_types_filter_sql}
        ORDER BY score DESC
        LIMIT :limit
    """
    )

    results = session.execute(stmt, params).fetchall()

    return [
        ChunkFTSResult(
            chunk_id=str(row.chunk_id),
            conversation_id=str(row.conversation_id),
            chunk_type=str(row.chunk_type or "message_body"),
            score=float(row.score or 0.0),
            snippet=str(row.snippet or ""),
            text=str(row.text or ""),
            is_attachment=bool(row.is_attachment),
            attachment_id=str(row.attachment_id) if row.attachment_id else None,
            metadata=dict(row.extra_data) if row.extra_data else {},
        )
        for row in results
    ]


# -----------------------------------------------------------------------------
# Canonical Blueprint Aliases (§8.2, §8.3)
# -----------------------------------------------------------------------------
# Blueprint uses search_fts_chunks, search_fts_messages naming convention
# We adapt to conversation-based schema

search_fts_chunks = search_chunks_fts
"""Canonical alias for search_chunks_fts per Blueprint §8.3."""

search_fts_conversations = search_conversations_fts
"""Conversation-level FTS search (adapted from Blueprint §8.2 search_fts_messages)."""

# Backward compat alias (search_messages_fts -> search_conversations_fts)
search_messages_fts = search_conversations_fts
