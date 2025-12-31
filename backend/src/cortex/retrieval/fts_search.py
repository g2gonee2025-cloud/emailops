"""
Full-Text Search (FTS) retrieval.

Implements §8.2 of the Canonical Blueprint.
Adapted for Conversation-based schema (conversation_id instead of thread_id/message_id).
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from html import escape
from typing import Any

from cortex.intelligence.query_expansion import expand_for_fts
from pydantic import BaseModel, Field
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY, TEXT
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.exc import DataError, ProgrammingError
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# FTS configuration
# -----------------------------------------------------------------------------

# Keep config centralized so it's easy to swap (for non-English tenants, etc.)
_FTS_CONFIG = "english"
_TS_RANK_NORMALIZATION = 32
_MAX_FTS_TEXT_CHARS = 4000

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
) -> list[FTSResult]:
    """
    Perform FTS search on conversations.

    Blueprint §8.2 (adapted for Conversation schema):
    * FTS search on conversations subject AND summary_text
    * Returns conversation-level hits
    """
    query = (query or "").strip()
    if not query:
        return []

    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
    expanded_query = expand_for_fts(query)
    logger.debug("FTS query expanded", extra={"query_hash": query_hash})

    # Search conversations by subject (weight A) and summary (weight B)
    query_template = """
        WITH q AS (
            SELECT {tsquery_func}(:cfg, :query) AS tsq
        ),
        docs AS (
            SELECT
                conv.conversation_id,
                conv.subject,
                conv.summary_text,
                setweight(to_tsvector(:cfg, COALESCE(conv.subject, '')), 'A') ||
                setweight(to_tsvector(:cfg, COALESCE(conv.summary_text, '')), 'B') AS document
            FROM conversations conv
            WHERE conv.tenant_id = :tenant_id
        )
        SELECT
            docs.conversation_id,
            docs.subject,
            ts_rank_cd(
                docs.document,
                q.tsq, :rank_norm
            ) AS score,
            ts_headline(:cfg,
                COALESCE(docs.subject, '') || ' | ' || COALESCE(docs.summary_text, ''),
                q.tsq, :headline_opts
            ) AS snippet
        FROM docs
        CROSS JOIN q
        WHERE docs.document @@ q.tsq
        ORDER BY score DESC
        LIMIT :limit
    """

    params = {
        "query": expanded_query,
        "tenant_id": tenant_id,
        "limit": limit,
        "cfg": _FTS_CONFIG,
        "headline_opts": _HEADLINE_OPTIONS,
        "rank_norm": _TS_RANK_NORMALIZATION,
    }

    try:
        stmt = text(query_template.format(tsquery_func="to_tsquery"))
        results = session.execute(stmt, params).fetchall()
    except (DataError, ProgrammingError):
        logger.warning(
            "Invalid FTS query; falling back to plainto_tsquery",
            extra={"query_hash": query_hash},
        )
        params["query"] = query
        stmt = text(query_template.format(tsquery_func="plainto_tsquery"))
        results = session.execute(stmt, params).fetchall()

    return [
        FTSResult(
            conversation_id=str(row.conversation_id),
            subject=str(row.subject or ""),
            score=float(row.score or 0.0),
            snippet=escape(str(row.snippet or "")),
        )
        for row in results
    ]


class ChunkFTSResult(BaseModel):
    """Chunk FTS search result."""

    chunk_id: str
    conversation_id: str
    chunk_type: str | None = None
    score: float
    snippet: str
    text: str
    is_attachment: bool | None = None
    attachment_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def search_chunks_fts(
    session: Session,
    query: str,
    tenant_id: str,
    limit: int = 50,
    conversation_ids: list[str] | None = None,
    is_attachment: bool | None = None,
    file_types: list[str] | None = None,
) -> list[ChunkFTSResult]:
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

    if conversation_ids is not None and not conversation_ids:
        return []

    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
    expanded_query = expand_for_fts(query)
    logger.debug("FTS query expanded", extra={"query_hash": query_hash})

    conversation_id_list: list[uuid.UUID] | None = None
    if conversation_ids is not None:
        conversation_id_list = []
        for conv_id in conversation_ids:
            try:
                conversation_id_list.append(uuid.UUID(str(conv_id)))
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid conversation_id filter for FTS query",
                    extra={"query_hash": query_hash},
                )
                return []

    params: dict[str, Any] = {
        "query": expanded_query,
        "tenant_id": tenant_id,
        "limit": limit,
        "cfg": _FTS_CONFIG,
        "headline_opts": _HEADLINE_OPTIONS,
        "rank_norm": _TS_RANK_NORMALIZATION,
        "max_text_chars": _MAX_FTS_TEXT_CHARS,
    }

    where_clauses = ["c.tenant_id = :tenant_id", "c.tsv_text @@ q.tsq"]

    if conversation_id_list is not None:
        where_clauses.append("c.conversation_id = ANY(:conversation_ids)")
        params["conversation_ids"] = conversation_id_list

    # Attachment filter
    if is_attachment is not None:
        where_clauses.append("c.is_attachment = :is_attachment")
        params["is_attachment"] = is_attachment

    # File types filter (e.g., type:pdf returns only PDF chunks)
    if file_types:
        where_clauses.append("c.extra_data->>'file_type' = ANY(:file_types)")
        params["file_types"] = file_types

    where_sql = " AND ".join(where_clauses)

    # Build the query string programmatically to make the dynamic nature of
    # the WHERE clause explicit. While the `where_clauses` are constructed from
    # safe, static strings, this pattern avoids formatting the entire SQL
    # string at once, which is less prone to future injection vulnerabilities.
    query_str = """
        WITH q AS (
            SELECT {tsquery_func}(:cfg, :query) AS tsq
        )
        SELECT
            c.chunk_id,
            c.conversation_id,
            c.chunk_type,
            LEFT(c.text, :max_text_chars) AS text,
            c.is_attachment,
            c.attachment_id,
            c.extra_data,
            ts_rank_cd(c.tsv_text, q.tsq, :rank_norm) AS score,
            ts_headline(
                :cfg,
                LEFT(c.text, :max_text_chars),
                q.tsq,
                :headline_opts
            ) AS snippet
        FROM chunks c
        CROSS JOIN q
        WHERE """
    query_str += where_sql
    query_str += " ORDER BY score DESC LIMIT :limit"

    # Build bindparams conditionally based on which filters are active
    bind_params_list = []
    if conversation_id_list is not None:
        bind_params_list.append(
            bindparam("conversation_ids", type_=ARRAY(PGUUID(as_uuid=True)))
        )
    if file_types:
        bind_params_list.append(bindparam("file_types", type_=ARRAY(TEXT())))

    stmt = text(query_str.format(tsquery_func="to_tsquery"))
    if bind_params_list:
        stmt = stmt.bindparams(*bind_params_list)

    try:
        results = session.execute(stmt, params).fetchall()
    except (DataError, ProgrammingError):
        logger.warning(
            "Invalid FTS query; falling back to plainto_tsquery",
            extra={"query_hash": query_hash},
        )
        params["query"] = query
        fallback_stmt = text(query_str.format(tsquery_func="plainto_tsquery"))
        if bind_params_list:
            fallback_stmt = fallback_stmt.bindparams(*bind_params_list)
        results = session.execute(fallback_stmt, params).fetchall()

    return [
        ChunkFTSResult(
            chunk_id=str(row.chunk_id),
            conversation_id=str(row.conversation_id),
            chunk_type=str(row.chunk_type) if row.chunk_type is not None else None,
            score=float(row.score or 0.0),
            snippet=escape(str(row.snippet or "")),
            text=str(row.text or ""),
            is_attachment=row.is_attachment if row.is_attachment is not None else None,
            attachment_id=str(row.attachment_id) if row.attachment_id else None,
            metadata=row.extra_data if isinstance(row.extra_data, dict) else {},
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
