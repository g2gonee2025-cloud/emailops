"""
Full-Text Search (FTS) retrieval.

Implements §8.2 of the Canonical Blueprint.
"""
from __future__ import annotations

import logging
from typing import List

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session


logger = logging.getLogger(__name__)


class FTSResult(BaseModel):
    """FTS search result."""
    message_id: str
    thread_id: str
    subject: str
    score: float
    snippet: str


def search_messages_fts(
    session: Session,
    query: str,
    tenant_id: str,
    limit: int = 50
) -> List[FTSResult]:
    """
    Perform FTS search on messages.
    
    Blueprint §8.2:
    * FTS search on messages.tsv_subject_body
    * Returns message-level hits
    """
    # Blueprint §8.2: FTS search on messages.tsv_subject_body
    # Use websearch_to_tsquery for robust query parsing
    
    stmt = text("""
        SELECT
            message_id,
            thread_id,
            subject,
            ts_rank(tsv_subject_body, websearch_to_tsquery('english', :query)) as score,
            ts_headline('english', body_plain, websearch_to_tsquery('english', :query)) as snippet
        FROM messages
        WHERE
            tenant_id = :tenant_id
            AND tsv_subject_body @@ websearch_to_tsquery('english', :query)
        ORDER BY score DESC
        LIMIT :limit
    """)
    
    results = session.execute(stmt, {
        "query": query,
        "tenant_id": tenant_id,
        "limit": limit
    }).fetchall()
    
    return [
        FTSResult(
            message_id=row.message_id,
            thread_id=str(row.thread_id),
            subject=row.subject or "",
            score=row.score,
            snippet=row.snippet or ""
        )
        for row in results
    ]

class ChunkFTSResult(BaseModel):
    """Chunk FTS search result."""
    chunk_id: str
    thread_id: str
    message_id: str
    score: float
    snippet: str


def search_chunks_fts(
    session: Session,
    query: str,
    tenant_id: str,
    limit: int = 50
) -> List[ChunkFTSResult]:
    """
    Perform FTS search on chunks.
    
    Blueprint §8.3:
    * FTS search on chunks.tsv_text
    * Returns chunk-level hits for hybrid fusion
    """
    stmt = text("""
        SELECT
            chunk_id,
            thread_id,
            message_id,
            ts_rank(tsv_text, websearch_to_tsquery('english', :query)) as score,
            ts_headline('english', text, websearch_to_tsquery('english', :query)) as snippet
        FROM chunks
        WHERE
            tenant_id = :tenant_id
            AND tsv_text @@ websearch_to_tsquery('english', :query)
        ORDER BY score DESC
        LIMIT :limit
    """)
    
    results = session.execute(stmt, {
        "query": query,
        "tenant_id": tenant_id,
        "limit": limit
    }).fetchall()
    
    return [
        ChunkFTSResult(
            chunk_id=str(row.chunk_id),
            thread_id=str(row.thread_id),
            message_id=row.message_id or "",
            score=row.score,
            snippet=row.snippet or ""
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

