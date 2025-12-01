"""
Hybrid Search (FTS + Vector + RRF).

Implements §8.3 of the Canonical Blueprint.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

# Import QueryClassification from dedicated module per Blueprint §8.2
from cortex.retrieval.query_classifier import QueryClassification


logger = logging.getLogger(__name__)


class KBSearchInput(BaseModel):
    """
    Retrieval tool input.
    
    Blueprint §8.4:
    * tenant_id: str
    * user_id: str
    * query: str
    * classification: Optional[QueryClassification]
    * k: Optional[int]
    * fusion_method: Literal["rrf", "weighted_sum"]
    * filters: Dict[str, Any]
    """
    tenant_id: str
    user_id: str
    query: str
    classification: Optional[QueryClassification] = None
    k: Optional[int] = None
    fusion_method: Literal["rrf", "weighted_sum"] = "rrf"
    filters: Dict[str, Any] = Field(default_factory=dict)


class SearchResultItem(BaseModel):
    """
    Search result item.
    
    Blueprint §8.4:
    * chunk_id: Optional[UUID]
    * score: float
    * thread_id: UUID
    * message_id: str
    * attachment_id: Optional[UUID]
    * highlights: List[str]
    * snippet: str
    * content: Optional[str]
    * source: Optional[str]
    * filename: Optional[str]
    * metadata: Dict[str, Any]
    """
    chunk_id: Optional[str]
    score: float
    thread_id: str
    message_id: str
    attachment_id: Optional[str] = None
    highlights: List[str] = Field(default_factory=list)
    snippet: str = ""
    content: Optional[str] = None
    source: Optional[str] = None
    filename: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResults(BaseModel):
    """
    Search results container.
    
    Blueprint §8.4:
    * type: Literal["search_results"]
    * query: str
    * reranker: Optional[str]
    * results: List[SearchResultItem]
    """
    type: Literal["search_results"] = "search_results"
    query: str
    reranker: Optional[str] = None
    results: List[SearchResultItem]


def apply_recency_boost(
    results: List[SearchResultItem],
    thread_updated_at: Dict[str, datetime],
    half_life_days: float = 30.0,
    boost_strength: float = 1.0
) -> List[SearchResultItem]:
    """
    Apply exponential decay recency boost to search results.
    
    Blueprint §8.6:
    Formula: boosted_score = score * (1 + boost_strength * exp(-decay * days_old))
    where decay = ln(2) / half_life_days
    """
    now = datetime.now(timezone.utc)
    decay_rate = math.log(2) / half_life_days
    
    for item in results:
        updated = thread_updated_at.get(item.thread_id)
        if updated:
            # Handle timezone-naive datetimes
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            days_old = (now - updated).total_seconds() / 86400
            boost = math.exp(-decay_rate * days_old) * boost_strength
            item.score = item.score * (1 + boost)
    
    return sorted(results, key=lambda x: x.score, reverse=True)


def deduplicate_by_hash(results: List[SearchResultItem]) -> List[SearchResultItem]:
    """
    Remove duplicate chunks by content_hash, keeping highest score.
    
    Blueprint §8.8: Deduplication by content_hash
    """
    seen_hashes: Dict[str, SearchResultItem] = {}
    deduped: List[SearchResultItem] = []
    
    for item in results:
        content_hash = item.metadata.get("content_hash")
        if content_hash:
            if content_hash in seen_hashes:
                # Keep the one with higher score
                if item.score > seen_hashes[content_hash].score:
                    # Replace in the list
                    deduped.remove(seen_hashes[content_hash])
                    seen_hashes[content_hash] = item
                    deduped.append(item)
            else:
                seen_hashes[content_hash] = item
                deduped.append(item)
        else:
            deduped.append(item)
    
    return deduped


def downweight_quoted_history(
    results: List[SearchResultItem],
    factor: float = 0.7
) -> List[SearchResultItem]:
    """
    Down-weight results from quoted_history chunks.
    
    Blueprint §8.8: Quoted history down-weighting
    """
    for item in results:
        chunk_type = item.metadata.get("chunk_type")
        if chunk_type == "quoted_history":
            item.score *= factor
    
    return sorted(results, key=lambda x: x.score, reverse=True)


def fuse_rrf(
    fts_results: List[SearchResultItem],
    vector_results: List[SearchResultItem],
    k: int = 60
) -> List[SearchResultItem]:
    """
    Reciprocal Rank Fusion of FTS and vector search results.
    
    Blueprint §8.7:
    RRF score = sum(1 / (k + rank_i)) for each ranking
    """
    scores: Dict[str, float] = {}
    items: Dict[str, SearchResultItem] = {}
    
    # Score from FTS ranking
    for rank, item in enumerate(fts_results, start=1):
        key = item.chunk_id or item.message_id
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
        items[key] = item
    
    # Score from vector ranking
    for rank, item in enumerate(vector_results, start=1):
        key = item.chunk_id or item.message_id
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
        if key not in items:
            items[key] = item
    
    # Sort by fused score
    fused = []
    for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        item = items[key]
        item.score = score
        fused.append(item)
    
    return fused


def tool_kb_search_hybrid(args: KBSearchInput) -> SearchResults:
    """
    Perform hybrid search (FTS + Vector + RRF).
    
    Blueprint §8.3:
    1. Prefilter
    2. Lexical search (FTS)
    3. Vector search (pgvector)
    4. Recency boost
    5. Deduplication
    6. Fusion (RRF)
    7. Rerank (optional)
    8. MMR (optional)
    9. Quoted history down-weighting
    """
    from cortex.db.session import SessionLocal
    from cortex.retrieval.fts_search import search_messages_fts, search_chunks_fts
    from cortex.retrieval.vector_search import search_chunks_vector
    from cortex.llm.client import embed_texts
    from cortex.config.loader import get_config
    
    config = get_config()
    k = args.k or config.search.k
    
    # 1. Embed query for vector search
    try:
        embeddings = embed_texts([args.query])
        query_embedding = embeddings[0]
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        raise

    with SessionLocal() as session:
        # 2. Lexical search (FTS)
        # If navigational, search messages. If semantic, search chunks.
        
        if args.classification and args.classification.type == "navigational":
             # Navigational: FTS on messages only
             fts_results = search_messages_fts(
                 session,
                 args.query,
                 args.tenant_id,
                 limit=k
             )
             # Convert to SearchResultItem
             final_results = [
                 SearchResultItem(
                     chunk_id=None,
                     score=r.score,
                     thread_id=r.thread_id,
                     message_id=r.message_id,
                     attachment_id=None,
                     highlights=[r.snippet],
                     snippet=r.snippet
                 )
                 for r in fts_results
             ]
             return SearchResults(query=args.query, results=final_results)

        # Semantic/Hybrid: FTS on chunks + Vector on chunks
        candidates_multiplier = config.search.candidates_multiplier
        
        fts_chunk_results = search_chunks_fts(
            session,
            args.query,
            args.tenant_id,
            limit=k * candidates_multiplier
        )
        
        # 3. Vector search
        vector_results = search_chunks_vector(
            session,
            query_embedding,
            args.tenant_id,
            limit=k * candidates_multiplier
        )
        
        # Convert to SearchResultItem format
        fts_items = [
            SearchResultItem(
                chunk_id=res.chunk_id,
                score=res.score,
                thread_id=res.thread_id,
                message_id=res.message_id,
                attachment_id=None,
                highlights=[res.snippet],
                snippet=res.snippet,
                metadata={}
            )
            for res in fts_chunk_results
        ]
        
        vector_items = [
            SearchResultItem(
                chunk_id=res.chunk_id,
                score=res.score,
                thread_id=str(res.metadata.get("thread_id", "")),
                message_id=str(res.metadata.get("message_id", "")),
                attachment_id=str(res.metadata.get("attachment_id", "")) if res.metadata.get("attachment_id") else None,
                highlights=[res.text[:200]] if res.text else [],
                snippet=res.text[:200] if res.text else "",
                content=res.text,
                metadata=res.metadata
            )
            for res in vector_results
        ]
        
        # 4. Deduplication by content_hash
        fts_deduped = deduplicate_by_hash(fts_items)
        vector_deduped = deduplicate_by_hash(vector_items)
        
        # 5. Fusion (RRF)
        fused_results = fuse_rrf(fts_deduped, vector_deduped, k=60)
        
        # 6. Get thread timestamps for recency boost
        thread_ids = list(set(r.thread_id for r in fused_results if r.thread_id))
        thread_updated_at = _get_thread_timestamps(session, thread_ids, args.tenant_id)
        
        # 7. Apply recency boost
        fused_results = apply_recency_boost(
            fused_results,
            thread_updated_at,
            half_life_days=config.search.half_life_days,
            boost_strength=config.search.recency_boost_strength
        )
        
        # 8. Down-weight quoted_history
        fused_results = downweight_quoted_history(fused_results, factor=0.7)
            
        # Return top-k
        return SearchResults(query=args.query, results=fused_results[:k])


def _get_thread_timestamps(session, thread_ids: List[str], tenant_id: str) -> Dict[str, datetime]:
    """Fetch updated_at timestamps for threads."""
    if not thread_ids:
        return {}
    
    from sqlalchemy import text
    
    stmt = text("""
        SELECT thread_id, updated_at
        FROM threads
        WHERE thread_id = ANY(:thread_ids)
        AND tenant_id = :tenant_id
    """)
    
    results = session.execute(stmt, {
        "thread_ids": thread_ids,
        "tenant_id": tenant_id
    }).fetchall()
    
    return {str(row.thread_id): row.updated_at for row in results}