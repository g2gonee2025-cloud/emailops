"""
Hybrid Search (FTS + Vector + RRF).

Implements §8.3 of the Canonical Blueprint.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from cortex.common.exceptions import RetrievalError
from cortex.common.types import Err, Ok, Result
from cortex.retrieval._hybrid_helpers import (
    _convert_fts_to_items,
    _convert_vector_to_items,
    _get_query_embedding,
    _resolve_target_conversations,
)
from cortex.retrieval.filters import parse_filter_grammar

# Import QueryClassification from dedicated module per Blueprint §8.2
from cortex.retrieval.query_classifier import QueryClassification
from cortex.retrieval.reranking import (
    apply_mmr,
    call_external_reranker,
    rerank_results,
)
from cortex.retrieval.results import SearchResultItem, SearchResults
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Constants
RRF_K_DEFAULT = 60
QUOTED_HISTORY_DOWNWEIGHT_FACTOR = 0.7
RECENCY_HALF_LIFE_DAYS = 30.0
RECENCY_BOOST_STRENGTH = 1.0
SUMMARY_BOOST_FACTOR = 1.2
SUMMARY_SEARCH_LIMIT = 5
WEIGHTED_SUM_ALPHA_DEFAULT = 0.5


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


def apply_recency_boost(
    results: List[SearchResultItem],
    conversation_updated_at: Dict[str, datetime],
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
    boost_strength: float = RECENCY_BOOST_STRENGTH,
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
        if not item.conversation_id:
            continue

        updated = conversation_updated_at.get(item.conversation_id)
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
    # Keep the best-scoring item per hash.
    # Avoid O(n^2) list removals when many near-duplicates are present.
    best_by_hash: Dict[str, SearchResultItem] = {}
    passthrough: List[SearchResultItem] = []

    for item in results:
        content_hash = item.content_hash or item.metadata.get("content_hash")
        if not content_hash:
            passthrough.append(item)
            continue

        existing = best_by_hash.get(content_hash)
        if existing is None or item.score > existing.score:
            best_by_hash[content_hash] = item

    # Order isn't critical here since later steps re-sort, but keep stable-ish output.
    return passthrough + list(best_by_hash.values())


def downweight_quoted_history(
    results: List[SearchResultItem], factor: float = QUOTED_HISTORY_DOWNWEIGHT_FACTOR
) -> List[SearchResultItem]:
    """
    Down-weight results from quoted_history chunks.

    Blueprint §8.8: Quoted history down-weighting
    """
    for item in results:
        chunk_type = item.metadata.get("chunk_type")
        if chunk_type == "quoted_history":
            item.score *= factor
    return results


def _merge_result_fields(
    into: SearchResultItem, other: SearchResultItem
) -> SearchResultItem:
    """Merge fields from another SearchResultItem into an existing one."""
    # IDs
    if not into.conversation_id:
        into.conversation_id = other.conversation_id
    if not into.attachment_id:
        into.attachment_id = other.attachment_id

    # Content
    if not into.content:
        into.content = other.content
    if not into.snippet:
        into.snippet = other.snippet

    # Highlights
    if other.highlights:
        if into.highlights is None:
            into.highlights = []
        # Use simple list extend and let deduplication happen if strictly needed later,
        # or just ensure uniqueness simply.
        existing = set(into.highlights or [])
        for h in other.highlights:
            if h and h not in existing:
                into.highlights.append(h)
                existing.add(h)

    # Scores - prefer non-zero/non-none
    if into.lexical_score is None:
        into.lexical_score = other.lexical_score
    if into.vector_score is None:
        into.vector_score = other.vector_score

    # Metadata & Hash
    if other.metadata:
        into.metadata.update(
            {k: v for k, v in other.metadata.items() if k not in into.metadata}
        )

    if not into.content_hash:
        into.content_hash = other.content_hash

    return into


def fuse_rrf(
    fts_results: List[SearchResultItem],
    vector_results: List[SearchResultItem],
    k: int = RRF_K_DEFAULT,
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
        key = item.chunk_id or item.content_hash or f"fts-unkeyed-{rank}"
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
        if key in items:
            items[key] = _merge_result_fields(items[key], item)
        else:
            items[key] = item

    # Score from vector ranking
    for rank, item in enumerate(vector_results, start=1):
        key = item.chunk_id or item.content_hash or f"vector-unkeyed-{rank}"
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
        if key in items:
            items[key] = _merge_result_fields(items[key], item)
        else:
            items[key] = item

    # Sort by fused score
    fused = []
    for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        item = items[key]
        item.score = score
        item.fusion_score = score
        fused.append(item)

    return fused


def fuse_weighted_sum(
    fts_results: List[SearchResultItem],
    vector_results: List[SearchResultItem],
    alpha: float = WEIGHTED_SUM_ALPHA_DEFAULT,
) -> List[SearchResultItem]:
    """Fuse rankings via weighted sum.

    We assume lexical and vector signals are already scaled to roughly [0, 1]
    (ts_rank_cd(..., 32) and cosine-similarity-to-[0,1] mapping). That makes
    fusion stable without ad-hoc max-normalization.
    """

    combined: Dict[str, SearchResultItem] = {}

    for i, item in enumerate(fts_results):
        key = item.chunk_id or item.content_hash or f"fts-unkeyed-{i}"
        if key in combined:
            combined[key] = _merge_result_fields(combined[key], item)
        else:
            combined[key] = item

    for i, item in enumerate(vector_results):
        key = item.chunk_id or item.content_hash or f"vector-unkeyed-{i}"
        if key in combined:
            combined[key] = _merge_result_fields(combined[key], item)
        else:
            combined[key] = item

    for item in combined.values():
        lex = float(item.lexical_score or 0.0)
        vec = float(item.vector_score or 0.0)
        item.fusion_score = alpha * vec + (1 - alpha) * lex
        item.score = item.fusion_score

    return sorted(combined.values(), key=lambda i: i.fusion_score or 0.0, reverse=True)


async def tool_kb_search_hybrid(args: KBSearchInput) -> Result[SearchResults, RetrievalError]:
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
    try:
        from cortex.config.loader import get_config
        from cortex.db.session import SessionLocal
        from cortex.retrieval.fts_search import (
            search_chunks_fts,
            search_conversations_fts,
        )
        from cortex.retrieval.vector_search import search_chunks_vector

        config = get_config()
        k = args.k or config.search.k
        fusion_method = args.fusion_method or config.search.fusion_strategy

        candidates_multiplier = config.search.candidates_multiplier

        # 0. Parse filters from query (e.g., from:john@example.com subject:"budget")
        parsed_filters, clean_query = parse_filter_grammar(args.query)
        if not parsed_filters.is_empty():
            logger.debug(f"Parsed filters: {parsed_filters.to_dict()}")

        # Use cleaned query for embedding and FTS
        search_query = clean_query if clean_query.strip() else args.query

        # 1. Embed query for vector search (with caching)
        query_embedding = _get_query_embedding(search_query, config)

        with SessionLocal() as session:
            from cortex.db.session import set_session_tenant

            set_session_tenant(session, args.tenant_id)

            # 1.5. Summary-Awareness: Find highly relevant threads by summary
            summary_boost_ids = set()
            try:
                summary_hits = search_conversations_fts(
                    session,
                    search_query,
                    args.tenant_id,
                    limit=SUMMARY_SEARCH_LIMIT,
                )
                summary_boost_ids = {h.conversation_id for h in summary_hits}
                if summary_boost_ids:
                    logger.info(
                        f"Summary boost active for threads: {summary_boost_ids}"
                    )
            except Exception as e:
                logger.warning("Summary search failed: %s", e)

            # 2. Resolve Conversations (Navigational + Filters)
            final_conversation_ids = _resolve_target_conversations(
                session, args, search_query, k * candidates_multiplier, parsed_filters
            )

            # 3. Lexical search (FTS) on chunks
            fts_chunk_results = search_chunks_fts(
                session,
                search_query,
                args.tenant_id,
                limit=k * candidates_multiplier,
                conversation_ids=final_conversation_ids,
                is_attachment=parsed_filters.has_attachment,
                file_types=(
                    list(parsed_filters.file_types)
                    if parsed_filters.file_types
                    else None
                ),  # P1 Fix
            )

            # 4. Vector search (if we have an embedding)
            vector_results = []
            if query_embedding is not None:
                logger.info(f"Query embedding generated. Dim: {len(query_embedding)}")
                hnsw_ef_search = getattr(config.search, "hnsw_ef_search", None)
                vector_results = search_chunks_vector(
                    session,
                    query_embedding,
                    args.tenant_id,
                    limit=k * candidates_multiplier,
                    ef_search=hnsw_ef_search,
                    conversation_ids=final_conversation_ids,
                    is_attachment=parsed_filters.has_attachment,
                    file_types=(
                        list(parsed_filters.file_types)
                        if parsed_filters.file_types
                        else None
                    ),  # P1 Fix
                )
                logger.info(f"Vector search returned {len(vector_results)} chunks")
            else:
                logger.warning("Query embedding is None, skipping vector search")

            # Convert to SearchResultItem format
            fts_items = _convert_fts_to_items(fts_chunk_results)
            vector_items = _convert_vector_to_items(vector_results)

            # 5. Deduplication by content_hash
            fts_deduped = deduplicate_by_hash(fts_items)
            vector_deduped = deduplicate_by_hash(vector_items)

            # 6. Fusion (configurable)
            if fusion_method == "weighted_sum":
                fused_results = fuse_weighted_sum(
                    fts_deduped, vector_deduped, alpha=config.search.rerank_alpha
                )
            else:
                fused_results = fuse_rrf(fts_deduped, vector_deduped, k=RRF_K_DEFAULT)

            # 6.5. Apply Summary Boost
            if summary_boost_ids:
                for item in fused_results:
                    if item.conversation_id in summary_boost_ids:
                        # Boost score by 20%
                        item.score *= SUMMARY_BOOST_FACTOR
                        item.fusion_score = item.score

                # Re-sort after boosting
                fused_results = sorted(
                    fused_results, key=lambda r: r.score, reverse=True
                )

            # 7. Reranking (External vs Lightweight)
            if config.search.reranker_endpoint:
                # Use Qwen/External Cross-Encoder
                fused_results = await call_external_reranker(
                    config.search.reranker_endpoint,
                    args.query,
                    fused_results,
                    top_n=50,  # Rerank top 50 candidates
                )
            else:
                # Lightweight blending
                fused_results = rerank_results(
                    fused_results, alpha=config.search.rerank_alpha
                )

            # 8. Get conversation timestamps for recency boost
            conversation_ids = list(
                {r.conversation_id for r in fused_results if r.conversation_id}
            )
            conversation_updated_at = _get_conversation_timestamps(
                session, conversation_ids, args.tenant_id
            )

            # 9. Apply recency boost
            fused_results = apply_recency_boost(
                fused_results,
                conversation_updated_at,
                half_life_days=config.search.half_life_days,
                boost_strength=config.search.recency_boost_strength,
            )

            # 10. Down-weight quoted_history (do this before diversification)
            fused_results = downweight_quoted_history(
                fused_results, factor=QUOTED_HISTORY_DOWNWEIGHT_FACTOR
            )
            fused_results = sorted(fused_results, key=lambda r: r.score, reverse=True)

            # 11. MMR diversity for the final top-k list
            fused_results = apply_mmr(
                fused_results,
                lambda_param=config.search.mmr_lambda,
                limit=min(len(fused_results), k),
            )

            # 12. Post-filtering for fields not handled in-query (e.g., participants)
            final_results = fused_results

            reranker_label = f"{fusion_method}|alpha={config.search.rerank_alpha:.2f}|mmr={config.search.mmr_lambda:.2f}"
            return Ok(
                SearchResults(
                    query=args.query,
                    reranker=reranker_label,
                    results=final_results[:k],
                )
            )
    except Exception as e:
        logger.exception("Hybrid search failed")
        return Err(
            RetrievalError(
                f"Hybrid search failed: {e}",
                query=args.query,
                context={"original_exception": str(e)},
            )
        )


def _get_conversation_timestamps(
    session, conversation_ids: List[str], tenant_id: str
) -> Dict[str, datetime]:
    """Fetch updated_at timestamps for conversations."""
    if not conversation_ids:
        return {}

    from sqlalchemy import text

    stmt = text(
        """
        SELECT conversation_id, updated_at
        FROM conversations
        WHERE conversation_id = ANY(CAST(:conversation_ids AS UUID[]))
        AND tenant_id = :tenant_id
    """
    )

    try:
        results = session.execute(
            stmt, {"conversation_ids": conversation_ids, "tenant_id": tenant_id}
        ).fetchall()
        return {str(row.conversation_id): row.updated_at for row in results}
    except Exception as e:
        logger.warning("Failed to fetch conversation timestamps: %s", e)
        return {}
