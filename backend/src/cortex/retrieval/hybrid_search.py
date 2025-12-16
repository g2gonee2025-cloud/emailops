"""
Hybrid Search (FTS + Vector + RRF).

Implements §8.3 of the Canonical Blueprint.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from cortex.embeddings.client import EmbeddingsClient

# Import QueryClassification from dedicated module per Blueprint §8.2
from cortex.retrieval.query_classifier import QueryClassification
from pydantic import BaseModel, Field

_embedding_client = EmbeddingsClient()


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

    Blueprint §8.4 (adapted for Conversation schema):
    * chunk_id: Optional[UUID]
    * score: float
    * conversation_id: str (was thread_id)
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
    conversation_id: str = ""  # Primary key for Conversation schema
    attachment_id: Optional[str] = None
    is_attachment: bool = False
    highlights: List[str] = Field(default_factory=list)
    snippet: str = ""
    content: Optional[str] = None
    source: Optional[str] = None
    filename: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    lexical_score: Optional[float] = None
    vector_score: Optional[float] = None
    fusion_score: Optional[float] = None
    rerank_score: Optional[float] = None
    content_hash: Optional[str] = None

    # Backward compatibility aliases
    @property
    def thread_id(self) -> str:
        """Alias for conversation_id (backward compat)."""
        return self.conversation_id

    @property
    def message_id(self) -> str:
        """Backward compat - empty since we use conversation model."""
        return ""


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
    boost_strength: float = 1.0,
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
    results: List[SearchResultItem], factor: float = 0.7
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

    # Prefer already-populated structured identifiers, but fill blanks.
    if not into.conversation_id and other.conversation_id:
        into.conversation_id = other.conversation_id
    if not into.attachment_id and other.attachment_id:
        into.attachment_id = other.attachment_id

    # Preserve the richer text/snippet if one side is missing.
    if not into.content and other.content:
        into.content = other.content
    if not into.snippet and other.snippet:
        into.snippet = other.snippet

    # Merge highlight fragments.
    if other.highlights:
        seen = set(into.highlights or [])
        for h in other.highlights:
            if h and h not in seen:
                into.highlights.append(h)
                seen.add(h)

    # Merge score components.
    if (into.lexical_score or 0.0) == 0.0 and (other.lexical_score or 0.0) > 0.0:
        into.lexical_score = other.lexical_score
    if (into.vector_score or 0.0) == 0.0 and (other.vector_score or 0.0) > 0.0:
        into.vector_score = other.vector_score

    # Merge metadata (keep existing keys unless missing)
    for k, v in (other.metadata or {}).items():
        into.metadata.setdefault(k, v)

    # Prefer explicit content_hash if present
    if not into.content_hash and other.content_hash:
        into.content_hash = other.content_hash

    return into


def fuse_rrf(
    fts_results: List[SearchResultItem],
    vector_results: List[SearchResultItem],
    k: int = 60,
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
        if key in items:
            items[key] = _merge_result_fields(items[key], item)
        else:
            items[key] = item

    # Score from vector ranking
    for rank, item in enumerate(vector_results, start=1):
        key = item.chunk_id or item.message_id
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
    alpha: float = 0.5,
) -> List[SearchResultItem]:
    """Fuse rankings via weighted sum.

    We assume lexical and vector signals are already scaled to roughly [0, 1]
    (ts_rank_cd(..., 32) and cosine-similarity-to-[0,1] mapping). That makes
    fusion stable without ad-hoc max-normalization.
    """

    combined: Dict[str, SearchResultItem] = {}

    for item in fts_results:
        key = item.chunk_id or item.message_id
        if key in combined:
            combined[key] = _merge_result_fields(combined[key], item)
        else:
            combined[key] = item

    for item in vector_results:
        key = item.chunk_id or item.message_id
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


def rerank_results(
    results: List[SearchResultItem],
    alpha: float,
) -> List[SearchResultItem]:
    """Apply lightweight reranker blending lexical/vector evidence."""

    if not results:
        return results

    max_lex = max((item.lexical_score or 0.0 for item in results), default=0.0)
    max_vec = max((item.vector_score or 0.0 for item in results), default=0.0)

    for item in results:
        lex_component = (item.lexical_score or 0.0) / max_lex if max_lex > 0 else 0.0
        vec_component = (item.vector_score or 0.0) / max_vec if max_vec > 0 else 0.0
        rerank_score = alpha * vec_component + (1 - alpha) * lex_component
        item.rerank_score = rerank_score
        item.score = 0.5 * item.score + 0.5 * rerank_score
        item.metadata["rerank_score"] = rerank_score

    return sorted(results, key=lambda itm: itm.score, reverse=True)


def _tokenize_for_similarity(text: str) -> set[str]:
    tokens = [t.lower() for t in text.split() if t]
    return set(tokens)


def _text_similarity(a: SearchResultItem, b: SearchResultItem) -> float:
    """Compute simple Jaccard similarity between two result snippets/contents."""
    text_a = a.content or a.snippet or ""
    text_b = b.content or b.snippet or ""
    if not text_a or not text_b:
        return 0.0
    tokens_a = _tokenize_for_similarity(text_a)
    tokens_b = _tokenize_for_similarity(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union else 0.0


def apply_mmr(
    results: List[SearchResultItem],
    lambda_param: float,
    limit: Optional[int] = None,
) -> List[SearchResultItem]:
    """Apply Maximal Marginal Relevance to diversify top results."""

    if not results:
        return results

    limit = limit or len(results)
    selected: List[SearchResultItem] = []
    candidates = results.copy()

    while candidates and len(selected) < limit:
        best_idx = 0
        best_score = float("-inf")
        for idx, candidate in enumerate(candidates):
            relevance = candidate.score
            if selected:
                redundancy = max(
                    _text_similarity(candidate, chosen) for chosen in selected
                )
            else:
                redundancy = 0.0
            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        selected.append(candidates.pop(best_idx))

    # Append any remaining candidates to preserve ordering info
    selected.extend(candidates)
    return selected


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
    from cortex.config.loader import get_config
    from cortex.db.session import SessionLocal
    from cortex.retrieval.fts_search import search_chunks_fts, search_messages_fts
    from cortex.retrieval.vector_search import search_chunks_vector

    config = get_config()
    k = args.k or config.search.k
    fusion_method = args.fusion_method or config.search.fusion_strategy

    candidates_multiplier = config.search.candidates_multiplier

    # 1. Embed query for vector search
    # Best-effort: if the embedding service is temporarily unavailable, we still
    # want FTS to keep the product functional.
    query_embedding: Optional[List[float]] = None
    try:
        query_embedding = _embedding_client.embed(args.query)
    except Exception as e:
        logger.error(f"Failed to embed query (falling back to FTS only): {e}")

    with SessionLocal() as session:
        from cortex.db.session import set_session_tenant

        set_session_tenant(session, args.tenant_id)

        # 2. Optional navigational narrowing
        # When users are clearly looking for a specific email/thread, message-level
        # FTS is a great cheap filter. Since you always have embedded chunks, we
        # then run chunk retrieval inside those threads.
        nav_conversation_ids: Optional[List[str]] = None
        if args.classification and args.classification.type == "navigational":
            nav_hits = search_messages_fts(
                session,
                args.query,
                args.tenant_id,
                limit=min(k * candidates_multiplier, 200),
            )
            nav_conversation_ids = (
                list({h.conversation_id for h in nav_hits if h.conversation_id}) or None
            )

        # 3. Lexical search (FTS) on chunks
        fts_chunk_results = search_chunks_fts(
            session,
            args.query,
            args.tenant_id,
            limit=k * candidates_multiplier,
            conversation_ids=nav_conversation_ids,
        )

        # 4. Vector search (if we have an embedding)
        vector_results = []
        if query_embedding is not None:
            hnsw_ef_search = getattr(config.search, "hnsw_ef_search", None)
            vector_results = search_chunks_vector(
                session,
                query_embedding,
                args.tenant_id,
                limit=k * candidates_multiplier,
                ef_search=hnsw_ef_search,
                conversation_ids=nav_conversation_ids,
            )

        # Convert to SearchResultItem format
        fts_items = []
        for res in fts_chunk_results:
            metadata = dict(res.metadata or {})
            metadata.setdefault("chunk_type", res.chunk_type)
            fts_items.append(
                SearchResultItem(
                    chunk_id=res.chunk_id,
                    score=res.score,
                    conversation_id=res.conversation_id,
                    attachment_id=res.attachment_id,
                    is_attachment=res.is_attachment,
                    highlights=[res.snippet],
                    snippet=res.snippet,
                    content=res.text,
                    source=metadata.get("source"),
                    metadata=metadata,
                    lexical_score=res.score,
                    vector_score=0.0,
                    content_hash=metadata.get("content_hash"),
                )
            )

        vector_items = []
        for res in vector_results:
            metadata = dict(res.metadata or {})
            if res.chunk_type:
                metadata.setdefault("chunk_type", res.chunk_type)
            vector_items.append(
                SearchResultItem(
                    chunk_id=res.chunk_id,
                    score=res.score,
                    conversation_id=res.conversation_id,
                    attachment_id=res.attachment_id or None,
                    is_attachment=res.is_attachment,
                    highlights=[res.text[:200]] if res.text else [],
                    snippet=res.text[:200] if res.text else "",
                    content=res.text,
                    metadata=metadata,
                    lexical_score=0.0,
                    vector_score=res.score,
                    content_hash=metadata.get("content_hash"),
                )
            )

        # 5. Deduplication by content_hash
        fts_deduped = deduplicate_by_hash(fts_items)
        vector_deduped = deduplicate_by_hash(vector_items)

        # 6. Fusion (configurable)
        if fusion_method == "weighted_sum":
            fused_results = fuse_weighted_sum(
                fts_deduped, vector_deduped, alpha=config.search.rerank_alpha
            )
        else:
            fused_results = fuse_rrf(fts_deduped, vector_deduped, k=60)

        # 7. Lightweight rerank blending lexical/vector signals
        fused_results = rerank_results(fused_results, alpha=config.search.rerank_alpha)

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
        fused_results = downweight_quoted_history(fused_results, factor=0.7)
        fused_results = sorted(fused_results, key=lambda r: r.score, reverse=True)

        # 11. MMR diversity for the final top-k list
        fused_results = apply_mmr(
            fused_results,
            lambda_param=config.search.mmr_lambda,
            limit=min(len(fused_results), k),
        )

        reranker_label = f"{fusion_method}|alpha={config.search.rerank_alpha:.2f}|mmr={config.search.mmr_lambda:.2f}"
        return SearchResults(
            query=args.query, reranker=reranker_label, results=fused_results[:k]
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

    results = session.execute(
        stmt, {"conversation_ids": conversation_ids, "tenant_id": tenant_id}
    ).fetchall()

    return {str(row.conversation_id): row.updated_at for row in results}


# Backward compat alias
_get_thread_timestamps = _get_conversation_timestamps
