import logging
from typing import Any, List, Optional

import numpy as np
from cortex.embeddings.client import EmbeddingsClient
from cortex.retrieval.cache import cache_query_embedding, get_cached_query_embedding
from cortex.retrieval.filter_resolution import _resolve_filter_conversation_ids
from cortex.retrieval.results import SearchResultItem

logger = logging.getLogger(__name__)
_embedding_client = EmbeddingsClient()


def _get_query_embedding(query: str, config: Any) -> Optional[List[float]]:
    """Helper: Get query embedding with cache fallback."""
    try:
        # Check cache first
        cached = get_cached_query_embedding(query, model=config.embedding.model_name)
        if cached is not None:
            logger.debug("Using cached query embedding for: %s", query[:50])
            return cached.tolist()

        # Call client
        embedding = _embedding_client.embed(query)
        # Cache the embedding
        cache_query_embedding(
            query,
            np.asarray(embedding, dtype=np.float32),
            model=config.embedding.model_name,
        )
        return embedding
    except Exception as e:
        logger.error(f"Failed to embed query (falling back to FTS only): {e}")
        return None


def _resolve_target_conversations(
    session, args: Any, search_query: str, limit: int, parsed_filters: Any
) -> Optional[List[str]]:
    """Helper: Resolve specific conversation IDs from navigation or filters."""
    from cortex.retrieval.fts_search import search_messages_fts

    # 2. Optional navigational narrowing
    nav_conversation_ids: Optional[List[str]] = None
    if args.classification and args.classification.type == "navigational":
        nav_hits = search_messages_fts(
            session,
            search_query,
            args.tenant_id,
            limit=min(limit, 200),
        )
        nav_conversation_ids = (
            list({h.conversation_id for h in nav_hits if h.conversation_id}) or None
        )

    # 2.5 Resolve Rich Filters
    filter_resolved_ids = _resolve_filter_conversation_ids(
        session, parsed_filters, args.tenant_id
    )

    # Merge
    final_ids: Optional[List[str]] = None
    if nav_conversation_ids:
        final_ids = nav_conversation_ids

    if filter_resolved_ids is not None:
        if final_ids is not None:
            # Intersect
            final_ids = list(set(final_ids) & set(filter_resolved_ids)) or None
        else:
            final_ids = filter_resolved_ids

    # If intersection is empty but filters existed, return empty list (no match)
    if (
        nav_conversation_ids is not None or filter_resolved_ids is not None
    ) and not final_ids:
        return []

    return final_ids


def _convert_fts_to_items(fts_results: List[Any]) -> List[SearchResultItem]:
    """Helper: Convert FTS DB results to SearchResultItem."""
    items = []
    for res in fts_results:
        metadata = dict(res.metadata or {})
        metadata.setdefault("chunk_type", res.chunk_type)
        items.append(
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
    return items


def _convert_vector_to_items(vector_results: List[Any]) -> List[SearchResultItem]:
    """Helper: Convert Vector DB results to SearchResultItem."""
    items = []
    for res in vector_results:
        metadata = dict(res.metadata or {})
        if res.chunk_type:
            metadata.setdefault("chunk_type", res.chunk_type)
        items.append(
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
    return items
