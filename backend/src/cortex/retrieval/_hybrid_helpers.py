import asyncio
import logging
from typing import Any

import numpy as np
from cortex.llm.runtime import LLMRuntime
from cortex.retrieval.async_cache import (
    cache_query_embedding,
    get_cached_query_embedding,
)
from cortex.retrieval.filter_resolution import _resolve_filter_conversation_ids
from cortex.retrieval.fts_search import ChunkFTSResult
from cortex.retrieval.results import SearchResultItem
from cortex.retrieval.vector_store import VectorResult

logger = logging.getLogger(__name__)
_llm_runtime: LLMRuntime | None = None
_runtime_lock: asyncio.Lock | None = None
_runtime_lock_loop: asyncio.AbstractEventLoop | None = None

MAX_NAV_LIMIT = 200
SNIPPET_LENGTH = 200


def _get_runtime_lock() -> asyncio.Lock:
    global _runtime_lock
    global _runtime_lock_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if _runtime_lock is None or (loop is not None and loop is not _runtime_lock_loop):
        _runtime_lock = asyncio.Lock()
        _runtime_lock_loop = loop

    return _runtime_lock


async def _get_runtime() -> LLMRuntime:
    """Get or create LLMRuntime singleton."""
    global _llm_runtime
    if _llm_runtime is None:
        async with _get_runtime_lock():
            if _llm_runtime is None:
                _llm_runtime = LLMRuntime()
    return _llm_runtime


async def _get_query_embedding(query: str, config: Any) -> np.ndarray | None:
    """Helper: Get query embedding with cache fallback."""
    try:
        embedding_config = getattr(config, "embedding", None)
        model_name = getattr(embedding_config, "model_name", None)
        if not model_name:
            logger.warning("Embedding model not configured; skipping vector search")
            return None

        # Check cache first
        cached = await get_cached_query_embedding(query, model=model_name)
        if cached is not None:
            logger.debug("Using cached query embedding for: %s", query[:50])
            return cached

        # Use LLMRuntime which has CPU fallback logic
        runtime = await _get_runtime()
        embedding_array = await asyncio.to_thread(runtime.embed_queries, [query])
        if embedding_array is None or len(embedding_array) == 0:
            logger.warning("Embedding runtime returned empty result")
            return None
        embedding = embedding_array[0]

        # Cache the embedding
        await cache_query_embedding(
            query,
            embedding,
            model=model_name,
        )
        return embedding
    except Exception:
        logger.exception("Failed to embed query; falling back to FTS only")
        return None


def _resolve_target_conversations(
    session, args: Any, search_query: str, limit: int, parsed_filters: Any
) -> list[str] | None:
    """Helper: Resolve specific conversation IDs from navigation or filters."""
    from cortex.retrieval.fts_search import search_messages_fts

    # 2. Optional navigational narrowing
    nav_conversation_ids: list[str] | None = None
    classification = getattr(args, "classification", None)
    classification_type = (
        getattr(classification, "type", None) if classification else None
    )
    tenant_id = getattr(args, "tenant_id", None) or "default"
    limit_value = limit if isinstance(limit, int) else MAX_NAV_LIMIT
    safe_limit = max(0, min(limit_value, MAX_NAV_LIMIT))

    if classification_type == "navigational":
        nav_hits = search_messages_fts(
            session,
            search_query,
            tenant_id,
            limit=safe_limit,
        )
        nav_conversation_ids = list(
            {
                str(h.conversation_id)
                for h in nav_hits
                if getattr(h, "conversation_id", None)
            }
        )

    # 2.5 Resolve Rich Filters
    filter_resolved_ids = _resolve_filter_conversation_ids(
        session, parsed_filters, tenant_id
    )

    # Merge
    final_ids: list[str] | None = None
    if nav_conversation_ids is not None:
        final_ids = nav_conversation_ids

    if filter_resolved_ids is not None:
        if final_ids is not None:
            # Intersect
            final_ids = list(set(final_ids) & set(filter_resolved_ids))
        else:
            final_ids = filter_resolved_ids

    # If intersection is empty but filters existed, return empty list (no match)
    if (
        nav_conversation_ids is not None or filter_resolved_ids is not None
    ) and final_ids is not None:
        if len(final_ids) == 0:
            return []

    if (
        nav_conversation_ids is not None or filter_resolved_ids is not None
    ) and final_ids is None:
        return []

    return final_ids if final_ids is not None else None


def _convert_fts_to_items(
    fts_results: list[ChunkFTSResult],
) -> list[SearchResultItem]:
    """Helper: Convert FTS DB results to SearchResultItem."""
    items = []
    for res in fts_results:
        metadata = dict(res.metadata or {})
        if res.chunk_type:
            metadata.setdefault("chunk_type", res.chunk_type)
        snippet = res.snippet or ""
        if not isinstance(snippet, str):
            snippet = str(snippet)
        content = res.text or ""
        if not isinstance(content, str):
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            else:
                content = str(content)
        highlights = [snippet] if snippet else []
        items.append(
            SearchResultItem(
                chunk_id=res.chunk_id,
                score=res.score,
                conversation_id=res.conversation_id,
                attachment_id=res.attachment_id,
                is_attachment=(
                    bool(res.is_attachment) if res.is_attachment is not None else False
                ),
                highlights=highlights,
                snippet=snippet,
                content=content,
                source=metadata.get("source"),
                metadata=metadata,
                lexical_score=res.score,
                vector_score=0.0,
                content_hash=metadata.get("content_hash"),
            )
        )
    return items


def _convert_vector_to_items(
    vector_results: list[VectorResult],
) -> list[SearchResultItem]:
    """Helper: Convert Vector DB results to SearchResultItem."""
    items = []
    for res in vector_results:
        metadata = dict(res.metadata or {})
        if res.chunk_type:
            metadata.setdefault("chunk_type", res.chunk_type)
        content = res.text or ""
        if not isinstance(content, str):
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            else:
                content = str(content)
        snippet = content[:SNIPPET_LENGTH] if content else ""
        highlights = [snippet] if snippet else []
        items.append(
            SearchResultItem(
                chunk_id=res.chunk_id,
                score=res.score,
                conversation_id=res.conversation_id,
                attachment_id=res.attachment_id,
                is_attachment=res.is_attachment,
                highlights=highlights,
                snippet=snippet,
                content=content,
                metadata=metadata,
                lexical_score=0.0,
                vector_score=res.score,
                content_hash=metadata.get("content_hash"),
            )
        )
    return items
