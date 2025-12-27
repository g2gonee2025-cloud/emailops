"""
Query Embedding Cache.

Implements thread-safe, TTL-based caching for query embeddings
to avoid redundant API calls for repeated queries.

Ported from reference code feature_search_draft.py.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, Optional, Tuple

import numpy as np

# Constants
LOG_QUERY_TRUNCATE_LEN = 50


logger = logging.getLogger(__name__)

# Default TTL: 5 minutes (matches reference code)
DEFAULT_CACHE_TTL_SECONDS = 300.0
DEFAULT_MAX_CACHE_SIZE = 100


class QueryEmbeddingCache:
    """
    Async-safe LRU cache for query embeddings with time-to-idle expiration.

    Features:
    - Time-to-idle expiration (default 5 minutes): TTL resets on access.
    - LRU eviction when max size exceeded
    - Async-safe access via lock
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        max_size: int = DEFAULT_MAX_CACHE_SIZE,
    ):
        # Use OrderedDict for O(1) LRU
        self._cache: OrderedDict[str, tuple[float, np.ndarray]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._ttl = ttl_seconds
        self._max_size = max_size

    def _make_key(self, query: str, model: str = "") -> str:
        """Create cache key from query and model name."""
        return f"{model}::{query}"

    async def get(self, query: str, model: str = "") -> np.ndarray | None:
        """
        Get cached embedding if available and not expired.

        Returns None if not found or expired.
        """
        key = self._make_key(query, model)
        async with self._lock:
            if key not in self._cache:
                return None

            timestamp, embedding = self._cache[key]
            if (time.time() - timestamp) >= self._ttl:
                # Expired - remove and return None
                self._cache.pop(key)
                logger.debug(
                    "Cache miss (expired): query=%s", query[:LOG_QUERY_TRUNCATE_LEN]
                )
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            # Update timestamp to refresh TTL on access
            self._cache[key] = (time.time(), embedding)
            logger.debug("Cache hit: query=%s", query[:LOG_QUERY_TRUNCATE_LEN])
            return embedding.copy()

    async def put(self, query: str, embedding: np.ndarray, model: str = "") -> None:
        """
        Cache an embedding with current timestamp.

        If max size exceeded, evicts oldest entry.
        """
        key = self._make_key(query, model)
        async with self._lock:
            # If exists, move to end
            if key in self._cache:
                self._cache.move_to_end(key)

            # Store (or update)
            self._cache[key] = (time.time(), embedding.copy())

            # LRU eviction if over max size
            while len(self._cache) > self._max_size:
                # popitem(last=False) removes the first (oldest) item
                self._cache.popitem(last=False)
                logger.debug(
                    "Cache eviction: removed oldest entry, %d remaining",
                    len(self._cache),
                )

    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            self._cache.clear()

    async def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        async with self._lock:
            now = time.time()
            valid_count = sum(
                1 for ts, _ in self._cache.values() if (now - ts) < self._ttl
            )
            return {
                "total_entries": len(self._cache),
                "valid_entries": valid_count,
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
            }


# Global singleton instance
_query_cache: QueryEmbeddingCache | None = None
_cache_lock = asyncio.Lock()


async def get_query_cache() -> QueryEmbeddingCache:
    """Get or create the global query embedding cache."""

    global _query_cache
    if _query_cache is None:
        async with _cache_lock:
            if _query_cache is None:
                _query_cache = QueryEmbeddingCache()
    return _query_cache


async def get_cached_query_embedding(
    query: str,
    model: str = "",
) -> np.ndarray | None:
    """Convenience function to get cached embedding."""
    cache = await get_query_cache()
    return await cache.get(query, model)


async def cache_query_embedding(
    query: str,
    embedding: np.ndarray,
    model: str = "",
) -> None:
    """Convenience function to cache an embedding."""
    cache = await get_query_cache()
    await cache.put(query, embedding, model)
