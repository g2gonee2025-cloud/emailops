"""
Query Embedding Cache.

Implements thread-safe, TTL-based caching for query embeddings
to avoid redundant API calls for repeated queries.

Ported from reference code feature_search_draft.py.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default TTL: 5 minutes (matches reference code)
DEFAULT_CACHE_TTL_SECONDS = 300.0
DEFAULT_MAX_CACHE_SIZE = 100


class QueryEmbeddingCache:
    """
    Thread-safe LRU cache for query embeddings with TTL expiration.

    Features:
    - TTL-based expiration (default 5 minutes)
    - LRU eviction when max size exceeded
    - Thread-safe access via lock
    - Automatic cleanup of expired entries
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        max_size: int = DEFAULT_MAX_CACHE_SIZE,
    ):
        self._cache: dict[str, Tuple[float, np.ndarray]] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._max_size = max_size

    def _make_key(self, query: str, model: str = "") -> str:
        """Create cache key from query and model name."""
        return f"{model}::{query}"

    def get(self, query: str, model: str = "") -> Optional[np.ndarray]:
        """
        Get cached embedding if available and not expired.

        Returns None if not found or expired.
        """
        key = self._make_key(query, model)
        with self._lock:
            if key not in self._cache:
                return None

            timestamp, embedding = self._cache[key]
            if (time.time() - timestamp) >= self._ttl:
                # Expired - remove and return None
                del self._cache[key]
                logger.debug("Cache miss (expired): query=%s", query[:50])
                return None

            logger.debug("Cache hit: query=%s", query[:50])
            return embedding.copy()  # Return copy to prevent mutation

    def put(self, query: str, embedding: np.ndarray, model: str = "") -> None:
        """
        Cache an embedding with current timestamp.

        If max size exceeded, evicts oldest entry.
        """
        key = self._make_key(query, model)
        with self._lock:
            # Store copy to prevent external mutation
            self._cache[key] = (time.time(), embedding.copy())

            # LRU eviction if over max size
            if len(self._cache) > self._max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
                logger.debug(
                    "Cache eviction: removed oldest entry, %d remaining",
                    len(self._cache),
                )

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
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
_query_cache: Optional[QueryEmbeddingCache] = None
_cache_lock = threading.Lock()


def get_query_cache() -> QueryEmbeddingCache:
    """Get or create the global query embedding cache."""
    global _query_cache
    if _query_cache is None:
        with _cache_lock:
            if _query_cache is None:
                _query_cache = QueryEmbeddingCache()
    return _query_cache


def get_cached_query_embedding(
    query: str,
    model: str = "",
) -> Optional[np.ndarray]:
    """Convenience function to get cached embedding."""
    return get_query_cache().get(query, model)


def cache_query_embedding(
    query: str,
    embedding: np.ndarray,
    model: str = "",
) -> None:
    """Convenience function to cache an embedding."""
    get_query_cache().put(query, embedding, model)
