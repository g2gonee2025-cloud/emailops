"""
Unit tests for Query Embedding Cache.
"""

import asyncio
import time

import numpy as np
import pytest
from cortex.retrieval.async_cache import (
    QueryEmbeddingCache,
    cache_query_embedding,
    get_cached_query_embedding,
    get_query_cache,
)


@pytest.mark.asyncio
class TestQueryEmbeddingCache:
    """Tests for QueryEmbeddingCache class."""

    async def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        cache = QueryEmbeddingCache(ttl_seconds=60.0, max_size=10)
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        await cache.put("test query", embedding, model="test-model")
        result = await cache.get("test query", model="test-model")

        assert result is not None
        np.testing.assert_array_equal(result, embedding)

    async def test_cache_miss_when_not_present(self):
        """Test that get returns None for missing keys."""
        cache = QueryEmbeddingCache(ttl_seconds=60.0, max_size=10)

        result = await cache.get("nonexistent query", model="test-model")

        assert result is None

    async def test_cache_expiration(self):
        """Test that entries expire after TTL."""
        cache = QueryEmbeddingCache(ttl_seconds=0.1, max_size=10)  # 100ms TTL
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        await cache.put("test query", embedding)

        # Should be found immediately
        assert await cache.get("test query") is not None

        # Wait for expiration
        await asyncio.sleep(0.15)

        # Should be expired now
        assert await cache.get("test query") is None

    async def test_lru_eviction(self):
        """Test that oldest entries are evicted when max size exceeded."""
        cache = QueryEmbeddingCache(ttl_seconds=60.0, max_size=3)

        for i in range(5):
            embedding = np.array([float(i)], dtype=np.float32)
            await cache.put(f"query_{i}", embedding)

        # Oldest entries should be evicted
        assert await cache.get("query_0") is None
        assert await cache.get("query_1") is None

        # Newest entries should still be present
        assert await cache.get("query_3") is not None
        assert await cache.get("query_4") is not None

    async def test_cache_returns_copy(self):
        """Test that get returns a copy to prevent mutation."""
        cache = QueryEmbeddingCache(ttl_seconds=60.0, max_size=10)
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        await cache.put("test query", embedding)
        result = await cache.get("test query")

        # Mutate the returned array
        result[0] = 999.0

        # Original cached value should be unchanged
        cached = await cache.get("test query")
        assert cached[0] == pytest.approx(1.0)

    async def test_different_models_are_separate_keys(self):
        """Test that same query with different models are cached separately."""
        cache = QueryEmbeddingCache(ttl_seconds=60.0, max_size=10)
        emb1 = np.array([1.0], dtype=np.float32)
        emb2 = np.array([2.0], dtype=np.float32)

        await cache.put("same query", emb1, model="model-a")
        await cache.put("same query", emb2, model="model-b")

        result_a = await cache.get("same query", model="model-a")
        result_b = await cache.get("same query", model="model-b")

        assert result_a[0] == pytest.approx(1.0)
        assert result_b[0] == pytest.approx(2.0)

    async def test_stats(self):
        """Test cache statistics."""
        cache = QueryEmbeddingCache(ttl_seconds=60.0, max_size=10)
        embedding = np.array([1.0], dtype=np.float32)

        await cache.put("query1", embedding)
        await cache.put("query2", embedding)

        stats = await cache.stats()

        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 2
        assert stats["max_size"] == 10
        assert stats["ttl_seconds"] == pytest.approx(60.0)

    async def test_clear(self):
        """Test cache clear operation."""
        cache = QueryEmbeddingCache(ttl_seconds=60.0, max_size=10)
        embedding = np.array([1.0], dtype=np.float32)

        await cache.put("query1", embedding)
        await cache.put("query2", embedding)
        await cache.clear()

        assert await cache.get("query1") is None
        assert await cache.get("query2") is None


@pytest.mark.asyncio
class TestGlobalCacheFunctions:
    """Tests for global cache convenience functions."""

    async def test_global_cache_singleton(self):
        """Test that get_query_cache returns a singleton."""
        cache1 = await get_query_cache()
        cache2 = await get_query_cache()

        assert cache1 is cache2

    async def test_convenience_functions(self):
        """Test cache_query_embedding and get_cached_query_embedding."""
        # Clear any existing cache entries
        cache = await get_query_cache()
        await cache.clear()

        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Initially not cached
        assert (
            await get_cached_query_embedding("unique_test_query_123", "test-model")
            is None
        )

        # Cache it
        await cache_query_embedding("unique_test_query_123", embedding, "test-model")

        # Now should be found
        result = await get_cached_query_embedding("unique_test_query_123", "test-model")
        assert result is not None
        np.testing.assert_array_equal(result, embedding)
