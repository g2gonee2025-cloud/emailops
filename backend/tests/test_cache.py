"""
Unit tests for Query Embedding Cache.
"""
import time

import numpy as np
from cortex.retrieval.cache import (
    QueryEmbeddingCache,
    cache_query_embedding,
    get_cached_query_embedding,
    get_query_cache,
)


class TestQueryEmbeddingCache:
    """Tests for QueryEmbeddingCache class."""

    def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        cache = QueryEmbeddingCache(ttl_seconds=60.0, max_size=10)
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        cache.put("test query", embedding, model="test-model")
        result = cache.get("test query", model="test-model")

        assert result is not None
        np.testing.assert_array_equal(result, embedding)

    def test_cache_miss_when_not_present(self):
        """Test that get returns None for missing keys."""
        cache = QueryEmbeddingCache(ttl_seconds=60.0, max_size=10)

        result = cache.get("nonexistent query", model="test-model")

        assert result is None

    def test_cache_expiration(self):
        """Test that entries expire after TTL."""
        cache = QueryEmbeddingCache(ttl_seconds=0.1, max_size=10)  # 100ms TTL
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        cache.put("test query", embedding)

        # Should be found immediately
        assert cache.get("test query") is not None

        # Wait for expiration
        time.sleep(0.15)

        # Should be expired now
        assert cache.get("test query") is None

    def test_lru_eviction(self):
        """Test that oldest entries are evicted when max size exceeded."""
        cache = QueryEmbeddingCache(ttl_seconds=60.0, max_size=3)

        for i in range(5):
            embedding = np.array([float(i)], dtype=np.float32)
            cache.put(f"query_{i}", embedding)

        # Oldest entries should be evicted
        assert cache.get("query_0") is None
        assert cache.get("query_1") is None

        # Newest entries should still be present
        assert cache.get("query_3") is not None
        assert cache.get("query_4") is not None

    def test_cache_returns_copy(self):
        """Test that get returns a copy to prevent mutation."""
        cache = QueryEmbeddingCache(ttl_seconds=60.0, max_size=10)
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        cache.put("test query", embedding)
        result = cache.get("test query")

        # Mutate the returned array
        result[0] = 999.0

        # Original cached value should be unchanged
        cached = cache.get("test query")
        assert cached[0] == 1.0

    def test_different_models_are_separate_keys(self):
        """Test that same query with different models are cached separately."""
        cache = QueryEmbeddingCache(ttl_seconds=60.0, max_size=10)
        emb1 = np.array([1.0], dtype=np.float32)
        emb2 = np.array([2.0], dtype=np.float32)

        cache.put("same query", emb1, model="model-a")
        cache.put("same query", emb2, model="model-b")

        result_a = cache.get("same query", model="model-a")
        result_b = cache.get("same query", model="model-b")

        assert result_a[0] == 1.0
        assert result_b[0] == 2.0

    def test_stats(self):
        """Test cache statistics."""
        cache = QueryEmbeddingCache(ttl_seconds=60.0, max_size=10)
        embedding = np.array([1.0], dtype=np.float32)

        cache.put("query1", embedding)
        cache.put("query2", embedding)

        stats = cache.stats()

        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 2
        assert stats["max_size"] == 10
        assert stats["ttl_seconds"] == 60.0

    def test_clear(self):
        """Test cache clear operation."""
        cache = QueryEmbeddingCache(ttl_seconds=60.0, max_size=10)
        embedding = np.array([1.0], dtype=np.float32)

        cache.put("query1", embedding)
        cache.put("query2", embedding)
        cache.clear()

        assert cache.get("query1") is None
        assert cache.get("query2") is None


class TestGlobalCacheFunctions:
    """Tests for global cache convenience functions."""

    def test_global_cache_singleton(self):
        """Test that get_query_cache returns a singleton."""
        cache1 = get_query_cache()
        cache2 = get_query_cache()

        assert cache1 is cache2

    def test_convenience_functions(self):
        """Test cache_query_embedding and get_cached_query_embedding."""
        # Clear any existing cache entries
        get_query_cache().clear()

        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Initially not cached
        assert get_cached_query_embedding("unique_test_query_123", "test-model") is None

        # Cache it
        cache_query_embedding("unique_test_query_123", embedding, "test-model")

        # Now should be found
        result = get_cached_query_embedding("unique_test_query_123", "test-model")
        assert result is not None
        np.testing.assert_array_equal(result, embedding)
