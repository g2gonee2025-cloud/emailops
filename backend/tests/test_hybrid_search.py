"""Integration tests for hybrid search functionality.

Uses live infrastructure (DB, embeddings, config) - no mocks.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from cortex.retrieval.hybrid_search import (
    KBSearchInput,
    SearchResultItem,
    apply_recency_boost,
    deduplicate_by_hash,
    downweight_quoted_history,
    fuse_rrf,
    fuse_weighted_sum,
    tool_kb_search_hybrid,
)
from cortex.retrieval.query_classifier import QueryClassification


class TestHybridSearchUnit:
    """Unit tests for hybrid search helper functions (no DB/embedding needed)."""

    @pytest.fixture
    def sample_results(self):
        """Create sample search results for testing."""
        return [
            SearchResultItem(
                chunk_id="1",
                score=0.9,
                content="test 1",
                conversation_id="c1",
                content_hash="h1",
                metadata={"chunk_type": "text"},
            ),
            SearchResultItem(
                chunk_id="2",
                score=0.8,
                content="test 2",
                conversation_id="c2",
                content_hash="h2",
                metadata={"chunk_type": "text"},
            ),
            SearchResultItem(
                chunk_id="3",
                score=0.7,
                content="test 3",
                conversation_id="c3",
                content_hash="h1",  # Duplicate hash
                metadata={"chunk_type": "text"},
            ),
        ]

    def test_apply_recency_boost(self, sample_results):
        """Test timestamp-based score boosting."""
        now = datetime.now(UTC)
        thread_updated_at = {
            "c1": now,  # Very fresh
            "c2": now - timedelta(days=30),  # 1 half-life old
        }

        boosted = apply_recency_boost(
            sample_results[:2],
            thread_updated_at,
            half_life_days=30.0,
            boost_strength=1.0,
        )

        assert boosted[0].chunk_id == "1"
        assert boosted[0].score > 1.7
        assert boosted[1].chunk_id == "2"
        assert boosted[1].score > 1.1 and boosted[1].score < 1.3

    def test_deduplicate_by_hash(self, sample_results):
        """Test deduplication logic."""
        deduped = deduplicate_by_hash(sample_results)

        assert len(deduped) == 2
        ids = {x.chunk_id for x in deduped}
        assert "1" in ids
        assert "3" not in ids
        assert "2" in ids

    def test_downweight_quoted_history(self, sample_results):
        """Test downweighting of quoted history."""
        sample_results[0].metadata["chunk_type"] = "quoted_history"
        original_score = sample_results[0].score

        weighted = downweight_quoted_history(sample_results, factor=0.5)

        assert weighted[0].score == pytest.approx(original_score * 0.5)
        assert weighted[1].score == 0.8  # Unchanged

    def test_fuse_rrf(self):
        """Test RRF fusion."""
        fts = [
            SearchResultItem(chunk_id="A", score=10, content="A"),
            SearchResultItem(chunk_id="B", score=5, content="B"),
        ]
        vector = [
            SearchResultItem(chunk_id="B", score=0.9, content="B"),
            SearchResultItem(chunk_id="C", score=0.8, content="C"),
        ]

        fused = fuse_rrf(fts, vector, k=1)

        assert fused[0].chunk_id == "B"
        assert fused[1].chunk_id == "A"
        assert fused[2].chunk_id == "C"

    def test_fuse_weighted_sum(self):
        """Test weighted sum fusion."""
        fts = [
            SearchResultItem(chunk_id="A", score=0.5, lexical_score=0.5, content="A")
        ]
        vector = [
            SearchResultItem(chunk_id="A", score=0.8, vector_score=0.8, content="A")
        ]

        fused = fuse_weighted_sum(fts, vector, alpha=0.5)

        assert len(fused) == 1
        assert abs(fused[0].score - 0.65) < 0.0001
        assert fused[0].lexical_score == pytest.approx(0.5)
        assert fused[0].vector_score == pytest.approx(0.8)


class TestHybridSearchIntegration:
    """Integration tests using live infrastructure."""

    @pytest.mark.asyncio
    async def test_tool_kb_search_hybrid_empty_query(self):
        """Test search with empty/minimal query against live DB."""
        input_args = KBSearchInput(
            tenant_id="default",
            user_id="test-user",
            query="test",
        )
        result = await tool_kb_search_hybrid(input_args)

        # Should return Ok result (even if empty)
        assert result.is_ok()
        results = result.unwrap()
        assert hasattr(results, "results")
        assert hasattr(results, "reranker")

    @pytest.mark.asyncio
    async def test_tool_kb_search_hybrid_with_filters(self):
        """Test search with filter syntax."""
        input_args = KBSearchInput(
            tenant_id="default",
            user_id="test-user",
            query="type:pdf contract",
        )
        result = await tool_kb_search_hybrid(input_args)

        assert result.is_ok()

    @pytest.mark.asyncio
    async def test_tool_kb_search_hybrid_navigational(self):
        """Test navigational search classification."""
        input_args = KBSearchInput(
            tenant_id="default",
            user_id="test-user",
            query="find email from john",
            classification=QueryClassification(
                query="find email from john",
                type="navigational",
                flags=[],
            ),
        )
        result = await tool_kb_search_hybrid(input_args)

        assert result.is_ok()

    @pytest.mark.asyncio
    async def test_tool_kb_search_weighted_sum_fusion(self):
        """Test weighted_sum fusion method."""
        input_args = KBSearchInput(
            tenant_id="default",
            user_id="test-user",
            query="insurance claim",
            fusion_method="weighted_sum",
        )
        result = await tool_kb_search_hybrid(input_args)

        assert result.is_ok()
        results = result.unwrap()
        assert "weighted_sum" in results.reranker

    @pytest.mark.asyncio
    async def test_sql_injection_through_file_types(self):
        """Test that file_types filter is not vulnerable to SQL injection."""
        # Malicious query attempting to inject SQL
        malicious_query = "type:pdf'), OR 1=1; --"
        input_args = KBSearchInput(
            tenant_id="default",
            user_id="test-user",
            query=malicious_query,
        )

        # Should not raise an exception - injection should be sanitized
        result = await tool_kb_search_hybrid(input_args)
        assert result.is_ok()
