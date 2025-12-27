from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
import numpy as np
import pytest

# Import the correct return types
from cortex.retrieval.fts_search import ChunkFTSResult, FTSResult
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
from cortex.retrieval.vector_store import VectorResult


class TestHybridSearch:
    """Test suite for hybrid search functionality."""

    @pytest.fixture
    def mock_results(self):
        """Create sample search results."""
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

    def test_apply_recency_boost(self, mock_results):
        """Test timestamp-based score boosting."""
        now = datetime.now(timezone.utc)
        thread_updated_at = {
            "c1": now,  # Very fresh
            "c2": now - timedelta(days=30),  # 1 half-life old
        }

        boosted = apply_recency_boost(
            mock_results[:2], thread_updated_at, half_life_days=30.0, boost_strength=1.0
        )

        assert boosted[0].chunk_id == "1"
        assert boosted[0].score > 1.7
        assert boosted[1].chunk_id == "2"
        assert boosted[1].score > 1.1 and boosted[1].score < 1.3

    def test_deduplicate_by_hash(self, mock_results):
        """Test deduplication logic."""
        deduped = deduplicate_by_hash(mock_results)

        assert len(deduped) == 2
        ids = {x.chunk_id for x in deduped}
        assert "1" in ids
        assert "3" not in ids
        assert "2" in ids

    def test_downweight_quoted_history(self, mock_results):
        """Test downweighting of quoted history."""
        mock_results[0].metadata["chunk_type"] = "quoted_history"
        original_score = mock_results[0].score

        weighted = downweight_quoted_history(mock_results, factor=0.5)

        assert weighted[0].score == pytest.approx(original_score * 0.5)
        assert weighted[1].score == 0.8  # Unchanged

    def test_fuse_rrf(self):
        """Test various RRF fusion scenarios."""
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

    @patch("cortex.config.loader.get_config")
    @patch("cortex.db.session.SessionLocal")
    @patch("cortex.retrieval._hybrid_helpers._get_runtime")
    @patch("cortex.retrieval.fts_search.search_chunks_fts")
    @patch("cortex.retrieval.vector_search.search_chunks_vector")
    @patch("cortex.retrieval.cache.get_cached_query_embedding")
    @patch("cortex.retrieval.hybrid_search._get_conversation_timestamps")
    def test_tool_kb_search_hybrid_flow(
        self,
        mock_timestamps,
        mock_get_cache,
        mock_search_vector,
        mock_search_fts,
        mock_get_runtime,
        mock_session_cls,
        mock_get_config,
    ):
        """Test the main search flow with mocks."""
        # Setup Config
        mock_config = Mock()
        mock_config.search.k = 10
        mock_config.search.fusion_strategy = "rrf"
        mock_config.search.candidates_multiplier = 2
        mock_config.search.reranker_endpoint = None
        mock_config.search.rerank_alpha = 0.5
        mock_config.search.half_life_days = 30
        mock_config.search.recency_boost_strength = 1.0
        mock_config.search.mmr_lambda = 0.5
        mock_config.embedding.model_name = "test-model"

        mock_get_config.return_value = mock_config

        # Setup Components
        mock_session = Mock()
        mock_session_cls.return_value.__enter__.return_value = mock_session

        # Setup Cache miss then Embed
        mock_get_cache.return_value = None
        mock_runtime = Mock()
        mock_runtime.embed_queries.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_get_runtime.return_value = mock_runtime

        # Setup Results with correct types
        # FTS returns ChunkFTSResult
        mock_search_fts.return_value = [
            ChunkFTSResult(
                chunk_id="1",
                conversation_id="c1",
                chunk_type="message_body",
                score=10.0,
                snippet="FTS Hit",
                text="FTS Hit Body",
            )
        ]

        # Vector returns VectorResult
        mock_search_vector.return_value = [
            VectorResult(
                chunk_id="1",
                conversation_id="c1",
                chunk_type="message_body",
                score=0.9,
                text="Vector Hit Body",
            )
        ]

        mock_timestamps.return_value = {"c1": datetime.now(timezone.utc)}

        # Execute
        input_args = KBSearchInput(
            tenant_id="t1",
            user_id="u1",
            query="test query",
        )
        results = tool_kb_search_hybrid(input_args)

        # Verification
        assert len(results.results) == 1
        assert results.results[0].chunk_id == "1"
        assert "rrf" in results.reranker

        # Check calls
        mock_runtime.embed_queries.assert_called_once_with(["test query"])
        mock_search_fts.assert_called_once()
        mock_search_vector.assert_called_once()
        mock_timestamps.assert_called_once()

    @patch("cortex.config.loader.get_config")
    @patch("cortex.db.session.SessionLocal")
    @patch("cortex.retrieval.fts_search.search_messages_fts")
    @patch("cortex.retrieval.fts_search.search_chunks_fts")
    @patch("cortex.retrieval.vector_search.search_chunks_vector")
    def test_navigational_search(
        self, mock_vector, mock_fts, mock_msg_fts, mock_session_cls, mock_config
    ):
        """Test navigational search logic triggers message search."""
        mock_config.return_value.search.k = 10
        mock_config.return_value.search.candidates_multiplier = 2
        mock_config.return_value.search.fusion_strategy = "rrf"
        mock_config.return_value.search.reranker_endpoint = None
        mock_config.return_value.search.rerank_alpha = 0.5
        mock_config.return_value.search.half_life_days = 30.0
        mock_config.return_value.search.recency_boost_strength = 1.0
        mock_config.return_value.search.mmr_lambda = 0.5
        mock_config.return_value.embedding.model_name = "test-model"

        mock_session_cls.return_value.__enter__.return_value = Mock()

        # Setup Navigational Hit (FTSResult)
        mock_msg_fts.return_value = [
            FTSResult(
                conversation_id="c_nav",
                subject="Subject",
                score=10.0,
                snippet="Snippet",
            )
        ]
        # And ensure normal search returns nothing
        mock_fts.return_value = []
        mock_vector.return_value = []

        input_args = KBSearchInput(
            tenant_id="t1",
            user_id="u1",
            query="find email",
            classification=QueryClassification(
                query="find email", type="navigational", flags=[]
            ),
        )

        with patch("cortex.retrieval._hybrid_helpers._get_runtime") as mock_get_runtime:
            mock_runtime = Mock()
            mock_runtime.embed_queries.return_value = np.array([[0.1, 0.2, 0.3]])
            mock_get_runtime.return_value = mock_runtime
            tool_kb_search_hybrid(input_args)

        # Verify message search was called
        mock_msg_fts.assert_called_once()

        # Verify FTS called with filtered conversation_id from navigational result
        _, kwargs = mock_fts.call_args
        assert "conversation_ids" in kwargs
        assert "c_nav" in kwargs["conversation_ids"]
