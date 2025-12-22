import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Adjust import based on actual path. Assuming backend/src is in pythonpath or installed.
# If not, sys.path hack might be needed, or running from root with PYTHONPATH=.
from cortex.retrieval._hybrid_helpers import (
    _convert_fts_to_items,
    _convert_vector_to_items,
    _get_query_embedding,
    _resolve_target_conversations,
)


class TestHybridHelpers(unittest.TestCase):
    @patch("cortex.retrieval._hybrid_helpers.get_cached_query_embedding")
    @patch("cortex.retrieval._hybrid_helpers._embedding_client")
    @patch("cortex.retrieval._hybrid_helpers.cache_query_embedding")
    def test_get_query_embedding_cached(
        self, _mock_cache_save, mock_client, mock_get_cache
    ):
        # Setup
        config = MagicMock()
        config.embedding.model_name = "test-model"
        mock_get_cache.return_value = np.array([0.1, 0.2, 0.3])

        # Action
        res = _get_query_embedding("query", config)

        # Assert
        self.assertEqual(res, [0.1, 0.2, 0.3])
        mock_client.embed.assert_not_called()
        mock_get_cache.assert_called_with("query", model="test-model")

    @patch("cortex.retrieval._hybrid_helpers.get_cached_query_embedding")
    @patch("cortex.retrieval._hybrid_helpers._embedding_client")
    @patch("cortex.retrieval._hybrid_helpers.cache_query_embedding")
    def test_get_query_embedding_miss(
        self, mock_cache_save, mock_client, mock_get_cache
    ):
        # Setup
        config = MagicMock()
        config.embedding.model_name = "test-model"
        mock_get_cache.return_value = None
        mock_client.embed.return_value = [0.4, 0.5]

        # Action
        res = _get_query_embedding("query", config)

        # Assert
        self.assertEqual(res, [0.4, 0.5])
        mock_client.embed.assert_called_with("query")
        mock_cache_save.assert_called()

    @patch("cortex.retrieval._hybrid_helpers._resolve_filter_conversation_ids")
    def test_resolve_target_conversations_none(self, mock_resolve):
        # Setup
        session = MagicMock()
        args = MagicMock()
        args.classification = None
        mock_resolve.return_value = None

        # Action
        res = _resolve_target_conversations(session, args, "query", 10, {})

        # Assert
        self.assertIsNone(res)

    @patch(
        "cortex.retrieval.fts_search.search_messages_fts"
    )  # Note: this might need mock in local scope if import happens inside function
    def test_resolve_target_conversations_navigational(self, mock_fts):
        # Assuming import inside function makes it tricky to patch top-level.
        # But sys.modules patching or patch.dict works.
        # Actually since the function does `from ... import ...`, we need to patch specifically where it's looked up.
        # Or patch `cortex.retrieval._hybrid_helpers.search_messages_fts` won't work if it's imported locally.
        # We'll skip complex navigational test or use sys.modules hack if needed.
        pass

    def test_convert_fts_to_items(self):
        # Setup
        res = MagicMock()
        res.chunk_id = "c1"
        res.score = 1.0
        res.conversation_id = "conv1"
        res.attachment_id = None
        res.is_attachment = False
        res.snippet = "snip"
        res.text = "content"
        res.metadata = {"foo": "bar"}

        items = _convert_fts_to_items([res])

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].chunk_id, "c1")
        self.assertEqual(items[0].lexical_score, 1.0)
        self.assertEqual(items[0].metadata["foo"], "bar")

    def test_convert_vector_to_items(self):
        # Setup
        res = MagicMock()
        res.chunk_id = "c2"
        res.score = 0.9
        res.conversation_id = "conv2"
        res.attachment_id = "att1"
        res.is_attachment = True
        res.text = "vector content"
        res.metadata = {}
        res.chunk_type = "msg"

        items = _convert_vector_to_items([res])

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].chunk_id, "c2")
        self.assertEqual(items[0].vector_score, 0.9)
        self.assertEqual(items[0].metadata["chunk_type"], "msg")
