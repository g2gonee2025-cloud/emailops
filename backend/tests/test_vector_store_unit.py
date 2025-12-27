from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from cortex.common.exceptions import RetrievalError
from cortex.retrieval.vector_store import (
    PgvectorStore,
    QdrantVectorStore,
    VectorResult,
    _process_pgvector_row,
    _process_qdrant_point,
    _validate_embedding,
)


class TestVectorStoreUtils:
    def test_validate_embedding_success(self):
        emb = [0.1, 0.2, 0.3]
        res = _validate_embedding(emb, 3)
        assert isinstance(res, np.ndarray)
        assert np.allclose(res, np.array(emb))

    def test_validate_embedding_dimension_mismatch(self):
        with pytest.raises(RetrievalError) as exc:
            _validate_embedding([0.1], 3)
        assert "Embedding dimension mismatch" in str(exc.value)

    def test_validate_embedding_non_numeric(self):
        with pytest.raises(RetrievalError) as exc:
            _validate_embedding(["a", "b"], 2)
        assert "numeric values" in str(exc.value)

    def test_process_pgvector_row(self):
        row = MagicMock()
        row.distance = 0.2
        row.chunk_id = "abc"
        row.chunk_type = "text"
        row.text = "content"
        row.conversation_id = "conv1"
        row.attachment_id = None
        row.extra_data = {"meta": "data"}
        row.is_attachment = False

        res = _process_pgvector_row(row)
        assert isinstance(res, VectorResult)
        assert res.chunk_id == "abc"
        assert res.score == 0.9  # 1 - 0.2 = 0.8, then normalized?
        # Logic: 1 - 0.2 = 0.8. score = max(0, min(1, (0.8 + 1)/2)) = 0.9. Correct.

    def test_process_qdrant_point(self):
        point = {
            "id": "abc",
            "score": 0.8,
            "payload": {
                "text": "content",
                "conversation_id": "conv1",
                "extra_data": {"meta": "data"},
            },
        }
        res = _process_qdrant_point(point)
        assert res.chunk_id == "abc"
        assert res.score == 0.8


class TestPgvectorStore:
    def test_search(self):
        session = MagicMock()
        # Mock result
        row = MagicMock()
        row.distance = 0.1
        row.chunk_id = "1"
        row.text = "text"
        row.extra_data = {}
        row.chunk_type = "text"
        row.is_attachment = False
        row.attachment_id = ""
        row.conversation_id = "c1"

        session.execute.return_value.fetchall.return_value = [row]

        store = PgvectorStore(session, 3)
        results = store.search([0.1, 0.1, 0.1], "tenant", limit=5)

        assert len(results) == 1
        assert results[0].chunk_id == "1"
        # Check normalization
        # dist=0.1 -> sim=0.9 -> score=(0.9+1)/2 = 0.95
        assert np.isclose(results[0].score, 0.95)


class TestQdrantVectorStore:
    def test_search_success(self):
        config = MagicMock()
        config.url = "http://qdrant"
        config.collection_name = "test"
        config.api_key = "key"

        store = QdrantVectorStore(config, 3)

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "result": [{"id": "1", "score": 0.9, "payload": {"text": "found"}}]
            }

            results = store.search([0.1, 0.1, 0.1], "tenant")
            assert len(results) == 1
            assert results[0].score == 0.9

    def test_search_with_file_types_filter(self):
        config = MagicMock()
        config.url = "http://qdrant"
        config.collection_name = "test"
        config.api_key = "key"

        store = QdrantVectorStore(config, 3)

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"result": []}

            store.search([0.1, 0.1, 0.1], "tenant", file_types=["pdf", "docx"])

            mock_post.assert_called_once()
            call_args, call_kwargs = mock_post.call_args
            payload = call_kwargs.get("json", {})
            filters = payload.get("filter", {})
            must_filters = filters.get("must", [])

            file_type_filter = next(
                (
                    f
                    for f in must_filters
                    if "should" in f
                    and any("file_type" in s.get("key", "") for s in f["should"])
                ),
                None,
            )

            assert file_type_filter is not None
            assert len(file_type_filter["should"]) == 2
            assert file_type_filter["should"][0] == {
                "key": "file_type",
                "match": {"any": ["pdf", "docx"]},
            }
            assert file_type_filter["should"][1] == {
                "key": "source_type",
                "match": {"any": ["pdf", "docx"]},
            }
