import os
import unittest
from unittest.mock import MagicMock, patch

from cortex.health import (
    check_embeddings,
    check_postgres,
    check_redis,
    check_reranker,
)


class HealthCheckTests(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock()

    @patch("sqlalchemy.create_engine")
    def test_check_postgres_success(self, mock_create_engine):
        mock_conn = MagicMock()
        mock_create_engine.return_value.connect.return_value.__enter__.return_value = (
            mock_conn
        )
        self.mock_config.database.url = "postgresql://user:pass@host/db"

        result = check_postgres(self.mock_config)

        self.assertTrue(result.success)
        self.assertEqual(result.name, "PostgreSQL")
        self.assertEqual(result.details, {"message": "Connected"})
        self.assertIsNone(result.error)

    @patch("sqlalchemy.create_engine")
    def test_check_postgres_failure(self, mock_create_engine):
        mock_create_engine.side_effect = Exception("Connection failed")
        self.mock_config.database.url = "postgresql://user:pass@host/db"

        result = check_postgres(self.mock_config)

        self.assertFalse(result.success)
        self.assertEqual(result.name, "PostgreSQL")
        self.assertEqual(result.error, "Connection failed")

    @patch("redis.from_url")
    def test_check_redis_success(self, mock_from_url):
        mock_from_url.return_value.ping.return_value = True
        self.mock_config.cache.url = "redis://localhost:6379"

        result = check_redis(self.mock_config)

        self.assertTrue(result.success)
        self.assertEqual(result.name, "Redis")
        self.assertEqual(result.details, {"message": "Connected"})

    @patch("redis.from_url")
    def test_check_redis_failure(self, mock_from_url):
        mock_from_url.return_value.ping.side_effect = Exception("Ping failed")
        self.mock_config.cache.url = "redis://localhost:6379"

        result = check_redis(self.mock_config)

        self.assertFalse(result.success)
        self.assertEqual(result.name, "Redis")
        self.assertEqual(result.error, "Ping failed")

    @patch.dict(os.environ, {"OUTLOOKCORTEX_DB_URL": "sqlite:///:memory:"})
    @patch("cortex.llm.client.embed_texts")
    def test_check_embeddings_success(self, mock_embed_texts):
        mock_embed_texts.return_value = [[0.1, 0.2, 0.3]]
        result = check_embeddings(self.mock_config)

        self.assertTrue(result.success)
        self.assertEqual(result.name, "Embeddings")
        self.assertEqual(result.details, {"dimension": 3})

    @patch.dict(os.environ, {"OUTLOOKCORTEX_DB_URL": "sqlite:///:memory:"})
    @patch("cortex.llm.client.embed_texts")
    def test_check_embeddings_failure(self, mock_embed_texts):
        mock_embed_texts.side_effect = Exception("Embedding failed")
        result = check_embeddings(self.mock_config)

        self.assertFalse(result.success)
        self.assertEqual(result.name, "Embeddings")
        self.assertEqual(result.error, "Embedding failed")

    @patch("httpx.get")
    def test_check_reranker_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp
        self.mock_config.search.reranker_endpoint = "http://reranker"

        result = check_reranker(self.mock_config)

        self.assertTrue(result.success)
        self.assertEqual(result.name, "Reranker")
        self.assertEqual(result.details, {"message": "Connected"})

    @patch("httpx.get")
    def test_check_reranker_failure(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp
        self.mock_config.search.reranker_endpoint = "http://reranker"

        result = check_reranker(self.mock_config)

        self.assertFalse(result.success)
        self.assertEqual(result.name, "Reranker")
        self.assertIn("status 500", result.error)

    def test_check_reranker_no_endpoint(self):
        self.mock_config.search.reranker_endpoint = None
        result = check_reranker(self.mock_config)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "No reranker endpoint configured")


if __name__ == "__main__":
    unittest.main()
