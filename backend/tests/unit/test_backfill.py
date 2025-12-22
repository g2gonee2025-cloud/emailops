import unittest
from unittest.mock import MagicMock, patch

from cortex.ingestion.backfill import (
    _get_chunks_without_embeddings,
    _get_missing_embeddings_count,
    _process_embedding_batch,
    backfill_embeddings,
    get_openai_client,
)


class TestBackfill(unittest.TestCase):
    @patch("cortex.ingestion.backfill.os.getenv")
    def test_get_openai_client(self, mock_env):
        mock_env.return_value = "http://fake/v1"

        # Mock sys.modules to ensure openai appears present so import works (or absent to test fallback)
        # But here we want to test success path if installed.
        # Actually backfill.py does `try: from openai ...`.
        # To test success, we need to make sure `import openai` succeeds.

        import sys

        mock_openai_module = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai_module}):
            # We also need to patch behavior of the import inside the function?
            # No, if it's in sys.modules, import returns it.

            # However, get_openai_client() does `from openai import OpenAI`.
            # So mock_module must have OpenAI attribute.
            mock_openai_module.OpenAI = MagicMock()

            client = get_openai_client()
            self.assertIsNotNone(client)
            mock_openai_module.OpenAI.assert_called()

    @patch("cortex.ingestion.backfill.SessionLocal")
    @patch("cortex.ingestion.backfill.get_openai_client")
    @patch("cortex.ingestion.backfill._get_missing_embeddings_count")
    @patch("cortex.ingestion.backfill._get_chunks_without_embeddings")
    @patch("cortex.ingestion.backfill._process_embedding_batch")
    def test_backfill_flow(
        self,
        mock_process,
        mock_get_chunks,
        mock_count,
        mock_client_factory,
        mock_session,
    ):
        client = MagicMock()
        mock_client_factory.return_value = client
        mock_count.return_value = 10

        # Determine chunks returned
        chunk = MagicMock()
        chunk.text = "abc"
        mock_get_chunks.side_effect = [
            [chunk],
            [],
        ]  # First call returns 1 chunk, second call empty (end loop)

        mock_process.return_value = 1

        backfill_embeddings(limit=None)

        mock_process.assert_called()

    def test_get_missing_embeddings_count(self):
        session = MagicMock()
        session.execute.return_value.scalar.return_value = 5
        res = _get_missing_embeddings_count(session)
        self.assertEqual(res, 5)

    def test_get_chunks_without_embeddings(self):
        session = MagicMock()
        session.execute.return_value.scalars.return_value.all.return_value = ["c1"]
        res = _get_chunks_without_embeddings(session, 10)
        self.assertEqual(res, ["c1"])

    def test_process_embedding_batch_success(self):
        client = MagicMock()
        session = MagicMock()
        chunk = MagicMock()
        chunk.text = "t"

        resp = MagicMock()
        data_item = MagicMock()
        data_item.embedding = [0.1, 0.2]
        resp.data = [data_item]
        client.embeddings.create.return_value = resp

        count = _process_embedding_batch(client, session, [chunk], ["t"], "model")
        self.assertEqual(count, 1)
        self.assertEqual(chunk.embedding, [0.1, 0.2])
        session.commit.assert_called()

    @patch("cortex.ingestion.backfill._process_chunks_serially")
    def test_process_embedding_batch_failure_fallback(self, mock_serial):
        client = MagicMock()
        session = MagicMock()
        client.embeddings.create.side_effect = Exception("Boom")

        _process_embedding_batch(client, session, [], [], "model")
        mock_serial.assert_called()
