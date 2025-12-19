import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from cortex.ingestion.mailroom import _generate_stable_id, _ingest_conversation
from cortex.ingestion.models import IngestJobRequest, IngestJobSummary


class TestIdempotency(unittest.TestCase):
    def test_stable_id_generation(self):
        # Verify that generating an ID twice with same inputs produces same UUID
        ns = uuid.NAMESPACE_DNS
        id1 = _generate_stable_id(ns, "test", "foo")
        id2 = _generate_stable_id(ns, "test", "foo")
        id3 = _generate_stable_id(ns, "test", "bar")

        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)
        self.assertIsInstance(id1, uuid.UUID)

    @patch("cortex.ingestion.conv_loader.load_conversation")
    @patch("cortex.ingestion.attachments_log.parse_attachments_log")
    @patch("cortex.ingestion.writer.DBWriter")
    @patch("cortex.db.session.SessionLocal")
    @patch("cortex.ingestion.text_preprocessor.get_text_preprocessor")
    @patch("cortex.chunking.chunker.chunk_text")
    def test_ingest_conversation_stable_ids(
        self,
        mock_chunk,
        mock_preproc,
        mock_session,
        mock_writer_cls,
        mock_log,
        mock_load,
    ):
        # Setup Mocks
        convo_dir = Path("/tmp/test_convo")
        # Ensure only the LAST component is used for ID generation to match implementation

        job = IngestJobRequest(
            job_id=uuid.uuid4(),
            tenant_id="tenant-123",
            source_uri="s3://bucket/test_convo",
            source_type="s3",
        )
        summary = IngestJobSummary(job_id=job.job_id, tenant_id=job.tenant_id)

        mock_load.return_value = {
            "path": "/tmp/test_convo",
            "manifest": {"message_count": 1},
            "conversation_txt": "Body text",
            "attachments": [
                {"path": "/tmp/test_convo/att1.pdf", "text": "Attachment text"}
            ],
        }
        mock_log.return_value = {}

        # Mock Chunking
        mock_chunk_obj = MagicMock()
        mock_chunk_obj.text = "Chunk text"
        mock_chunk_obj.position = 0
        mock_chunk_obj.char_start = 0
        mock_chunk_obj.char_end = 10
        mock_chunk_obj.section_path = "root"
        mock_chunk_obj.metadata = {}
        mock_chunk.return_value = [mock_chunk_obj]

        # Mock Preprocessor
        mock_preproc.return_value.prepare_for_indexing.return_value = (
            "Cleaned text",
            {},
        )

        # Capture the written results
        mock_writer = mock_writer_cls.return_value

        # Run 1
        _ingest_conversation(convo_dir, job, summary)

        # Capture IDs from Run 1
        call_args_1 = mock_writer.write_job_results.call_args[0]
        results_1 = call_args_1[1]

        conv_id_1 = results_1["conversation"]["conversation_id"]
        att_id_1 = results_1["attachments"][0]["attachment_id"]
        chunk_id_1 = results_1["chunks"][0]["chunk_id"]  # Body chunk
        chunk_id_2 = results_1["chunks"][1]["chunk_id"]  # Attachment chunk

        # Run 2 (Same input)
        _ingest_conversation(convo_dir, job, summary)

        # Capture IDs from Run 2
        call_args_2 = mock_writer.write_job_results.call_args_list[1][0]
        results_2 = call_args_2[1]

        conv_id_2 = results_2["conversation"]["conversation_id"]
        att_id_2 = results_2["attachments"][0]["attachment_id"]
        chunk_id_3 = results_2["chunks"][0]["chunk_id"]
        chunk_id_4 = results_2["chunks"][1]["chunk_id"]

        # Assert assertions
        self.assertEqual(conv_id_1, conv_id_2, "Conversation ID should be stable")
        self.assertEqual(att_id_1, att_id_2, "Attachment ID should be stable")
        self.assertEqual(chunk_id_1, chunk_id_3, "Body Chunk ID should be stable")
        self.assertEqual(chunk_id_2, chunk_id_4, "Attachment Chunk ID should be stable")
