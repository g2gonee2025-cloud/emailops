import unittest
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from cortex.ingestion.models import IngestJobSummary
from cortex.ingestion.processor import IngestionProcessor
from cortex.ingestion.s3_source import S3ConversationFolder


class TestIncrementalIngestion(unittest.TestCase):
    @patch("cortex.ingestion.processor.process_job")
    @patch("cortex.db.session.SessionLocal")
    @patch("cortex.ingestion.s3_source.get_config")
    @patch("cortex.ingestion.processor.get_config")
    def test_incremental_processing(
        self, mock_processor_config, mock_s3_config, mock_session_cls, mock_process_job
    ):
        # Setup
        mock_s3_config.return_value.storage.endpoint_url = "http://localhost:9000"
        tenant_id = "test_tenant"
        processor = IngestionProcessor(tenant_id=tenant_id)

        # Mock folder with specific last_modified
        base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        folder = S3ConversationFolder(
            prefix="raw/test_folder/",
            name="test_folder",
            files=["key1"],
            last_modified=base_time,
        )

        # Mock DB Session
        mock_session = mock_session_cls.return_value.__enter__.return_value

        # SCENARIO 1: No existing record -> Should Process
        mock_session.query.return_value.filter.return_value.first.return_value = None
        mock_process_job.return_value = IngestJobSummary(
            job_id=uuid.uuid4(), tenant_id=tenant_id
        )

        result = processor.process_folder(folder)

        self.assertIsNotNone(result)
        self.assertEqual(processor.stats.folders_processed, 1)
        mock_process_job.assert_called()

        # Verify job options contained timestamp
        call_args = mock_process_job.call_args[0][0]
        self.assertEqual(
            call_args.options["source_last_modified"], base_time.isoformat()
        )

        # Reset mocks for next scenario
        mock_process_job.reset_mock()
        processor.stats.folders_processed = 0

        # SCENARIO 2: Existing record is OLDER -> Should Process
        existing_record_old = MagicMock()
        existing_record_old.extra_data = {
            "source_last_modified": (base_time - timedelta(hours=1)).isoformat()
        }
        mock_session.query.return_value.filter.return_value.first.return_value = (
            existing_record_old
        )

        result = processor.process_folder(folder)

        self.assertIsNotNone(result)
        mock_process_job.assert_called()

        # Reset
        mock_process_job.reset_mock()

        # SCENARIO 3: Existing record is NEWER/EQUAL -> Should SKIP
        existing_record_current = MagicMock()
        existing_record_current.extra_data = {
            "source_last_modified": base_time.isoformat()
        }
        mock_session.query.return_value.filter.return_value.first.return_value = (
            existing_record_current
        )

        result = processor.process_folder(folder)

        self.assertIsNone(result)
        mock_process_job.assert_not_called()
