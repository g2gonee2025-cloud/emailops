
import unittest
from unittest.mock import MagicMock, patch

from cortex.orchestrator import PipelineOrchestrator


class TestPipelineOrchestrator(unittest.TestCase):
    @patch("cortex.orchestrator.S3SourceHandler")
    @patch("cortex.orchestrator.IngestionProcessor")
    @patch("cortex.orchestrator.Indexer")
    def test_run(self, MockIndexer, MockIngestionProcessor, MockS3SourceHandler):
        # Arrange
        mock_s3_handler = MockS3SourceHandler.return_value
        mock_s3_handler.list_conversation_folders.return_value = iter(
            [MagicMock(), MagicMock()]
        )

        orchestrator = PipelineOrchestrator()
        orchestrator.s3_handler = mock_s3_handler
        orchestrator._process_single_folder = MagicMock()

        # Act
        stats = orchestrator.run()

        # Assert
        self.assertEqual(stats.folders_found, 2)
        self.assertEqual(orchestrator._process_single_folder.call_count, 2)

if __name__ == "__main__":
    unittest.main()
