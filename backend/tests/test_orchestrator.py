import unittest
from unittest.mock import MagicMock, patch

from cortex.orchestrator import PipelineOrchestrator


class TestPipelineOrchestrator(unittest.TestCase):
    @patch("cortex.orchestrator.S3SourceHandler")
    @patch("cortex.orchestrator.IngestionProcessor")
    @patch("cortex.orchestrator.Indexer")
    @patch("cortex.orchestrator.get_queue", create=True)
    def test_run(
        self, MockGetQueue, MockIndexer, MockIngestionProcessor, MockS3SourceHandler
    ):
        # Arrange
        mock_s3_handler = MockS3SourceHandler.return_value

        # Create mock folders that will be consumed by the iterator
        mock_folder1 = MagicMock()
        mock_folder1.name = "folder1"
        mock_folder2 = MagicMock()
        mock_folder2.name = "folder2"
        mock_s3_handler.list_conversation_folders.return_value = iter(
            [mock_folder1, mock_folder2]
        )

        # Mock the queue
        mock_queue = MagicMock()
        MockGetQueue.return_value = mock_queue
        mock_queue.enqueue.return_value = "job-123"

        # Act
        orchestrator = PipelineOrchestrator()
        stats = orchestrator.run()

        # Assert - folders_found is set based on successfully enqueued jobs
        self.assertEqual(stats.folders_found, 2)
        self.assertEqual(mock_queue.enqueue.call_count, 2)


if __name__ == "__main__":
    unittest.main()
