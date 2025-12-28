import unittest
from unittest.mock import MagicMock

from cortex.orchestrator import PipelineOrchestrator


class TestPipelineOrchestrator(unittest.TestCase):
    def test_run_dry_run(self):
        """Test orchestrator in dry-run mode (no external dependencies)."""
        orchestrator = PipelineOrchestrator(dry_run=True)
        stats = orchestrator.run()

        # Dry run mode processes no folders
        self.assertEqual(stats.folders_found, 0)
        self.assertEqual(stats.folders_processed, 0)
        self.assertEqual(stats.folders_failed, 0)


if __name__ == "__main__":
    unittest.main()
