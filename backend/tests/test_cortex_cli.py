"""Unit tests for cortex_cli."""

from unittest.mock import MagicMock, patch
from argparse import Namespace

from cortex_cli.main import main

# This test file has been adapted from a previous version that was testing a
# Typer-based CLI. The CLI is now based on argparse, so the tests have been
# rewritten to work with the new structure.

class TestCortexCli:
    @patch("cortex_cli.main._handle_doctor")
    def test_main_routes_doctor(self, mock_handle_doctor):
        """Test that 'cortex doctor' is routed correctly."""
        main(["doctor"])
        mock_handle_doctor.assert_called_once()

    @patch("cortex_cli.main.sys.exit")
    @patch("cortex_cli.cmd_doctor.main")
    def test_doctor_command_integration(self, mock_doctor_main, mock_sys_exit):
        """Test that the full 'doctor' command can be invoked."""
        main(["doctor", "--provider", "local"])
        mock_doctor_main.assert_called_once()
        mock_sys_exit.assert_not_called()

    # The ingest tests from the original file are commented out because they
    # are based on a deprecated implementation of the 'ingest' command.
    # The old tests were for a version of `ingest` that took a source type
    # (e.g., 's3', 'backfill-embeddings'), whereas the current version expects a
    # file path and uses the mailroom for processing.
    # TODO: Rewrite ingest tests for the new mailroom-based implementation.

    # @patch("cortex.ingestion.processor.IngestionProcessor")
    # def test_ingest_s3(self, mock_processor_cls):
    #     """Test ingest command with s3 source."""
    #     ...

    # @patch("cortex.ingestion.backfill.backfill_embeddings")
    # def test_ingest_backfill_embeddings(self, mock_backfill):
    #     """Test ingest command with backfill-embeddings source."""
    #     ...
