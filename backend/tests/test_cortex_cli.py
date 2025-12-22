"""Unit tests for cortex.cli module (Typer-based CLI)."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

# Always mock CortexDoctor before importing app
with patch("cortex.cli.CortexDoctor"):
    from cortex.cli import app

runner = CliRunner()


class TestCortexCli:
    @patch("cortex.cli.CortexDoctor")
    def test_doctor_success(self, mock_doctor_cls):
        """Test doctor command when checks pass."""
        mock_doctor = MagicMock()
        mock_doctor.run_all.return_value = True
        mock_doctor_cls.return_value = mock_doctor

        result = runner.invoke(app, ["doctor"])

        assert result.exit_code == 0
        mock_doctor.run_all.assert_called_once()

    @patch("cortex.cli.CortexDoctor")
    def test_doctor_failure(self, mock_doctor_cls):
        """Test doctor command when checks fail."""
        mock_doctor = MagicMock()
        mock_doctor.run_all.return_value = False
        mock_doctor_cls.return_value = mock_doctor

        result = runner.invoke(app, ["doctor"])

        assert result.exit_code == 1

    @patch("cortex.cli.CortexDoctor")
    def test_doctor_check_all(self, mock_doctor_cls):
        """Test doctor command with --check-all flag."""
        mock_doctor = MagicMock()
        mock_doctor.run_all.return_value = True
        mock_doctor_cls.return_value = mock_doctor

        result = runner.invoke(app, ["doctor", "--check-all"])

        assert result.exit_code == 0

    @patch("cortex.ingestion.processor.IngestionProcessor")
    @patch("cortex.cli.CortexDoctor", MagicMock())
    def test_ingest_s3(self, mock_processor_cls):
        """Test ingest command with s3 source."""
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor

        result = runner.invoke(app, ["ingest", "--source", "s3"])

        assert result.exit_code == 0
        mock_processor.run_full_ingestion.assert_called_once()

    @patch("cortex.cli.CortexDoctor", MagicMock())
    def test_ingest_unsupported_source(self):
        """Test ingest command with unsupported source."""
        result = runner.invoke(app, ["ingest", "--source", "ftp"])

        # Should print error but not crash
        assert "not supported" in result.output

    @patch("cortex.ingestion.backfill.backfill_embeddings")
    @patch("cortex.cli.CortexDoctor", MagicMock())
    def test_ingest_backfill_embeddings(self, mock_backfill):
        """Test ingest command with backfill-embeddings source."""
        result = runner.invoke(
            app, ["ingest", "--source", "backfill-embeddings", "--tenant", "acme"]
        )

        assert result.exit_code == 0
        mock_backfill.assert_called_once_with(tenant_id="acme")
