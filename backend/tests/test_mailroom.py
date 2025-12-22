"""Unit tests for cortex.ingestion.mailroom module."""

import uuid
from unittest.mock import patch

import pytest
from cortex.ingestion.models import IngestJobRequest, IngestJobSummary


class TestMailroomHelpers:
    def test_validate_local_path_valid(self, tmp_path):
        from cortex.ingestion.mailroom import _validate_local_path

        # Create a valid path
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        result = _validate_local_path(str(test_dir))
        # Result could be Path or str
        assert str(result) == str(test_dir)

    def test_validate_local_path_invalid(self):
        from cortex.ingestion.mailroom import _validate_local_path

        with pytest.raises(ValueError):
            _validate_local_path("/nonexistent/path/12345")

    def test_normalize_s3_prefix_simple(self):
        from cortex.ingestion.mailroom import _normalize_s3_prefix

        result = _normalize_s3_prefix("exports/emails", "my-bucket")
        # Returns (prefix, bucket) tuple
        assert result is not None

    def test_generate_stable_id(self):
        from cortex.ingestion.mailroom import _generate_stable_id

        ns = uuid.UUID("12345678-1234-5678-1234-567812345678")

        id1 = _generate_stable_id(ns, "arg1", "arg2")
        id2 = _generate_stable_id(ns, "arg1", "arg2")
        id3 = _generate_stable_id(ns, "arg1", "arg3")

        assert id1 == id2  # Same inputs = same output
        assert id1 != id3  # Different inputs = different output


class TestProcessJob:
    @patch("cortex.ingestion.mailroom._resolve_source")
    @patch("cortex.ingestion.mailroom._ingest_conversation")
    def test_process_job_local_source(self, mock_ingest, mock_resolve, tmp_path):
        from cortex.ingestion.mailroom import process_job

        # Setup
        convo_dir = tmp_path / "conv1"
        convo_dir.mkdir()
        (convo_dir / "manifest.json").write_text('{"subject": "test"}')

        mock_resolve.return_value = tmp_path

        job = IngestJobRequest(
            job_id=uuid.uuid4(),
            source_type="local_upload",
            source_uri=str(tmp_path),
            tenant_id="default",
        )

        summary = process_job(job)

        assert isinstance(summary, IngestJobSummary)

    @patch("cortex.ingestion.mailroom._resolve_source")
    def test_process_job_no_conversations(self, mock_resolve, tmp_path):
        from cortex.ingestion.mailroom import process_job

        # Empty directory - no conversation folders
        mock_resolve.return_value = tmp_path

        job = IngestJobRequest(
            job_id=uuid.uuid4(),
            source_type="local_upload",
            source_uri=str(tmp_path),
            tenant_id="default",
        )

        summary = process_job(job)

        # Should complete
        assert isinstance(summary, IngestJobSummary)


class TestResolveSource:
    @patch("cortex.ingestion.mailroom._download_s3_source")
    def test_resolve_source_s3(self, mock_download, tmp_path):
        from cortex.ingestion.mailroom import _resolve_source

        mock_download.return_value = tmp_path

        job = IngestJobRequest(
            job_id=uuid.uuid4(),
            source_type="s3",
            source_uri="s3://bucket/prefix",
            tenant_id="default",
        )

        result = _resolve_source(job)

        mock_download.assert_called()
        assert result == tmp_path
