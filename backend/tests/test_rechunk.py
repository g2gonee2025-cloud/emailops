"""Unit tests for cortex.ingestion.rechunk module."""

from unittest.mock import MagicMock, patch

import pytest


# Mock the database session
@pytest.fixture
def mock_session():
    with patch("cortex.ingestion.rechunk.SessionLocal") as mock:
        session = MagicMock()
        mock.return_value = session
        yield session


@pytest.mark.skip(reason="rechunk module does not exist - tests obsolete")
class TestRechunkFailed:
    @patch("cortex.ingestion.rechunk.SessionLocal")
    @patch("cortex.ingestion.rechunk.chunk_text")
    def test_rechunk_no_bad_chunks(self, mock_chunk_text, mock_session_local, capsys):
        """Test when no skipped chunks are found."""
        from cortex.ingestion.rechunk import rechunk_failed

        mock_session = MagicMock()
        mock_session_local.return_value = mock_session
        mock_session.execute.return_value.scalars.return_value.all.return_value = []

        rechunk_failed()

        mock_session.commit.assert_not_called()
        _ = capsys.readouterr()  # Consume output

    @patch("cortex.ingestion.rechunk.SessionLocal")
    @patch("cortex.ingestion.rechunk.chunk_text")
    def test_rechunk_with_bad_chunks(self, mock_chunk_text, mock_session_local):
        """Test processing of skipped chunks."""
        from cortex.ingestion.rechunk import rechunk_failed

        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        # Create mock bad chunk
        bad_chunk = MagicMock()
        bad_chunk.chunk_id = "chunk-1"
        bad_chunk.text = "This is test text that needs rechunking."
        bad_chunk.section_path = "section/path"
        bad_chunk.tenant_id = "default"
        bad_chunk.conversation_id = "conv-1"
        bad_chunk.attachment_id = None
        bad_chunk.is_attachment = False
        bad_chunk.position = 0
        bad_chunk.char_start = 0
        bad_chunk.char_end = 100

        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            bad_chunk
        ]

        # Mock chunk_text to return new models
        mock_model = MagicMock()
        mock_model.text = "new chunk text"
        mock_model.char_start = 0
        mock_model.char_end = 20
        mock_chunk_text.return_value = [mock_model]

        rechunk_failed()

        mock_session.add.assert_called()
        mock_session.delete.assert_called_with(bad_chunk)
        mock_session.commit.assert_called()

    @patch("cortex.ingestion.rechunk.SessionLocal")
    @patch("cortex.ingestion.rechunk.chunk_text")
    def test_rechunk_handles_exception(
        self, mock_chunk_text, mock_session_local, capsys
    ):
        """Test error handling during rechunk."""
        from cortex.ingestion.rechunk import rechunk_failed

        mock_session = MagicMock()
        mock_session_local.return_value = mock_session
        mock_session.execute.side_effect = Exception("DB Error")

        rechunk_failed()

        mock_session.rollback.assert_called()
        mock_session.close.assert_called()
