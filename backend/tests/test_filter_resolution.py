"""
Unit tests for Filter Resolution logic.
"""

from unittest.mock import MagicMock

from cortex.db.models import Conversation
from cortex.retrieval.filter_resolution import _resolve_filter_conversation_ids
from cortex.retrieval.filters import SearchFilters


class TestFilterResolution:
    """Tests for _resolve_filter_conversation_ids."""

    def test_no_filters_returns_none(self):
        """Test that empty filters return None (no restriction)."""
        session = MagicMock()
        filters = SearchFilters()

        result = _resolve_filter_conversation_ids(session, filters, "tenant-1")

        assert result is None
        session.query.assert_not_called()

    def test_only_conv_ids_returns_list(self):
        """Test that only conv_ids returns the list directly without DB query."""
        session = MagicMock()
        filters = SearchFilters(conv_ids={"uuid-1", "uuid-2"})

        result = _resolve_filter_conversation_ids(session, filters, "tenant-1")

        assert result is not None
        assert set(result) == {"uuid-1", "uuid-2"}
        session.query.assert_not_called()

    def test_subject_filter_constructs_query(self):
        """Test that subject filter triggers DB query."""
        session = MagicMock()
        mock_query = session.query.return_value
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []

        filters = SearchFilters(subject_contains=["budget"])

        _resolve_filter_conversation_ids(session, filters, "tenant-1")

        # Verify query structure
        session.query.assert_called_with(Conversation.conversation_id)
        # Check that filters were applied
        assert mock_query.filter.call_count >= 2  # tenant_id + subject

    def test_participant_filter_constructs_query(self):
        """Test that from_emails trigger query."""
        session = MagicMock()
        mock_query = session.query.return_value
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []

        filters = SearchFilters(from_emails={"john@test.com"})

        _resolve_filter_conversation_ids(session, filters, "tenant-1")

        # Check calls
        session.query.assert_called()
        # Should have called filter with OR clause for participants
        assert mock_query.filter.call_count >= 2
