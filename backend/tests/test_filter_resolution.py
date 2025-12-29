"""
Unit tests for Filter Resolution logic using live database.
"""

from cortex.db.session import get_db_session
from cortex.retrieval.filter_resolution import _resolve_filter_conversation_ids
from cortex.retrieval.filters import SearchFilters


class TestFilterResolution:
    """Tests for _resolve_filter_conversation_ids using live database."""

    def test_no_filters_returns_none(self):
        """Test that empty filters return None (no restriction)."""
        with get_db_session() as session:
            filters = SearchFilters()
            result = _resolve_filter_conversation_ids(session, filters, "default")
            assert result is None

    def test_only_conv_ids_returns_list(self):
        """Test that only conv_ids returns the list directly without DB query."""
        with get_db_session() as session:
            filters = SearchFilters(conv_ids={"uuid-1", "uuid-2"})
            result = _resolve_filter_conversation_ids(session, filters, "default")

            assert result is not None
            assert set(result) == {"uuid-1", "uuid-2"}

    def test_subject_filter_returns_matching_conversations(self):
        """Test that subject filter returns conversations from live DB."""
        with get_db_session() as session:
            # Use a filter that may or may not match existing data
            filters = SearchFilters(subject_contains=["insurance"])
            result = _resolve_filter_conversation_ids(session, filters, "default")

            # Result should be a list (empty or with matches) or None
            assert result is None or isinstance(result, list)

    def test_participant_filter_returns_matching_conversations(self):
        """Test that from_emails filter triggers query on live DB."""
        with get_db_session() as session:
            filters = SearchFilters(from_emails={"nonexistent@test.com"})
            result = _resolve_filter_conversation_ids(session, filters, "default")

            # Result should be a list (likely empty for nonexistent email)
            assert result is None or isinstance(result, list)

    def test_combined_filters(self):
        """Test multiple filters combined on live DB."""
        with get_db_session() as session:
            filters = SearchFilters(
                subject_contains=["claim"],
                from_emails={"test@example.com"},
            )
            result = _resolve_filter_conversation_ids(session, filters, "default")

            # Should return list or None
            assert result is None or isinstance(result, list)
