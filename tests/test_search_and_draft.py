"""
Unit tests for search_and_draft.py functions.
"""

from emailops.search_and_draft import SearchFilters, parse_filter_grammar


class TestParseFilterGrammar:
    """Test cases for parse_filter_grammar function."""

    def test_empty_query(self):
        """Test parsing empty query."""
        filters, cleaned = parse_filter_grammar("")
        assert isinstance(filters, SearchFilters)
        assert cleaned == ""
        assert filters.subject_contains is None
        assert filters.from_emails is None
        assert filters.to_emails is None
        assert filters.cc_emails is None
        assert filters.date_from is None
        assert filters.date_to is None
        assert filters.has_attachment is None
        assert filters.types is None
        assert filters.exclude_terms is None

    def test_subject_filter(self):
        """Test subject filter parsing."""
        filters, cleaned = parse_filter_grammar('subject:"important meeting" some text')
        assert filters.subject_contains == ["important meeting"]
        assert cleaned == "some text"

    def test_from_filter(self):
        """Test from filter parsing."""
        filters, cleaned = parse_filter_grammar('from:user@example.com query')
        assert filters.from_emails == {"user@example.com"}
        assert cleaned == "query"

    def test_multiple_from_filters(self):
        """Test multiple from filters."""
        filters, cleaned = parse_filter_grammar('from:user1@example.com from:user2@example.com query')
        assert filters.from_emails == {"user1@example.com", "user2@example.com"}
        assert cleaned == "query"

    def test_to_filter(self):
        """Test to filter parsing."""
        filters, cleaned = parse_filter_grammar('to:recipient@example.com query')
        assert filters.to_emails == {"recipient@example.com"}
        assert cleaned == "query"

    def test_cc_filter(self):
        """Test cc filter parsing."""
        filters, cleaned = parse_filter_grammar('cc:cc@example.com query')
        assert filters.cc_emails == {"cc@example.com"}
        assert cleaned == "query"

    def test_after_date_filter(self):
        """Test after date filter."""
        filters, cleaned = parse_filter_grammar('after:2023-01-01 query')
        assert filters.date_from is not None
        assert cleaned == "query"

    def test_before_date_filter(self):
        """Test before date filter."""
        filters, cleaned = parse_filter_grammar('before:2023-12-31 query')
        assert filters.date_to is not None
        assert cleaned == "query"

    def test_has_attachment_filter(self):
        """Test has attachment filter."""
        filters, cleaned = parse_filter_grammar('has:attachment query')
        assert filters.has_attachment is True
        assert cleaned == "query"

        filters, cleaned = parse_filter_grammar('has:noattachment query')
        assert filters.has_attachment is False
        assert cleaned == "query"

    def test_type_filter(self):
        """Test type filter."""
        filters, cleaned = parse_filter_grammar('type:pdf query')
        assert filters.types == {"pdf"}
        assert cleaned == "query"

    def test_multiple_type_filters(self):
        """Test multiple type filters."""
        filters, cleaned = parse_filter_grammar('type:pdf type:docx query')
        assert filters.types == {"pdf", "docx"}
        assert cleaned == "query"

    def test_exclude_terms(self):
        """Test exclude terms with - prefix."""
        filters, cleaned = parse_filter_grammar('query -exclude1 -exclude2')
        assert filters.exclude_terms == ["exclude1", "exclude2"]
        assert cleaned == "query"

    def test_quoted_values(self):
        """Test quoted filter values."""
        filters, cleaned = parse_filter_grammar('subject:"quoted subject" from:"user name" query')
        assert filters.subject_contains == ["quoted subject"]
        assert filters.from_emails == {"user name"}
        assert cleaned == "query"

    def test_case_insensitive_keys(self):
        """Test case insensitive filter keys."""
        filters, cleaned = parse_filter_grammar('SUBJECT:test FROM:user@example.com query')
        assert filters.subject_contains == ["test"]
        assert filters.from_emails == {"user@example.com"}
        assert cleaned == "query"

    def test_mixed_filters_and_text(self):
        """Test mixed filters and free text."""
        filters, cleaned = parse_filter_grammar('subject:meeting from:user@example.com urgent meeting notes -draft')
        assert filters.subject_contains == ["meeting"]
        assert filters.from_emails == {"user@example.com"}
        assert filters.exclude_terms == ["draft"]
        assert cleaned == "urgent meeting notes"

    def test_invalid_date_format(self):
        """Test invalid date format handling."""
        filters, cleaned = parse_filter_grammar('after:invalid-date query')
        # Should not crash, date_from should remain None
        assert filters.date_from is None
        assert cleaned == "query"

    def test_unquoted_values_with_spaces(self):
        """Test unquoted values that might have spaces."""
        # The regex captures until whitespace, so this should work
        filters, cleaned = parse_filter_grammar('subject:some subject query')
        assert filters.subject_contains == ["some"]
        assert cleaned == "subject query"
