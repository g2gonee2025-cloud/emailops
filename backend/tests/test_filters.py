"""
Unit tests for Search Filters and Grammar Parser.
"""

from datetime import datetime, timezone

import pytest
from cortex.retrieval.filters import (
    SearchFilters,
    apply_filters_to_sql,
    parse_filter_grammar,
)


class TestSearchFilters:
    """Tests for SearchFilters dataclass."""

    def test_empty_filters(self):
        """Test is_empty() for default empty filters."""
        f = SearchFilters()
        assert f.is_empty() is True

    def test_non_empty_with_from_emails(self):
        """Test is_empty() when a filter is set."""
        f = SearchFilters(from_emails={"john@example.com"})
        assert f.is_empty() is False

    def test_to_dict(self):
        """Test serialization to dict."""
        f = SearchFilters(
            from_emails={"john@example.com"},
            subject_contains=["budget"],
            has_attachment=True,
        )
        d = f.to_dict()

        assert "from_emails" in d
        assert "john@example.com" in d["from_emails"]
        assert d["subject_contains"] == ["budget"]
        assert d["has_attachment"] is True


class TestParseFilterGrammar:
    """Tests for parse_filter_grammar function."""

    def test_simple_from_filter(self):
        """Test parsing from:email filter."""
        filters, clean = parse_filter_grammar("from:john@example.com budget report")

        assert filters.from_emails == {"john@example.com"}
        assert clean == "budget report"

    def test_quoted_subject_filter(self):
        """Test parsing subject with quotes."""
        filters, clean = parse_filter_grammar('subject:"Q4 Budget" review')

        assert filters.subject_contains == ["q4 budget"]
        assert clean == "review"

    def test_multiple_filters(self):
        """Test parsing multiple filters."""
        filters, clean = parse_filter_grammar(
            "from:john@test.com to:jane@test.com subject:budget after:2024-01-01"
        )

        assert filters.from_emails == {"john@test.com"}
        assert filters.to_emails == {"jane@test.com"}
        assert filters.subject_contains == ["budget"]
        assert filters.date_from is not None
        assert filters.date_from.year == 2024
        assert clean == ""

    def test_has_attachment_filter(self):
        """Test has:attachment filter."""
        filters, clean = parse_filter_grammar("has:attachment invoice")

        assert filters.has_attachment is True
        assert clean == "invoice"

    def test_no_attachment_filter(self):
        """Test has:noattachment filter."""
        filters, clean = parse_filter_grammar("has:noattachment text only")

        assert filters.has_attachment is False
        assert clean == "text only"

    def test_file_type_filter(self):
        """Test type:pdf,docx filter."""
        filters, clean = parse_filter_grammar("type:pdf,docx documents")

        assert filters.file_types == {"pdf", "docx"}
        assert clean == "documents"

    def test_exclude_terms(self):
        """Test -term exclusion."""
        filters, clean = parse_filter_grammar("budget report -draft -internal")

        assert filters.exclude_terms == ["draft", "internal"]
        assert clean == "budget report"

    def test_date_range_filters(self):
        """Test after: and before: date filters."""
        filters, clean = parse_filter_grammar(
            "after:2024-01-01 before:2024-12-31 annual report"
        )

        assert filters.date_from is not None
        assert filters.date_from.year == 2024
        assert filters.date_from.month == 1

        assert filters.date_to is not None
        assert filters.date_to.year == 2024
        assert filters.date_to.month == 12

        assert clean == "annual report"

    def test_no_filters_returns_empty(self):
        """Test query with no filters."""
        filters, clean = parse_filter_grammar("just a normal search query")

        assert filters.is_empty() is True
        assert clean == "just a normal search query"

    def test_case_insensitive_keys(self):
        """Test that filter keys are case-insensitive."""
        filters, clean = parse_filter_grammar("FROM:John@Test.com")

        assert filters.from_emails == {"john@test.com"}

    def test_empty_query(self):
        """Test empty query."""
        filters, clean = parse_filter_grammar("")

        assert filters.is_empty() is True
        assert clean == ""


class TestApplyFiltersToSql:
    """Tests for SQL filter generation."""

    def test_empty_filters_no_sql(self):
        """Test that empty filters produce empty SQL."""
        f = SearchFilters()
        clause, params = apply_filters_to_sql(f)

        assert clause == ""
        assert params == {}

    def test_date_from_filter(self):
        """Test date_from generates correct SQL."""
        f = SearchFilters(date_from=datetime(2024, 1, 1, tzinfo=timezone.utc))
        clause, params = apply_filters_to_sql(f)

        assert "latest_date >= :date_from" in clause
        assert "date_from" in params

    def test_has_attachment_filter(self):
        """Test has_attachment generates correct SQL."""
        f = SearchFilters(has_attachment=True)
        clause, params = apply_filters_to_sql(f)

        assert "is_attachment = TRUE" in clause

    def test_subject_contains_filter(self):
        """Test subject_contains generates ILIKE clause."""
        f = SearchFilters(subject_contains=["budget", "2024"])
        clause, params = apply_filters_to_sql(f)

        assert "subject ILIKE" in clause
        assert "subject_term_0" in params
        assert "subject_term_1" in params

    def test_from_emails_filter(self):
        """Test from_emails generates correct JSONB SQL."""
        f = SearchFilters(from_emails={"test@example.com"})
        clause, params = apply_filters_to_sql(f)
        assert "p->>'role' = 'sender'" in clause
        assert "lower(p->>'smtp') = ANY(:from_emails)" in clause
        assert params["from_emails"] == ["test@example.com"]

    def test_to_emails_filter(self):
        """Test to_emails generates correct JSONB SQL."""
        f = SearchFilters(to_emails={"test@example.com"})
        clause, params = apply_filters_to_sql(f)
        assert "p->>'role' = 'recipient'" in clause
        assert "lower(p->>'smtp') = ANY(:to_emails)" in clause
        assert params["to_emails"] == ["test@example.com"]

    def test_cc_emails_filter(self):
        """Test cc_emails generates correct JSONB SQL."""
        f = SearchFilters(cc_emails={"test@example.com"})
        clause, params = apply_filters_to_sql(f)
        assert "p->>'role' = 'cc'" in clause
        assert "lower(p->>'smtp') = ANY(:cc_emails)" in clause
        assert params["cc_emails"] == ["test@example.com"]

    def test_file_types_filter(self):
        """Test file_types generates correct SQL."""
        f = SearchFilters(file_types={"pdf", "docx"})
        clause, params = apply_filters_to_sql(f)
        assert "metadata->>'file_type'" in clause
        assert ":file_types" in clause
        # Order is not guaranteed in set, so check for presence
        assert "pdf" in params["file_types"]
        assert "docx" in params["file_types"]

    def test_sql_injection_invalid_alias(self):
        """Test that invalid table aliases raise ValueError."""
        f = SearchFilters(has_attachment=True)
        with pytest.raises(ValueError):
            apply_filters_to_sql(f, table_alias="c; DROP TABLE users;")

        with pytest.raises(ValueError):
            apply_filters_to_sql(f, conversation_table_alias="conv; --")
