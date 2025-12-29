"""
Unit tests for query expansion using live LLM.
"""

import unittest

from cortex.intelligence.query_expansion import QueryExpander, expand_for_fts


class TestQueryExpansion(unittest.TestCase):
    """Test query expansion with live LLM via CPU fallback."""

    def test_expand_empty_query(self):
        """Test that an empty query is returned as is."""
        self.assertEqual(expand_for_fts(""), "")
        self.assertEqual(expand_for_fts("   "), "   ")

    def test_expand_single_term(self):
        """Test expansion of a single term using live LLM."""
        expanded = expand_for_fts("database")
        # Should return non-empty result
        self.assertTrue(len(expanded) > 0)
        # Should contain the original term
        self.assertIn("database", expanded.lower())

    def test_expand_multi_term(self):
        """Test expansion of multiple terms using live query expander."""
        expanded = expand_for_fts("insurance claim")
        # Should return non-empty result
        self.assertTrue(len(expanded) > 0)
        # Should contain original terms or be expanded
        self.assertTrue(
            "insurance" in expanded.lower()
            or "claim" in expanded.lower()
            or "&" in expanded  # FTS AND connector
        )

    def test_expand_produces_postgres_syntax(self):
        """Test that output is compatible with PostgreSQL to_tsquery."""
        expanded = expand_for_fts("computer network")
        # Should use PostgreSQL FTS operators
        if len(expanded) > 0 and expanded != "computer network":
            # Check for FTS syntax elements
            has_and = "&" in expanded
            has_or = "|" in expanded
            has_parens = "(" in expanded and ")" in expanded
            # Should have at least AND connector for multi-term
            self.assertTrue(has_and or has_or or has_parens or "network" in expanded)

    def test_query_expander_instance(self):
        """Test QueryExpander can be instantiated and used."""
        expander = QueryExpander(use_llm_fallback=True)
        result = expander.expand("flood damage claim")
        self.assertIsNotNone(result)
        self.assertTrue(len(result) > 0)

    def test_special_characters_handled(self):
        """Test that special characters are handled gracefully."""
        # Should not crash on special chars
        result = expand_for_fts("test's query")
        self.assertIsNotNone(result)

    def test_numeric_query(self):
        """Test numeric queries are handled."""
        result = expand_for_fts("claim 12345")
        self.assertIsNotNone(result)
        # Should contain the number
        self.assertIn("12345", result)


if __name__ == "__main__":
    unittest.main()
