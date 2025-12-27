"""
Unit tests for query expansion.
"""

import unittest
from unittest.mock import Mock, patch

from cortex.intelligence.query_expansion import QueryExpander, expand_for_fts


class TestQueryExpansion(unittest.TestCase):
    @patch("cortex.intelligence.query_expansion.WORDNET_AVAILABLE", False)
    @patch("cortex.intelligence.query_expansion.QueryExpander._llm_runtime")
    def test_expand_no_wordnet_no_llm(self, mock_llm_runtime):
        """Test expansion when no synonym providers are available."""
        mock_llm_runtime.return_value = None
        expander = QueryExpander(use_llm_fallback=False)
        self.assertEqual(
            expander.expand("computer networking"), "computer & networking"
        )

    @patch("cortex.intelligence.query_expansion.WORDNET_AVAILABLE", True)
    @patch("cortex.intelligence.query_expansion.wordnet", create=True)
    def test_expand_with_wordnet(self, mock_wordnet):
        """Test expansion using mocked WordNet."""
        # Mock WordNet response
        mock_synset = Mock()
        mock_lemma = Mock()
        mock_lemma.name.return_value = "laptop"
        mock_synset.lemmas.return_value = [mock_lemma]
        mock_wordnet.synsets.return_value = [mock_synset]

        with patch(
            "cortex.intelligence.query_expansion._query_expander_instance"
        ) as mock_expander_instance:
            mock_expander = QueryExpander(use_llm_fallback=False)
            mock_expander_instance.expand.side_effect = mock_expander.expand

            query = "computer"
            expanded = expand_for_fts(query)
            self.assertIn("(computer | laptop)", expanded)

    @patch("cortex.intelligence.query_expansion.WORDNET_AVAILABLE", False)
    @patch("cortex.intelligence.query_expansion.QueryExpander._llm_runtime")
    def test_expand_with_llm_fallback(self, mock_llm_runtime):
        """Test expansion using LLM fallback."""
        # Mock LLM response
        mock_llm_runtime.complete_text.return_value = "notebook, desktop"

        expander = QueryExpander(use_llm_fallback=True)
        expander._llm = mock_llm_runtime  # Inject mock

        query = "computer"
        expanded = expander.expand(query)
        self.assertIn("computer", expanded)
        self.assertIn("notebook", expanded)
        self.assertIn("desktop", expanded)
        self.assertIn("|", expanded)
        self.assertIn("(", expanded)

    def test_expand_empty_query(self):
        """Test that an empty query is returned as is."""
        self.assertEqual(expand_for_fts(""), "")
        self.assertEqual(expand_for_fts("   "), "   ")

    def test_expand_single_term(self):
        """Test expansion of a single term (relies on available WordNet or no expansion)."""
        # This test is less deterministic, so we check for basic structure
        expanded = expand_for_fts("database")
        self.assertTrue(expanded.startswith("(") or expanded == "database")

    def test_postgres_syntax(self):
        """Verify the output syntax is compatible with to_tsquery."""
        with patch(
            "cortex.intelligence.query_expansion.QueryExpander._get_synonyms"
        ) as mock_syn:
            mock_syn.side_effect = [
                {"notebook", "laptop"},
                {"power", "supply"},
            ]
            with patch(
                "cortex.intelligence.query_expansion._query_expander_instance"
            ) as mock_expander_instance:
                mock_expander = QueryExpander(use_llm_fallback=False)
                mock_expander._get_synonyms = mock_syn
                mock_expander_instance.expand.side_effect = mock_expander.expand

                expanded = expand_for_fts("computer battery")

                # Sort terms within each group to ensure deterministic output
                parts = expanded.split(" & ")
                sorted_parts = []
                for part in parts:
                    if part.startswith("("):
                        terms = part.strip("()").split(" | ")
                        terms.sort()
                        sorted_parts.append(f"({' | '.join(terms)})")
                    else:
                        sorted_parts.append(part)
                sorted_expanded = " & ".join(sorted_parts)

                self.assertEqual(
                    sorted_expanded,
                    "(computer | laptop | notebook) & (battery | power | supply)",
                )


if __name__ == "__main__":
    unittest.main()
