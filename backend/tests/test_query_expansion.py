"""
Unit tests for query expansion.
"""

from unittest.mock import patch, Mock, AsyncMock

import pytest
from cortex.intelligence.query_expansion import QueryExpander, expand_for_fts


@pytest.mark.asyncio
async def test_expand_no_wordnet_no_llm():
    """Test expansion when no synonym providers are available."""
    with patch("cortex.intelligence.query_expansion.WORDNET_AVAILABLE", False):
        expander = QueryExpander(use_llm_fallback=False)
        expander._llm = None  # Ensure no LLM runtime
        assert await expander.expand("computer networking") == "computer & networking"


@pytest.mark.asyncio
@patch("cortex.intelligence.query_expansion.WORDNET_AVAILABLE", True)
@patch("cortex.intelligence.query_expansion.wordnet", create=True)
async def test_expand_with_wordnet(mock_wordnet):
    """Test expansion using mocked WordNet."""
    mock_synset = Mock()
    mock_lemma = Mock()
    mock_lemma.name.return_value = "laptop"
    mock_synset.lemmas.return_value = [mock_lemma]
    mock_wordnet.synsets.return_value = [mock_synset]

    expander = QueryExpander(use_llm_fallback=False)
    expanded = await expander.expand("computer")
    assert "(computer | laptop)" in expanded


@pytest.mark.asyncio
async def test_expand_with_llm_fallback():
    """Test expansion using LLM fallback."""
    with patch("cortex.intelligence.query_expansion.WORDNET_AVAILABLE", False):
        mock_llm_runtime = AsyncMock()
        mock_llm_runtime.complete_text.return_value = "notebook, desktop"

        expander = QueryExpander(use_llm_fallback=True)
        expander._llm = mock_llm_runtime

        query = "computer"
        expanded = await expander.expand(query)
        assert "computer" in expanded
        assert "notebook" in expanded
        assert "desktop" in expanded
        assert "|" in expanded
        assert "(" in expanded


@pytest.mark.asyncio
async def test_expand_empty_query():
    """Test that an empty query is returned as is."""
    assert await expand_for_fts("") == ""
    assert await expand_for_fts("   ") == "   "


@pytest.mark.asyncio
async def test_expand_single_term():
    """Test expansion of a single term."""
    expanded = await expand_for_fts("database")
    assert expanded.startswith("(") or expanded == "database"


@pytest.mark.asyncio
async def test_postgres_syntax():
    """Verify the output syntax is compatible with to_tsquery."""
    expander = QueryExpander(use_llm_fallback=False)

    async def mock_get_synonyms(term, max_count):
        if term == "computer":
            return {"notebook", "laptop"}
        if term == "battery":
            return {"power", "supply"}
        return set()

    with patch.object(expander, '_get_synonyms', side_effect=mock_get_synonyms):
        expanded = await expander.expand("computer battery")

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

        assert sorted_expanded == "(computer | laptop | notebook) & (battery | power | supply)"
