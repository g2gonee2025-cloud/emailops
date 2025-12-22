"""
Query Expansion.

Expands user queries with synonyms to improve retrieval recall.
Uses WordNet for English synonyms, with optional LLM fallback.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Set

logger = logging.getLogger(__name__)

# Try to import WordNet (NLTK)
try:
    from nltk.corpus import wordnet

    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False
    logger.warning("NLTK WordNet not available. Query expansion will be limited.")


class QueryExpander:
    """
    Expands queries with synonyms.
    """

    def __init__(self, use_llm_fallback: bool = False):
        self.use_llm_fallback = use_llm_fallback
        self._llm: Any = None

    def expand(self, query: str, max_synonyms_per_term: int = 3) -> str:
        """
        Expand a query with synonyms.

        Returns an expanded query string with OR clauses.
        Example: "laptop battery" -> "(laptop OR notebook) (battery OR power)"
        """
        if not query or not query.strip():
            return query

        # Tokenize (simple whitespace split, lowercase)
        tokens = re.findall(r"\b\w+\b", query.lower())

        expanded_parts = []
        for token in tokens:
            synonyms = self._get_synonyms(token, max_synonyms_per_term)
            if synonyms:
                # Create OR clause
                all_terms = [token] + list(synonyms)
                expanded_parts.append(f"({' OR '.join(all_terms)})")
            else:
                expanded_parts.append(token)

        return " ".join(expanded_parts)

    def _get_synonyms(self, term: str, max_count: int) -> Set[str]:
        """Get synonyms for a term."""
        synonyms: Set[str] = set()

        # Skip very short words or stopwords
        if len(term) <= 2:
            return synonyms

        # Try WordNet first
        if WORDNET_AVAILABLE:
            try:
                for syn in wordnet.synsets(term):
                    for lemma in syn.lemmas():
                        name = lemma.name().replace("_", " ").lower()
                        if name != term and len(name) > 2:
                            synonyms.add(name)
                            if len(synonyms) >= max_count:
                                return synonyms
            except Exception as e:
                logger.debug(f"WordNet lookup failed for '{term}': {e}")

        # LLM fallback (if enabled and WordNet found nothing)
        if self.use_llm_fallback and not synonyms:
            synonyms = self._llm_synonyms(term, max_count)

        return synonyms

    def _llm_synonyms(self, term: str, max_count: int) -> Set[str]:
        """Use LLM to generate synonyms (expensive, use sparingly)."""
        if self._llm is None:
            try:
                from cortex.llm.runtime import LLMRuntime

                self._llm = LLMRuntime()
            except Exception:
                return set()

        prompt = f"List up to {max_count} synonyms for the word '{term}'. Return only the words, comma-separated, no explanation."
        try:
            # User requested GPT-OSS-120B for fallback intelligence
            response = self._llm.complete_text(
                prompt, temperature=0.0, max_tokens=50, model="gpt-oss-120b"
            )
            # Parse comma-separated response
            synonyms = {s.strip().lower() for s in response.split(",") if s.strip()}
            synonyms.discard(term)
            return synonyms
        except Exception as e:
            logger.debug(f"LLM synonym lookup failed: {e}")
            return set()


def expand_for_fts(query: str) -> str:
    """
    Convenience function: expand query for Full-Text Search.
    Returns expanded query suitable for PostgreSQL ts_query or Elasticsearch.
    """
    # Enabled LLM fallback with GPT-OSS-120B as requested
    expander = QueryExpander(use_llm_fallback=True)
    return expander.expand(query)
