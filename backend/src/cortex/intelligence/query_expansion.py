"""
Query Expansion.

Expands user queries with synonyms to improve retrieval recall.
Uses WordNet for English synonyms, with optional LLM fallback.

Key improvements in this version:
- Produces PostgreSQL-compatible `to_tsquery` syntax.
- Efficiently caches the LLMRuntime instance.
- Singleton pattern for the expander to avoid re-initializing resources.
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
    Expands queries with synonyms, compatible with PostgreSQL FTS.
    """

    _llm: Any = None

    def __init__(self, use_llm_fallback: bool = False):
        self.use_llm_fallback = use_llm_fallback

    @property
    def _llm_runtime(self) -> Any:
        """Lazy-load the LLM runtime to avoid circular imports and startup costs."""
        if self._llm is None:
            try:
                from cortex.llm.runtime import LLMRuntime

                self._llm = LLMRuntime()
            except Exception as e:
                logger.error(f"Failed to initialize LLMRuntime: {e}")
                # Set a dummy object to prevent repeated initialization attempts
                self._llm = None
        return self._llm

    def expand(self, query: str, max_synonyms_per_term: int = 2) -> str:
        """
        Expand a query with synonyms for PostgreSQL `to_tsquery`.

        Returns a query string with `|` (OR) and `&` (AND) operators.
        Example: "laptop battery" -> "(laptop | notebook) & (battery | power)"
        """
        if not query or not query.strip():
            return query

        # Tokenize (simple whitespace split, lowercase)
        tokens = re.findall(r"\b\w+\b", query.lower())

        expanded_parts = []
        for token in tokens:
            synonyms = self._get_synonyms(token, max_synonyms_per_term)
            if synonyms:
                # Create OR clause for ts_query: (term1 | term2 | ...)
                all_terms = [token] + list(synonyms)
                # Sanitize terms for ts_query (remove special chars)
                sanitized_terms = [re.sub(r"[&|!()]", "", t) for t in all_terms]
                expanded_parts.append(f"({' | '.join(sanitized_terms)})")
            else:
                expanded_parts.append(token)

        # Join parts with AND for ts_query
        return " & ".join(expanded_parts)

    def _get_synonyms(self, term: str, max_count: int) -> Set[str]:
        """Get synonyms for a term."""
        synonyms: Set[str] = set()

        # Skip very short words
        if len(term) <= 2:
            return synonyms

        # Try WordNet first
        if WORDNET_AVAILABLE:
            try:
                for syn in wordnet.synsets(term):
                    for lemma in syn.lemmas():
                        name = lemma.name().replace("_", " ").lower()
                        # Basic validation
                        if name != term and " " not in name and len(name) > 2:
                            synonyms.add(name)
                            if len(synonyms) >= max_count:
                                return synonyms
            except Exception as e:
                # This can happen with first-time NLTK use, not an error.
                logger.debug(f"WordNet lookup failed for '{term}': {e}")

        # LLM fallback (if enabled and WordNet found nothing)
        if self.use_llm_fallback and not synonyms and self._llm_runtime:
            synonyms = self._llm_synonyms(term, max_count)

        return synonyms

    def _llm_synonyms(self, term: str, max_count: int) -> Set[str]:
        """Use LLM to generate synonyms (expensive, use sparingly)."""
        if not self._llm_runtime:
            return set()

        prompt = f"List up to {max_count} single-word synonyms for the word '{term}'. Return only the words, comma-separated, no explanation."
        try:
            # User requested GPT-OSS-120B for fallback intelligence
            response = self._llm_runtime.complete_text(
                prompt, temperature=0.0, max_tokens=50, model="gpt-oss-120b"
            )
            # Parse comma-separated response
            synonyms = {
                s.strip().lower()
                for s in response.split(",")
                if s.strip() and " " not in s.strip()
            }
            synonyms.discard(term)
            return synonyms
        except Exception as e:
            logger.debug(f"LLM synonym lookup failed: {e}")
            return set()


# --- Singleton Instance ---
# This ensures a single, shared instance of the expander is used across the app,
# preventing re-initialization of the LLM runtime.
_query_expander_instance = QueryExpander(use_llm_fallback=True)


def expand_for_fts(query: str) -> str:
    """
    Convenience function: expand query for Full-Text Search.
    Returns expanded query suitable for PostgreSQL to_tsquery.
    """
    return _query_expander_instance.expand(query)
