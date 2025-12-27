"""
Query Expansion.

Expands user queries with synonyms to improve retrieval recall.
Uses WordNet for English synonyms, with optional LLM fallback.

Key improvements in this version:
- Produces PostgreSQL-compatible `to_tsquery` syntax.
- Asynchronous and non-blocking, suitable for FastAPI.
- Hardened against prompt injection by using structured messages.
- Concurrent synonym lookups for improved performance.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, List, Set

from cortex.llm.async_runtime import AsyncLLMRuntime
from cortex.llm.runtime import ConfigurationError, ProviderError

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

    _llm: AsyncLLMRuntime | None = None

    def __init__(self, use_llm_fallback: bool = False):
        self.use_llm_fallback = use_llm_fallback

    @property
    def _llm_runtime(self) -> AsyncLLMRuntime | None:
        """Lazy-load the AsyncLLMRuntime to avoid circular imports and startup costs."""
        if self._llm is None:
            try:
                from cortex.llm.async_runtime import get_async_runtime
                self._llm = get_async_runtime()
            except (ImportError, ConfigurationError) as e:
                logger.error(f"LLM runtime for query expansion is unavailable: {e}")
                return None
            except Exception as e:
                logger.critical(f"Unexpected error initializing LLM runtime: {e}", exc_info=True)
                return None
        return self._llm

    async def expand(self, query: str, max_synonyms_per_term: int = 2) -> str:
        """
        Asynchronously expand a query with synonyms for PostgreSQL `to_tsquery`.

        Returns a query string with `|` (OR) and `&` (AND) operators.
        Example: "laptop battery" -> "(laptop | notebook) & (battery | power)"
        """
        if not query or not query.strip():
            return query

        tokens = re.findall(r"\b\w+\b", query.lower())

        # Concurrently get synonyms for all tokens
        synonym_lists = await asyncio.gather(
            *(self._get_synonyms(token, max_synonyms_per_term) for token in tokens)
        )

        expanded_parts = []
        for token, synonyms in zip(tokens, synonym_lists):
            if synonyms:
                all_terms = [token] + list(synonyms)
                sanitized_terms = [re.sub(r"[&|!()]", "", t) for t in all_terms]
                expanded_parts.append(f"({' | '.join(sanitized_terms)})")
            else:
                expanded_parts.append(token)

        return " & ".join(expanded_parts)

    async def _get_synonyms(self, term: str, max_count: int) -> Set[str]:
        """Get synonyms for a term, async."""
        synonyms: Set[str] = set()

        if len(term) <= 2:
            return synonyms

        if WORDNET_AVAILABLE:
            try:
                # WordNet is synchronous, but fast enough to not need run_in_executor
                for syn in wordnet.synsets(term):
                    for lemma in syn.lemmas():
                        name = lemma.name().replace("_", " ").lower()
                        if name != term and " " not in name and len(name) > 2:
                            synonyms.add(name)
                            if len(synonyms) >= max_count:
                                return synonyms
            except Exception as e:
                logger.debug(f"WordNet lookup failed for '{term}': {e}")

        if self.use_llm_fallback and not synonyms and self._llm_runtime:
            synonyms = await self._llm_synonyms(term, max_count)

        return synonyms

    async def _llm_synonyms(self, term: str, max_count: int) -> Set[str]:
        """Use LLM to generate synonyms asynchronously and securely."""
        if not self._llm_runtime:
            return set()

        # Mitigate Prompt Injection: Use structured messages
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant. List up to {max_count} single-word synonyms. Return only the words, comma-separated, without explanation.",
            },
            {"role": "user", "content": term},
        ]

        try:
            response = await self._llm_runtime.complete_text(
                messages,
                temperature=0.0,
                max_tokens=50,
                model="gpt-oss-120b",
            )
            synonyms = {
                s.strip().lower()
                for s in response.split(",")
                if s.strip() and " " not in s.strip()
            }
            synonyms.discard(term)
            return synonyms
        except ProviderError as e:
            logger.debug(f"Async LLM synonym lookup failed: {e}")
            return set()


# --- Singleton Instance ---
_query_expander_instance = QueryExpander(use_llm_fallback=True)


async def expand_for_fts(query: str) -> str:
    """
    Async convenience function: expand query for Full-Text Search.
    """
    return await _query_expander_instance.expand(query)
