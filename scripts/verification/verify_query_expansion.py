"""
Verify Query Expansion (Live).
"""

import logging
import sys

# Wrap import in a try-except block to handle ImportError gracefully.
# This script must be run with `backend/src` in the PYTHONPATH.
# For example: PYTHONPATH=backend/src python scripts/verification/verify_query_expansion.py
try:
    from cortex.intelligence.query_expansion import QueryExpander, expand_for_fts
except ImportError:
    sys.stderr.write(
        "Error: Could not import Cortex modules. Please run this script from the project root with `PYTHONPATH=backend/src`.\n"
    )
    sys.exit(1)

# Configure module logger without altering global logging configuration
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _validate(condition, message):
    """Custom assertion to ensure validation runs even with -O flag."""
    if not condition:
        raise AssertionError(message)


def test_query_expansion_live():
    """Test query expansion including LIVE LLM fallback."""

    # 1. Test WordNet (Local)
    logger.info("Testing WordNet expansion (Local)...")
    expander = QueryExpander(use_llm_fallback=False)
    query = "laptop battery"
    expanded = expander.expand(query)
    logger.info("Expanded (WordNet): %s", expanded)
    # NULL_SAFETY & TYPE_ERRORS: Check if the result is a non-empty string
    _validate(
        isinstance(expanded, str) and expanded,
        "WordNet expansion returned a non-string or empty value.",
    )
    # LOGIC_ERRORS: Strengthen assertion
    _validate(
        "laptop" in expanded.lower() and "& (battery" in expanded.lower(),
        f"WordNet expansion for '{query}' was incorrect. Got: {expanded}",
    )

    # 2. Test LLM Fallback (Live)
    logger.info("Testing LLM Fallback expansion (Live GPT-OSS-120B)...")
    expander_live = QueryExpander(use_llm_fallback=True)

    term = "Kubernetes"
    expanded_live = expander_live.expand(term)
    logger.info("Expanded (LLM Fallback) for '%s': %s", term, expanded_live)

    # NULL_SAFETY & TYPE_ERRORS: Check if the result is a non-empty string
    _validate(
        isinstance(expanded_live, str) and expanded_live,
        "LLM fallback expansion returned a non-string or empty value.",
    )

    # LOGIC_ERRORS: Add a real assertion to verify fallback and expansion
    _validate(
        term.lower() != expanded_live.lower(),
        f"LLM fallback failed to expand the term '{term}'. The output was the same as the input.",
    )
    _validate(
        term.lower() in expanded_live.lower(),
        f"LLM expansion for '{term}' did not contain the original term. Got: {expanded_live}",
    )
    logger.info("LLM successfully expanded the term!")

    # 3. Test Convenience Function (Default)
    logger.info("Testing default expand_for_fts...")
    query_fts = "deploy container"
    res = expand_for_fts(query_fts)
    logger.info("FTS Result: %s", res)
    # LOGIC_ERRORS: Add assertion for the convenience function
    _validate(
        isinstance(res, str) and "&" in res,
        f"expand_for_fts for '{query_fts}' did not produce a valid FTS query. Got: {res}",
    )

    logger.info("Query Expansion Live Test Passed!")


if __name__ == "__main__":
    # Configure basic logging to see output for standalone script execution.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    try:
        test_query_expansion_live()
        logger.info("QUERY EXPANSION VERIFICATION SUCCESSFUL")
    except Exception:
        logger.exception("QUERY EXPANSION VERIFICATION FAILED")
        sys.exit(1)
