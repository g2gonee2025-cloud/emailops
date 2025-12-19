"""
Verify Query Expansion.
"""
import logging
import sys

# Ensure backend/src is in path
sys.path.append("backend/src")

from cortex.intelligence.query_expansion import QueryExpander, expand_for_fts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_query_expansion():
    """Test query expansion with WordNet."""
    expander = QueryExpander(use_llm_fallback=False)

    # Test basic expansion
    query = "laptop battery"
    expanded = expander.expand(query)

    logger.info(f"Original: {query}")
    logger.info(f"Expanded: {expanded}")

    # Verify structure (should have OR clauses if synonyms found)
    assert "laptop" in expanded.lower(), "Original term should be preserved"

    # Test convenience function
    expanded2 = expand_for_fts("deadline project")
    logger.info(f"FTS Expansion: {expanded2}")

    logger.info("Query Expansion Test Passed!")


if __name__ == "__main__":
    try:
        test_query_expansion()
        print("QUERY EXPANSION VERIFICATION SUCCESSFUL")
    except Exception as e:
        print(f"QUERY EXPANSION VERIFICATION FAILED: {e}")
        sys.exit(1)
