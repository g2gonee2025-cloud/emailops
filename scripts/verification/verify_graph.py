"""
Verify Graph RAG Extraction (Live LLM).
"""

import logging
import os
import sys
import traceback


def configure_sys_path():
    """Add project root to sys.path to allow for absolute imports."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    backend_src = os.path.join(project_root, "backend", "src")
    if backend_src not in sys.path:
        sys.path.insert(0, backend_src)


configure_sys_path()

from cortex.db.models import EntityEdge, EntityNode
from cortex.intelligence.graph import GraphExtractor

logger = logging.getLogger(__name__)


def test_graph_extractor_live():
    """Test graph extraction from text using LIVE LLM."""

    # Real text input
    text = "Alice (alice@example.com) is the lead engineer for Project Alpha. She reports to Bob managed_by."

    logger.info("Initializing GraphExtractor (Live LLM)...")
    extractor = GraphExtractor()

    logger.info("Extracting graph from text...")
    try:
        G = extractor.extract_graph(text)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise

    if G is None:
        raise ValueError("Graph extraction returned None.")

    # Verify Nodes (LLM-dependent, so we check for existence of meaningful nodes)
    logger.info(
        f"Extracted {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
    )

    if G.number_of_nodes() == 0:
        raise ValueError("No nodes were extracted from the text.")

    # Loose assertions for live test
    # We expect "Alice" and "Project Alpha" at minimum
    node_names = [str(n).lower() for n in G.nodes if isinstance(n, str)]
    if not any("alice" in n for n in node_names):
        raise ValueError(
            "Verification failed: Expected 'Alice' to be in the extracted nodes."
        )

    logger.info("GraphExtractor Live Test Passed!")


def verify_db_schema():
    """Verify EntityNode and EntityEdge have the expected __tablename__ attribute."""
    if not hasattr(EntityNode, "__tablename__"):
        raise AttributeError(
            "EntityNode model is missing the '__tablename__' attribute."
        )
    if not hasattr(EntityEdge, "__tablename__"):
        raise AttributeError(
            "EntityEdge model is missing the '__tablename__' attribute."
        )
    logger.info("Graph DB Schema Passed!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    try:
        verify_db_schema()
        test_graph_extractor_live()
        logger.info("GRAPH VERIFICATION SUCCESSFUL")
    except Exception:
        logger.error("GRAPH VERIFICATION FAILED")
        logger.error(traceback.format_exc())
        sys.exit(1)
