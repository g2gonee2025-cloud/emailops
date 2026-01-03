import unittest
import logging
import asyncio
from cortex.intelligence.graph import GraphExtractor
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(".env")

TEST_TEXT = """
From: Test User (test.user@example.com)
To: Accounts Department
Subject: Urgent: Invoice #INV-000-A-1 Approval

Hi Team,

Please review Invoice #INV-000-A-1 from supplier 'Global Imports' for the amount of USD 1,234.
It is related to the Q1 shipment of widgets to the main warehouse.
The payment deadline is next week.

Thanks,
Test User
"""

class TestGraphExtraction(unittest.TestCase):
    async def test_extraction(self):
        """
        Tests the graph extraction process for a sample text.
        """
        logging.info("Initializing Extractor...")
        try:
            extractor = GraphExtractor()
            logging.info(f"Extracting graph from text ({len(TEST_TEXT)} chars)...")
            G = await extractor.extract_graph(TEST_TEXT)
        except Exception as e:
            self.fail(f"Graph extraction raised an exception: {e}")

        self.assertIsNotNone(G, "Graph extraction should not return None.")

        logging.info("Graph Extraction Complete.")
        node_count = G.number_of_nodes()
        edge_count = G.number_of_edges()
        logging.info(f"Nodes: {node_count}")
        logging.info(f"Edges: {edge_count}")

        # Log nodes
        logging.info("\n--- Nodes (showing max 20) ---")
        for i, (node, data) in enumerate(G.nodes(data=True)):
            if i >= 20:
                logging.info(f"... and {node_count - 20} more.")
                break
            logging.info(f"{node} ({data.get('type', 'UNKNOWN')})")

        # Log edges
        logging.info("\n--- Edges (showing max 20) ---")
        for i, (u, v, data) in enumerate(G.edges(data=True)):
            if i >= 20:
                logging.info(f"... and {edge_count - 20} more.")
                break
            logging.info(f"{u} --[{data.get('relation', 'RELATED')}]--> {v}")
            logging.info(f"   Desc: {data.get('description', '')}")

        self.assertTrue(node_count > 0 or edge_count > 0, "Graph should not be empty.")
        logging.info("SUCCESS: Graph extraction yielded results.")

if __name__ == "__main__":
    unittest.main()
