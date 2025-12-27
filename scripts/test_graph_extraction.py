from cortex.intelligence.graph import GraphExtractor
from dotenv import load_dotenv

load_dotenv(".env")

TEST_TEXT = """
From: Sarah Jenkins (sarah.jenkins@chalhoub.com)
To: Finance Department
Subject: Urgent: Invoice #INV-2024-001 Approval

Hi Team,

Please review Invoice #INV-2024-001 from supplier 'Logistics Co' for the amount of AED 5,000.
It is related to the Q4 shipment of cosmetics to the Dubai Mall store.
The payment deadline is next Friday.

Thanks,
Sarah
"""


def test_extraction():
    print("Initializing Extractor...")
    extractor = GraphExtractor()

    print(f"Extracting graph from text ({len(TEST_TEXT)} chars)...")
    G = extractor.extract_graph(TEST_TEXT)

    print("\nGraph Extraction Complete.")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    print("\n--- Nodes ---")
    for node, data in G.nodes(data=True):
        print(f"{node} ({data.get('type', 'UNKNOWN')})")

    print("\n--- Edges ---")
    for u, v, data in G.edges(data=True):
        print(f"{u} --[{data.get('relation', 'RELATED')}]--> {v}")
        print(f"   Desc: {data.get('description', '')}")

    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        print("\nSUCCESS: Graph extraction yielded results.")
    else:
        print("\nFAILURE: Graph is empty.")


if __name__ == "__main__":
    test_extraction()
