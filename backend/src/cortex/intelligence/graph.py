"""
Graph RAG Extractor.

Extracts Knowledge Graph entities and relationships from text using LLM.
Adapted from RAGFlow/GraphRAG.
"""
from __future__ import annotations

import logging
import re

import networkx as nx
from cortex.llm.runtime import LLMRuntime

logger = logging.getLogger(__name__)


# --- Prompts ---

GRAPH_EXTRACTION_PROMPT = """
You are a data scientist working for a company that is building a knowledge graph of its email communications.
Your task is to extract information from an email conversation and convert it into a knowledge graph.
You are given a text chunk from an email conversation.
Your goal is to identify the entities (nodes) and relations (edges) in the text.

Nodes are defined by the entities that appear in the text.
Edges are defined by the relationships between these entities.

Refine the nodes and edges to be distinct and unambiguous.
Use the following strict format for your output.
Do NOT output anything else except the list of tuples.

Nodes constraints:
1. "type" should be a high-level category like "PERSON", "ORGANIZATION", "PROJECT", "EVENT", "DATE", "LOCATION", "PRODUCT".
2. "name" should be the canonical name of the entity.

Edges constraints:
1. "relation" should be a SCREAMING_SNAKE_CASE verb phrase like "MANAGED_BY", "WORKED_ON", "HAS_DEADLINE", "LOCATED_IN".
2. "description" is a short sentence explaining the relationship based on the text.

Output Format:
("entity1_name", "entity1_type", "relation", "entity2_name", "entity2_type", "description")

Example Input:
"Alice (alice@example.com) said that Project Alpha is due on Friday."

Example Output:
("Alice", "PERSON", "PARTICIPATED_IN", "Project Alpha", "PROJECT", "Alice discussed Project Alpha")
("Project Alpha", "PROJECT", "HAS_DEADLINE", "Friday", "DATE", "Due on Friday")

Input Text:
{input_text}

Output:
"""


class GraphExtractor:
    """
    Extracts a knowledge graph from text using LLM.
    """

    def __init__(self) -> None:
        self.llm = LLMRuntime()

    def extract_graph(self, text: str) -> nx.Graph:
        """
        Extract entities and relations from text and return a NetworkX graph.
        """
        if not text or not text.strip():
            return nx.Graph()

        # 1. LLM Generation
        prompt = GRAPH_EXTRACTION_PROMPT.format(
            input_text=text[:20000]
        )  # Safety truncate
        try:
            # High temperature for creativity? No, low for extraction stability.
            response = self.llm.complete_text(prompt, temperature=0.1, max_tokens=2000)
        except Exception as e:
            logger.error(f"Graph extraction failed during LLM call: {e}")
            return nx.Graph()

        # 2. Parse Output
        return self._parse_tuples_to_graph(response)

    def _parse_tuples_to_graph(self, response: str) -> nx.Graph:
        """
        Parses LLM output tuples into a NetworkX graph.
        Expected format: ("A", "TypeA", "REL", "B", "TypeB", "Desc")
        """
        G = nx.Graph()

        # Regex to find tuples like ("...", "...", ...)
        # We assume values are quoted strings.
        # This regex matches: ( "val", "val", ... )
        tuple_pattern = re.compile(
            r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)'
        )

        for match in tuple_pattern.finditer(response):
            try:
                src_name, src_type, rel, tgt_name, tgt_type, desc = match.groups()

                # Add Source Node
                G.add_node(src_name, type=src_type)

                # Add Target Node
                G.add_node(tgt_name, type=tgt_type)

                # Add Edge
                # NetworkX allows multiple edges between nodes? Graph() does NOT (MultiGraph does).
                # We overwrite if exists, or append descriptions?
                # For simplicity, we overwrite or merge description.
                if G.has_edge(src_name, tgt_name):
                    old_desc = G[src_name][tgt_name].get("description", "")
                    new_desc = old_desc + "; " + desc
                    G.add_edge(src_name, tgt_name, relation=rel, description=new_desc)
                else:
                    G.add_edge(src_name, tgt_name, relation=rel, description=desc)

            except Exception as e:
                logger.warning(f"Failed to parse tuple match: {match.group(0)} -> {e}")
                continue

        return G
