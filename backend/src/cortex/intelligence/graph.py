"""
Graph RAG Extractor.

Extracts Knowledge Graph entities and relationships from text using LLM.
Optimized for Reasoning Models (e.g. openai-gpt-oss-120b), utilizing
sliding window chunking and graph merging for long contexts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import networkx as nx
from cortex.llm.runtime import LLMRuntime

# Constants
DEFAULT_CHUNK_SIZE = 8000
DEFAULT_OVERLAP = 500


logger = logging.getLogger(__name__)


# --- Prompts (Reasoning Model Optimized: No System Prompt, XML Structure) ---

# Valid entity types for schema enforcement
VALID_NODE_TYPES = [
    "PERSON",
    "ORGANIZATION",
    "DOCUMENT",
    "POLICY",
    "CLAIM",
    "MONEY",
    "DATE",
    "LOCATION",
    "ACTION_ITEM",
    "STATUS",
    "EVENT",
]

# Valid relation types - specific, not generic
VALID_RELATIONS = [
    # Email communication
    "SENT_EMAIL_TO",
    "CC_TO",
    "FORWARDED_TO",
    # Organizational
    "WORKS_FOR",
    "LOCATED_IN",
    "DEPARTMENT_OF",
    # Insurance/Document specific
    "OWNS_POLICY",
    "ISSUED_BY",
    "COVERED_BY",
    "CLAIMS_ON",
    # Financial
    "HAS_VALUE",
    "OWES",
    "PAID_BY",
    # Temporal
    "DUE_ON",
    "VALID_UNTIL",
    "EFFECTIVE_FROM",
    # Status/State
    "CURRENT_STATUS",
    "CHANGED_TO",
    # Actions
    "ASSIGNED_TO",
    "REQUESTED_BY",
    "APPROVED_BY",
    # Document relations
    "ATTACHED_TO",
    "REFERENCES",
    "AMENDS",
    # Fallback (use sparingly)
    "RELATED_TO",
]

GRAPH_EXTRACTION_PROMPT = """
<instructions>
You are an expert Data Analyst specializing in Email Operations for insurance and corporate communications.
Extract a structured Knowledge Graph from the provided email conversation text.
Identify key entities (Nodes) and their specific relationships (Edges).
</instructions>

<ontology>
NODE TYPES (Use ONLY these types):
- PERSON: Individual people with names. Extract email if visible.
- ORGANIZATION: Companies, insurers, departments, vendors.
- DOCUMENT: Invoices, policies, contracts, POs, reports. Include reference numbers.
- POLICY: Insurance policies specifically. Include policy number.
- CLAIM: Insurance claims. Include claim number.
- MONEY: Monetary amounts. Always include currency (SAR, AED, USD, etc.).
- DATE: Specific dates or deadlines.
- LOCATION: Cities, countries, offices, ports.
- ACTION_ITEM: Tasks or follow-ups required.
- STATUS: Business states (Pending, Approved, Overdue, Paid, Rejected).
- EVENT: Meetings, renewals, deadlines as events.

EDGE RELATIONS (Use SPECIFIC relations, avoid generic):
Communication: SENT_EMAIL_TO, CC_TO, FORWARDED_TO
Organizational: WORKS_FOR, LOCATED_IN, DEPARTMENT_OF
Insurance: OWNS_POLICY, ISSUED_BY, COVERED_BY, CLAIMS_ON
Financial: HAS_VALUE, OWES, PAID_BY
Temporal: DUE_ON, VALID_UNTIL, EFFECTIVE_FROM
Status: CURRENT_STATUS, CHANGED_TO
Actions: ASSIGNED_TO, REQUESTED_BY, APPROVED_BY
Documents: ATTACHED_TO, REFERENCES, AMENDS
Generic: RELATED_TO (use ONLY if no specific relation fits)
</ontology>

<examples>
Example: "Invoice from Acme Corp for $5,000, due Dec 15. - Sarah"
{{
  "nodes": [
    {{"name": "Sarah", "type": "PERSON", "properties": {{}}}},
    {{"name": "Acme Corp", "type": "ORGANIZATION", "properties": {{}}}},
    {{"name": "Invoice", "type": "DOCUMENT", "properties": {{}}}},
    {{"name": "USD 5,000", "type": "MONEY", "properties": {{"currency": "USD"}}}}
  ],
  "edges": [
    {{"source": "Invoice", "target": "Acme Corp", "relation": "ISSUED_BY", "description": "Vendor"}},
    {{"source": "Invoice", "target": "USD 5,000", "relation": "HAS_VALUE", "description": "Amount"}}
  ]
}}
</examples>

<rules>
1. Output MUST be valid JSON conforming to the schema.
2. De-duplicate entities: "Alice" and "Alice Smith" → use "Alice Smith".
3. DO NOT use RELATED_TO if a more specific relation exists.
4. Always extract PERSON entities from email senders/recipients.
5. For MONEY, always include currency in the name (e.g., "SAR 29,518" not just "29,518").
6. For POLICY/CLAIM, include reference numbers in the name.
7. Add meaningful properties: email addresses, roles, currency codes, amounts.
8. Be precise. Do not hallucinate entities not in the text.
</rules>

<text>
{input_text}
</text>
"""

GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "properties": {
                        "type": "object",
                        "description": "Additional metadata like email, role, currency, amount, etc.",
                    },
                },
                "required": ["name", "type"],
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["source", "target", "relation", "description"],
            },
        },
    },
    "required": ["nodes", "edges"],
}


def _normalize_relation(relation: str) -> str:
    """Normalize relation to UPPER_SNAKE_CASE and validate against ontology."""
    normalized = relation.strip().upper().replace(" ", "_").replace("-", "_")
    # Map common variations that are semantically equivalent and safe
    relation_map = {
        "MENTIONS": "REFERENCES",
        "DISCUSSES": "REFERENCES",
        "SENT_TO": "SENT_EMAIL_TO",
        "HAS_DEADLINE": "DUE_ON",
        "REQUIRES_ACTION_FROM": "ASSIGNED_TO",
        "AMOUNT_OF": "HAS_VALUE",
        "HAS_STATUS": "CURRENT_STATUS",
        "ATTACHES": "ATTACHED_TO",
    }
    final_relation = relation_map.get(normalized, normalized)

    # Validate against the official list
    if final_relation not in VALID_RELATIONS:
        logger.warning(
            f"LLM-extracted relation '{relation}' (normalized to '{final_relation}') "
            "is not in the ontology. Falling back to 'RELATED_TO'."
        )
        return "RELATED_TO"

    return final_relation


class GraphExtractor:
    """
    Extracts a knowledge graph from text using LLM with structured JSON output.
    Supports sliding window chunking for long texts.
    """

    def __init__(self) -> None:
        self.llm = LLMRuntime()
        self.chunk_size = 8000  # Conservative chunk size for reasoning models (leaving room for output)
        self.overlap = 500

    def extract_graph(self, text: str) -> nx.DiGraph:
        """
        Extract entities and relations from text and return a NetworkX graph.
        Handles long text via chunking and merging.
        """
        if not text or not text.strip():
            return nx.DiGraph()

        # 1. Chunking Strategy
        chunks = self._chunk_text(text)
        sub_graphs = []

        logger.info(f"Extracting graph from {len(text)} chars (Chunks: {len(chunks)})")

        # 2. Extract Sub-Graphs
        for i, chunk in enumerate(chunks):
            sub_G = self._process_chunk(chunk, i)
            sub_graphs.append(sub_G)

        # 3. Merge Graphs
        final_G = self._merge_graphs(sub_graphs)
        return final_G

    def _chunk_text(self, text: str) -> List[str]:
        """Simple sliding window chunker."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            # Try to break at newline if possible to respect boundaries
            if end < len(text):
                # Search for last newline in the overlap area
                last_newline = text.rfind("\n", start, end)
                if (
                    last_newline > start + self.chunk_size // 2
                ):  # Don't backtrack too much
                    end = last_newline

            chunks.append(text[start:end])
            start = end - self.overlap
        return chunks

    def _process_chunk(self, chunk_text: str, index: int) -> nx.DiGraph:
        """Call LLM for a single chunk."""
        prompt = GRAPH_EXTRACTION_PROMPT.format(input_text=chunk_text)

        try:
            response_json = self.llm.complete_json(
                prompt,
                schema=GRAPH_SCHEMA,
                temperature=0.1,
                max_tokens=4000,
            )
            return self._parse_json_to_graph(response_json)
        except Exception as e:
            logger.warning(f"Graph extraction failed for chunk {index}: {e}")
            return nx.DiGraph()

    def _parse_json_to_graph(self, data: Dict[str, Any]) -> nx.DiGraph:
        """Parses Validated JSON into a NetworkX graph."""
        G = nx.DiGraph()
        try:
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])

            for node in nodes:
                # Basic cleaning: strip whitespace
                name = node["name"].strip()
                node_type = node.get("type", "UNKNOWN").strip().upper()
                # Validate type
                if node_type not in VALID_NODE_TYPES:
                    node_type = "UNKNOWN"
                # Extract properties
                properties = node.get("properties", {})
                G.add_node(name, type=node_type, properties=properties)

            for edge in edges:
                src = edge["source"].strip()
                tgt = edge["target"].strip()

                # Ensure nodes exist
                if not G.has_node(src):
                    G.add_node(src, type="UNKNOWN", properties={})
                if not G.has_node(tgt):
                    G.add_node(tgt, type="UNKNOWN", properties={})

                # Normalize relation
                relation = _normalize_relation(edge.get("relation", "RELATED_TO"))

                G.add_edge(
                    src,
                    tgt,
                    relation=relation,
                    description=edge.get("description", ""),
                    source_chunk="llm",  # Metadata
                )
        except Exception as e:
            logger.warning(f"Failed to parse graph JSON: {e}")
        return G

    def _merge_graphs(self, graphs: List[nx.DiGraph]) -> nx.DiGraph:
        """Merges multiple subgraphs into one master graph."""
        if not graphs:
            return nx.DiGraph()

        final_G = nx.DiGraph()

        for G in graphs:
            # Merge Nodes
            for node, attrs in G.nodes(data=True):
                if not final_G.has_node(node):
                    final_G.add_node(node, **attrs)
                else:
                    # Update type if UNKNOWN -> something specific
                    current_type = final_G.nodes[node].get("type", "UNKNOWN")
                    new_type = attrs.get("type", "UNKNOWN")
                    if current_type == "UNKNOWN" and new_type != "UNKNOWN":
                        final_G.nodes[node]["type"] = new_type

                    # Merge properties dictionaries
                    current_props = final_G.nodes[node].get("properties", {})
                    new_props = attrs.get("properties", {})
                    if new_props:
                        current_props.update(new_props)
                        final_G.nodes[node]["properties"] = current_props

            # Merge Edges
            for u, v, attrs in G.edges(data=True):
                if final_G.has_edge(u, v):
                    # Combine descriptions
                    existing_desc = final_G[u][v].get("description", "")
                    new_desc = attrs.get("description", "")
                    if new_desc not in existing_desc:
                        final_G[u][v]["description"] = (
                            existing_desc + "; " + new_desc
                        ).strip("; ")
                else:
                    final_G.add_edge(u, v, **attrs)

        return final_G
