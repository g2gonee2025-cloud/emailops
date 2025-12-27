"""
Graph RAG Extractor.

Extracts Knowledge Graph entities and relationships from text using LLM.
Optimized for Reasoning Models (e.g. openai-gpt-oss-120b), utilizing
sliding window chunking and graph merging for long contexts.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, List

import networkx as nx
from cortex.llm.async_runtime import AsyncLLMRuntime
from thefuzz import fuzz, process

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

SYSTEM_PROMPT = """
You are an expert Data Analyst specializing in Email Operations for insurance and corporate communications.
Extract a structured Knowledge Graph from the provided email conversation text.
Identify key entities (Nodes) and their specific relationships (Edges).
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
        self.llm = AsyncLLMRuntime()
        self.chunk_size = 8000  # Conservative chunk size for reasoning models (leaving room for output)
        self.overlap = 500

    async def extract_graph(self, text: str) -> nx.DiGraph:
        """
        Extract entities and relations from text and return a NetworkX graph.
        Handles long text via chunking and merging.
        """
        if not text or not text.strip():
            return nx.DiGraph()

        # 1. Chunking Strategy
        chunks = self._chunk_text(text)

        logger.info(f"Extracting graph from {len(text)} chars (Chunks: {len(chunks)})")

        # 2. Extract Sub-Graphs concurrently
        tasks = [self._process_chunk(chunk, i) for i, chunk in enumerate(chunks)]
        sub_graphs = await asyncio.gather(*tasks)

        # 3. Merge Graphs
        # Run the CPU-bound merge operation in a thread pool to avoid blocking the event loop.
        loop = asyncio.get_running_loop()
        final_G = await loop.run_in_executor(None, self._merge_graphs, sub_graphs)
        return final_G

    def _chunk_text(self, text: str) -> list[str]:
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

    async def _process_chunk(self, chunk_text: str, index: int) -> nx.DiGraph:
        """Call LLM for a single chunk."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": GRAPH_EXTRACTION_PROMPT.format(input_text=chunk_text),
            },
        ]

        try:
            response_json = await self.llm.async_complete_json(
                prompt=messages,
                schema=GRAPH_SCHEMA,
                temperature=0.1,
                max_tokens=4000,
            )
            return self._parse_json_to_graph(response_json)
        except Exception as e:
            logger.warning(f"Graph extraction failed for chunk {index}: {e}")
            return nx.DiGraph()

    def _parse_json_to_graph(self, data: dict[str, Any]) -> nx.DiGraph:
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

    def _merge_graphs(self, graphs: list[nx.DiGraph]) -> nx.DiGraph:
        """
        Merges multiple subgraphs into one master graph using an optimized
        clustering approach for fuzzy node matching.
        """
        if not graphs:
            return nx.DiGraph()

        # 1. Collect all unique nodes and their attributes from all subgraphs
        all_nodes = {}
        for G in graphs:
            for node, attrs in G.nodes(data=True):
                if node not in all_nodes:
                    all_nodes[node] = attrs
                else:
                    # Merge attributes, prioritizing specific types over UNKNOWN
                    existing_attrs = all_nodes[node]
                    if (
                        existing_attrs.get("type", "UNKNOWN") == "UNKNOWN"
                        and attrs.get("type", "UNKNOWN") != "UNKNOWN"
                    ):
                        existing_attrs["type"] = attrs.get("type")

                    existing_props = existing_attrs.get("properties", {})
                    new_props = attrs.get("properties", {})
                    existing_props.update(new_props)
                    existing_attrs["properties"] = existing_props

        # 2. Efficiently cluster nodes using a blocking strategy to avoid O(N^2)
        node_names = list(all_nodes.keys())
        blocks = defaultdict(list)
        for name in node_names:
            # Use a simple blocking key (e.g., first word, case-insensitive)
            key = name.split()[0].lower() if name else ""
            blocks[key].append(name)

        node_map = {}  # Maps each node to its canonical representative
        processed_nodes = set()

        for block_nodes in blocks.values():
            for node in block_nodes:
                if node in processed_nodes:
                    continue

                # Find similar nodes only within the smaller block
                matches = process.extractBests(node, block_nodes, score_cutoff=80)
                if not matches:
                    processed_nodes.add(node)
                    node_map[node] = node
                    continue

                cluster = [match[0] for match in matches]
                canonical = max(cluster, key=len)
                for member in cluster:
                    node_map[member] = canonical
                    processed_nodes.add(member)

        # 3. Build the final graph
        final_G = nx.DiGraph()

        # Add canonical nodes to the graph
        for canonical_name in set(node_map.values()):
            final_G.add_node(canonical_name, **all_nodes[canonical_name])

        # Add and remap edges
        for G in graphs:
            for u, v, attrs in G.edges(data=True):
                src = node_map.get(u, u)
                tgt = node_map.get(v, v)

                if src == tgt:  # Avoid self-loops from merging
                    continue

                if final_G.has_edge(src, tgt):
                    existing_desc = final_G[src][tgt].get("description", "")
                    new_desc = attrs.get("description", "")
                    if new_desc not in existing_desc:
                        final_G[src][tgt]["description"] = (
                            f"{existing_desc}; {new_desc}"
                        ).strip("; ")
                else:
                    final_G.add_edge(src, tgt, **attrs)

        return final_G
