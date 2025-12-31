"""
Graph RAG Extractor.

Extracts Knowledge Graph entities and relationships from text using LLM.
Optimized for Reasoning Models (e.g. openai-gpt-oss-120b), utilizing
sliding window chunking and graph merging for long contexts.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import networkx as nx
from cortex.llm.async_runtime import (
    AsyncLLMRuntime,
    LLMOutputSchemaError,
    ProviderError,
)
from cortex.llm.async_runtime import ValidationError as LLMValidationError

# Constants
DEFAULT_CHUNK_SIZE = 8000
DEFAULT_OVERLAP = 500


logger = logging.getLogger(__name__)


# --- Prompts (Reasoning Model Optimized: System + XML Structure) ---

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
    if not isinstance(relation, str):
        return "RELATED_TO"
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
        self.chunk_size = DEFAULT_CHUNK_SIZE
        self.overlap = DEFAULT_OVERLAP

    async def extract_graph(self, text: str) -> nx.DiGraph:
        """
        Extract entities and relations from text and return a NetworkX graph.
        Handles long text via chunking and merging.
        """
        if not text or not text.strip():
            return nx.DiGraph()

        # 1. Chunking Strategy
        chunks = self._chunk_text(text)

        logger.info(
            "Extracting graph from %d chars (Chunks: %d)", len(text), len(chunks)
        )

        # 2. Extract Sub-Graphs sequentially to prevent OOM
        sub_graphs = []
        for i, chunk in enumerate(chunks):
            sub_graph = await self._process_chunk(chunk, i)
            sub_graphs.append(sub_graph)

        # 3. Merge Graphs
        # Run the CPU-bound merge operation in a thread pool to avoid blocking the event loop.
        loop = asyncio.get_running_loop()
        try:
            final_G = await loop.run_in_executor(None, self._merge_graphs, sub_graphs)
        except Exception:
            logger.exception("Graph merge failed; returning unmerged subgraphs")
            final_G = nx.compose_all(sub_graphs) if sub_graphs else nx.DiGraph()
        return final_G

    def _chunk_text(self, text: str) -> list[str]:
        """Simple sliding window chunker."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        chunk_size = max(1, int(self.chunk_size))
        overlap = max(0, min(int(self.overlap), chunk_size - 1))
        if overlap != self.overlap:
            logger.warning(
                "Overlap adjusted to %d to fit chunk_size %d", overlap, chunk_size
            )
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Try to break at newline if possible to respect boundaries
            if end < len(text):
                # Search for last newline in the overlap area
                last_newline = text.rfind("\n", start, end)
                if last_newline > start + chunk_size // 2:  # Don't backtrack too much
                    end = last_newline

            if end <= start:
                end = min(start + chunk_size, len(text))

            chunks.append(text[start:end])
            if end == len(text):
                break
            next_start = max(0, end - overlap)
            if next_start <= start:
                next_start = end
            start = next_start
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
        except (ProviderError, LLMOutputSchemaError, LLMValidationError) as e:
            logger.warning(
                "Graph extraction failed for chunk %d: %s",
                index,
                e,
                exc_info=True,
            )
            return nx.DiGraph()

    def _parse_json_to_graph(self, data: dict[str, Any]) -> nx.DiGraph:
        """Parses Validated JSON into a NetworkX graph."""
        G = nx.DiGraph()
        if not isinstance(data, dict):
            return G

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        if not isinstance(nodes, list):
            nodes = []
        if not isinstance(edges, list):
            edges = []

        for node in nodes:
            if not isinstance(node, dict):
                continue
            name = node.get("name")
            if not isinstance(name, str):
                continue
            name = name.strip()
            if not name:
                continue
            node_type = node.get("type", "UNKNOWN")
            if not isinstance(node_type, str):
                node_type = "UNKNOWN"
            node_type = node_type.strip().upper()
            if node_type not in VALID_NODE_TYPES:
                node_type = "UNKNOWN"
            properties = node.get("properties", {})
            if not isinstance(properties, dict):
                properties = {}
            G.add_node(name, type=node_type, properties=properties)

        for edge in edges:
            if not isinstance(edge, dict):
                continue
            src = edge.get("source")
            tgt = edge.get("target")
            if not isinstance(src, str) or not isinstance(tgt, str):
                continue
            src = src.strip()
            tgt = tgt.strip()
            if not src or not tgt:
                continue

            if not G.has_node(src):
                G.add_node(src, type="UNKNOWN", properties={})
            if not G.has_node(tgt):
                G.add_node(tgt, type="UNKNOWN", properties={})

            relation = _normalize_relation(edge.get("relation"))

            G.add_edge(
                src,
                tgt,
                relation=relation,
                description=edge.get("description", ""),
                source_chunk="llm",
            )
        return G

    def _merge_graphs(self, graphs: list[nx.DiGraph]) -> nx.DiGraph:
        """
        Merges multiple subgraphs into one master graph using an optimized
        clustering approach for case-insensitive + variant name matching.
        """
        if not graphs:
            return nx.DiGraph()

        # 1. Collect all unique nodes and their attributes from all subgraphs
        all_nodes: dict[str, dict[str, Any]] = {}
        for G in graphs:
            for node, attrs in G.nodes(data=True):
                if node not in all_nodes:
                    all_nodes[node] = dict(attrs)
                else:
                    existing_attrs = all_nodes[node]
                    existing_type = existing_attrs.get("type", "UNKNOWN")
                    new_type = attrs.get("type", "UNKNOWN")
                    if (
                        existing_type == "UNKNOWN"
                        and new_type
                        and new_type != "UNKNOWN"
                    ):
                        existing_attrs["type"] = new_type
                    elif (
                        new_type
                        and new_type != "UNKNOWN"
                        and existing_type not in ("UNKNOWN", new_type)
                    ):
                        variants = set(existing_attrs.get("type_variants", []))
                        variants.update([existing_type, new_type])
                        existing_attrs["type_variants"] = sorted(variants)

                    existing_props = existing_attrs.get("properties", {})
                    if not isinstance(existing_props, dict):
                        existing_props = {}
                    new_props = attrs.get("properties", {})
                    if isinstance(new_props, dict):
                        existing_props.update(new_props)
                    existing_attrs["properties"] = existing_props

        # 2. Group nodes by normalized name (case-insensitive) - O(n) instead of O(n²)
        node_names = list(all_nodes.keys())
        normalized_names = {name: name.strip().lower() for name in node_names}
        canonical_map: dict[str, str] = {}  # lowercase -> canonical (longest version)

        for name in node_names:
            normalized = normalized_names[name]
            if normalized not in canonical_map:
                canonical_map[normalized] = name
            elif len(name) > len(canonical_map[normalized]):
                canonical_map[normalized] = name

        # Merge shorter variants into longer canonical names when they are whole-word substrings.
        names_by_length = sorted(node_names, key=len, reverse=True)
        variant_map: dict[str, str] = {}
        for short_name in sorted(node_names, key=len):
            short_norm = normalized_names[short_name]
            if not short_norm:
                continue
            for long_name in names_by_length:
                if len(long_name) <= len(short_name):
                    continue
                long_norm = normalized_names[long_name]
                if f" {short_norm} " in f" {long_norm} ":
                    variant_map[short_name] = long_name
                    break

        # Build node_map: each node -> its canonical version
        node_map: dict[str, str] = {}
        for name in node_names:
            normalized = normalized_names[name]
            canonical = canonical_map[normalized]
            node_map[name] = variant_map.get(name, canonical)

        # 3. Build the final graph
        final_G = nx.DiGraph()

        canonical_attrs: dict[str, dict[str, Any]] = {}
        for name, attrs in all_nodes.items():
            canonical_name = node_map[name]
            existing = canonical_attrs.get(canonical_name, {})
            merged = dict(existing)
            existing_type = merged.get("type", "UNKNOWN")
            new_type = attrs.get("type", "UNKNOWN")
            if existing_type == "UNKNOWN" and new_type != "UNKNOWN":
                merged["type"] = new_type
            elif (
                new_type
                and new_type != "UNKNOWN"
                and existing_type not in ("UNKNOWN", new_type)
            ):
                variants = set(merged.get("type_variants", []))
                variants.update([existing_type, new_type])
                merged["type_variants"] = sorted(variants)

            merged_props = merged.get("properties", {})
            if not isinstance(merged_props, dict):
                merged_props = {}
            new_props = attrs.get("properties", {})
            if isinstance(new_props, dict):
                merged_props.update(new_props)
            merged["properties"] = merged_props

            canonical_attrs[canonical_name] = merged

        for canonical_name, attrs in canonical_attrs.items():
            final_G.add_node(canonical_name, **attrs)

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
                    existing_relation = final_G[src][tgt].get("relation")
                    new_relation = attrs.get("relation")
                    if (
                        new_relation
                        and existing_relation
                        and new_relation != existing_relation
                    ):
                        variants = set(final_G[src][tgt].get("relation_variants", []))
                        variants.update([existing_relation, new_relation])
                        final_G[src][tgt]["relation_variants"] = sorted(variants)
                        if (
                            existing_relation == "RELATED_TO"
                            and new_relation != "RELATED_TO"
                        ):
                            final_G[src][tgt]["relation"] = new_relation
                    elif new_relation and not existing_relation:
                        final_G[src][tgt]["relation"] = new_relation
                    if new_desc not in existing_desc:
                        final_G[src][tgt]["description"] = (
                            f"{existing_desc}; {new_desc}"
                        ).strip("; ")
                else:
                    final_G.add_edge(src, tgt, **attrs)

        return final_G
