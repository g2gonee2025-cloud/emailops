"""
Optimized Graph Backfill for Cortex CLI.

This module contains the core logic for backfilling the knowledge graph from
conversation summaries. It includes performance optimizations like batch
processing to avoid N+1 query problems and validation against the graph schema.
"""

from __future__ import annotations

import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Set, Tuple

import networkx as nx
from cortex.db.models import Conversation, EntityEdge, EntityNode
from cortex.db.session import SessionLocal
from cortex.intelligence.graph import (
    GRAPH_EXTRACTION_PROMPT,
    VALID_NODE_TYPES,
    VALID_RELATIONS,
    _normalize_relation,
)
from sqlalchemy import select

logger = logging.getLogger(__name__)

# --- Configuration ---
MODELS = [
    "openai-gpt-oss-120b",
    "openai-gpt-5",
    "deepseek-r1-distill-llama-70b",
]
CONVERSATION_BATCH_SIZE = 500  # Number of conversations to fetch at once


# --- Graph Extraction ---
class GraphExtractorWithFallback:
    """Graph extractor with model fallback on rate limit."""

    def __init__(self) -> None:
        from openai import OpenAI

        base_url = os.getenv("LLM_ENDPOINT", "https://inference.do-ai.run/v1")
        api_key = os.getenv("LLM_API_KEY", os.getenv("DO_LLM_API_KEY", "EMPTY"))
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.current_model_idx = 0

    def extract_graph(self, text: str) -> nx.DiGraph:
        """Extract graph with model fallback."""
        if not text or len(text.strip()) < 50:
            return nx.DiGraph()

        prompt = GRAPH_EXTRACTION_PROMPT.format(input_text=text)

        for i in range(len(MODELS)):
            model = MODELS[(self.current_model_idx + i) % len(MODELS)]
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=4000,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                if content.strip():
                    import json

                    return self._parse_json_to_graph(json.loads(content))
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    logger.warning(f"Model {model} rate limited, trying next.")
                    self.current_model_idx = (self.current_model_idx + 1) % len(MODELS)
                else:
                    logger.debug(f"Model {model} failed: {e}")
        return nx.DiGraph()

    def _parse_json_to_graph(self, data: dict[str, Any]) -> nx.DiGraph:
        G = nx.DiGraph()
        for node_data in data.get("nodes", []):
            name = node_data.get("name", "").strip()
            if not name:
                continue
            node_type = node_data.get("type", "UNKNOWN").strip().upper()
            if node_type not in VALID_NODE_TYPES:
                node_type = "UNKNOWN"
            G.add_node(name, type=node_type, properties=node_data.get("properties", {}))

        for edge_data in data.get("edges", []):
            src, tgt = (
                edge_data.get("source", "").strip(),
                edge_data.get("target", "").strip(),
            )
            if not src or not tgt:
                continue
            if not G.has_node(src):
                G.add_node(src, type="UNKNOWN", properties={})
            if not G.has_node(tgt):
                G.add_node(tgt, type="UNKNOWN", properties={})

            relation = _normalize_relation(edge_data.get("relation", "RELATED_TO"))
            if relation not in VALID_RELATIONS:
                logger.debug(f"Skipping invalid relation: {relation}")
                continue

            G.add_edge(
                src,
                tgt,
                relation=relation,
                description=edge_data.get("description", ""),
            )
        return G


# --- Database Operations ---
def get_conversations_without_graphs(
    tenant_id: str | None, limit: int | None
) -> List[Tuple[uuid.UUID, str, str]]:
    """Fetch conversations with summaries but no graph edges in batches."""
    conversations = []
    offset = 0
    with SessionLocal() as session:
        while True:
            has_edges_subq = select(EntityEdge.conversation_id).distinct()
            stmt = (
                select(
                    Conversation.conversation_id,
                    Conversation.tenant_id,
                    Conversation.summary_text,
                )
                .where(
                    Conversation.summary_text.isnot(None),
                    Conversation.summary_text != "",
                    ~Conversation.conversation_id.in_(has_edges_subq),
                )
                .offset(offset)
                .limit(CONVERSATION_BATCH_SIZE)
            )
            if tenant_id:
                stmt = stmt.where(Conversation.tenant_id == tenant_id)
            if limit and len(conversations) + CONVERSATION_BATCH_SIZE > limit:
                stmt = stmt.limit(limit - len(conversations))

            results = session.execute(stmt).all()
            if not results:
                break
            conversations.extend([(r[0], r[1], r[2]) for r in results])
            offset += CONVERSATION_BATCH_SIZE
            if limit and len(conversations) >= limit:
                break
    return conversations


def process_conversation_batch(
    batch: List[Tuple[uuid.UUID, str, str]],
    extractor: GraphExtractorWithFallback,
    dry_run: bool,
) -> Dict[str, Any]:
    """Process a batch of conversations, optimizing DB lookups."""
    batch_results = {
        "nodes_created": 0,
        "nodes_reused": 0,
        "edges_created": 0,
        "success": 0,
        "failed": 0,
    }
    graphs_to_persist = []

    # Step 1: Extract graphs concurrently
    for conv_id, tenant_id, summary in batch:
        if not summary or len(summary.strip()) < 50:
            batch_results["failed"] += 1
            continue
        G = extractor.extract_graph(summary)
        if G.number_of_nodes() > 0:
            graphs_to_persist.append(
                {"graph": G, "conv_id": conv_id, "tenant_id": tenant_id}
            )
            batch_results["success"] += 1
        else:
            batch_results["failed"] += 1

    if dry_run:
        for item in graphs_to_persist:
            G = item["graph"]
            batch_results["nodes_created"] += G.number_of_nodes()
            batch_results["edges_created"] += G.number_of_edges()
        return batch_results

    # Step 2: Batch DB operations
    with SessionLocal() as session:
        all_node_names: Set[str] = {
            name for item in graphs_to_persist for name in item["graph"].nodes
        }
        tenant_ids: Set[str] = {item["tenant_id"] for item in graphs_to_persist}

        # Pre-fetch existing nodes for this batch
        existing_nodes_q = select(EntityNode).where(
            EntityNode.tenant_id.in_(tenant_ids), EntityNode.name.in_(all_node_names)
        )
        existing_nodes = {
            (node.tenant_id, node.name): node
            for node in session.scalars(existing_nodes_q)
        }

        # Process each graph
        for item in graphs_to_persist:
            G, conv_id, tenant_id = item["graph"], item["conv_id"], item["tenant_id"]
            node_map = {}

            # Create/map nodes
            for name, attrs in G.nodes(data=True):
                node = existing_nodes.get((tenant_id, name))
                if not node:
                    node = EntityNode(
                        tenant_id=tenant_id,
                        name=name,
                        type=attrs.get("type", "UNKNOWN"),
                        description=f"From conversation {conv_id}",
                        properties=attrs.get("properties", {}),
                    )
                    session.add(node)
                    session.flush()  # To get node_id
                    existing_nodes[(tenant_id, name)] = node
                    batch_results["nodes_created"] += 1
                else:
                    batch_results["nodes_reused"] += 1
                node_map[name] = node.node_id

            # Create edges
            for src, dst, edge_attrs in G.edges(data=True):
                if node_map.get(src) and node_map.get(dst):
                    edge = EntityEdge(
                        tenant_id=tenant_id,
                        source_id=node_map[src],
                        target_id=node_map[dst],
                        relation=edge_attrs.get("relation", "RELATED_TO"),
                        description=edge_attrs.get("description", ""),
                        conversation_id=conv_id,
                        weight=1.0,
                    )
                    session.add(edge)
                    batch_results["edges_created"] += 1
        session.commit()
    return batch_results


# --- Main Runner ---
def run_backfill_graph(
    tenant_id: str | None,
    max_workers: int,
    limit: int | None,
    dry_run: bool,
    progress_callback: Callable | None,
):
    """Run the graph backfill using summaries with optimizations."""
    logger.info(
        f"Starting Optimized Graph Backfill. Workers: {max_workers}, "
        f"Limit: {limit or 'all'}, Dry run: {dry_run}"
    )

    conversations = get_conversations_without_graphs(tenant_id, limit)
    total = len(conversations)
    if total == 0:
        logger.info("No conversations found needing graphs.")
        return {"total": 0, "success": 0, "failed": 0}

    logger.info(f"Found {total} conversations to process.")
    extractor = GraphExtractorWithFallback()
    results = {
        "total": total,
        "success": 0,
        "failed": 0,
        "nodes_created": 0,
        "nodes_reused": 0,
        "edges_created": 0,
    }

    # Process in batches
    batches = [
        conversations[i : i + CONVERSATION_BATCH_SIZE]
        for i in range(0, total, CONVERSATION_BATCH_SIZE)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Associate each future with its batch size for accurate error reporting
        futures = {
            executor.submit(process_conversation_batch, batch, extractor, dry_run): len(
                batch
            )
            for batch in batches
        }

        for future in as_completed(futures):
            batch_size = futures[future]
            try:
                batch_result = future.result()
                results["success"] += batch_result["success"]
                results["failed"] += batch_result["failed"]
                results["nodes_created"] += batch_result["nodes_created"]
                results["nodes_reused"] += batch_result["nodes_reused"]
                results["edges_created"] += batch_result["edges_created"]
            except Exception as e:
                logger.error(f"A batch of size {batch_size} failed: {e}")
                results["failed"] += batch_size
            finally:
                if progress_callback:
                    progress_callback(results)

    return results
