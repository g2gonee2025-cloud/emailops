#!/usr/bin/env python3
"""
Fast Graph Backfill using Conversation Summaries with Model Fallback.

Uses summary_text from conversations table instead of reconstructing from chunks.
Falls back to gpt-5 when primary model hits rate limits.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Add backend to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
if BACKEND_SRC.exists():
    sys.path.insert(0, str(BACKEND_SRC))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

import networkx as nx  # noqa: E402
from cortex.db.models import Conversation, EntityEdge, EntityNode  # noqa: E402
from cortex.db.session import SessionLocal  # noqa: E402
from cortex.intelligence.graph import (  # noqa: E402
    GRAPH_EXTRACTION_PROMPT,
    VALID_NODE_TYPES,
    _normalize_relation,
)
from sqlalchemy import select  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "graph_backfill_fast.log"),
    ],
)
# Silence noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("cortex.llm.runtime").setLevel(logging.WARNING)

logger = logging.getLogger("graph_backfill_fast")

# Models to try in order (from DO Inference API)
MODELS = [
    "openai-gpt-oss-120b",  # Primary - fast, cheap
    "openai-gpt-5",  # Fallback 1 - powerful
    "deepseek-r1-distill-llama-70b",  # Fallback 2 - reasoning
]


class GraphExtractorWithFallback:
    """Graph extractor with model fallback on rate limit."""

    def __init__(self) -> None:
        from openai import OpenAI

        base_url = os.getenv("LLM_ENDPOINT", "https://inference.do-ai.run/v1")
        api_key = os.getenv("LLM_API_KEY", os.getenv("DO_LLM_API_KEY", "EMPTY"))
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.current_model = 0  # Index into MODELS

    def extract_graph(self, text: str) -> nx.DiGraph:
        """Extract graph with model fallback."""
        if not text or len(text.strip()) < 50:
            return nx.DiGraph()

        prompt = GRAPH_EXTRACTION_PROMPT.format(input_text=text)

        # Try each model in order
        for model_idx in range(len(MODELS)):
            model = MODELS[(self.current_model + model_idx) % len(MODELS)]
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=4000,
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content or ""
                if not content.strip():
                    continue

                import json

                data = json.loads(content)
                return self._parse_json_to_graph(data)

            except Exception as e:
                error_str = str(e).lower()
                # Check for rate limit / circuit breaker errors
                if "rate" in error_str or "circuit" in error_str or "429" in error_str:
                    logger.warning(f"Model {model} rate limited, trying fallback...")
                    self.current_model = (self.current_model + 1) % len(MODELS)
                    continue
                else:
                    logger.debug(f"Model {model} failed: {e}")
                    continue

        return nx.DiGraph()

    def _parse_json_to_graph(self, data: dict[str, Any]) -> nx.DiGraph:
        """Parse JSON to NetworkX graph."""
        G = nx.DiGraph()
        try:
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])

            for node in nodes:
                name = node.get("name", "").strip()
                if not name:
                    continue
                node_type = node.get("type", "UNKNOWN").strip().upper()
                if node_type not in VALID_NODE_TYPES:
                    node_type = "UNKNOWN"
                properties = node.get("properties", {})
                G.add_node(name, type=node_type, properties=properties)

            for edge in edges:
                src = edge.get("source", "").strip()
                tgt = edge.get("target", "").strip()
                if not src or not tgt:
                    continue

                if not G.has_node(src):
                    G.add_node(src, type="UNKNOWN", properties={})
                if not G.has_node(tgt):
                    G.add_node(tgt, type="UNKNOWN", properties={})

                relation = _normalize_relation(edge.get("relation", "RELATED_TO"))
                G.add_edge(
                    src,
                    tgt,
                    relation=relation,
                    description=edge.get("description", ""),
                )
        except Exception as e:
            logger.debug(f"Failed to parse graph JSON: {e}")
        return G


def get_conversations_without_graphs(
    tenant_id: str | None = None, limit: int | None = None
) -> list[tuple[uuid.UUID, str, str]]:
    """Get conversations that have summaries but no graph edges."""
    with SessionLocal() as session:
        # Subquery to find conversations with edges
        has_edges_subq = (
            select(EntityEdge.conversation_id)
            .where(EntityEdge.conversation_id.isnot(None))
            .distinct()
        )

        # Get conversations with summaries but no edges
        stmt = select(
            Conversation.conversation_id,
            Conversation.tenant_id,
            Conversation.summary_text,
        ).where(
            Conversation.summary_text.isnot(None),
            Conversation.summary_text != "",
            ~Conversation.conversation_id.in_(has_edges_subq),
        )

        if tenant_id:
            stmt = stmt.where(Conversation.tenant_id == tenant_id)

        if limit:
            stmt = stmt.limit(limit)

        results = session.execute(stmt).all()
        return [(row[0], row[1], row[2]) for row in results]


def process_conversation(
    conversation_id: uuid.UUID,
    tenant_id: str,
    summary_text: str,
    extractor: GraphExtractorWithFallback,
    dry_run: bool = False,
) -> dict:
    """Process a single conversation using its summary."""
    result = {
        "conversation_id": str(conversation_id),
        "nodes_created": 0,
        "nodes_reused": 0,
        "edges_created": 0,
        "success": False,
        "error": None,
    }

    try:
        if not summary_text or len(summary_text.strip()) < 50:
            result["error"] = "Summary too short"
            return result

        # Extract graph from summary
        G = extractor.extract_graph(summary_text)
        if G.number_of_nodes() == 0:
            result["error"] = "No nodes extracted"
            return result

        if dry_run:
            result["nodes_created"] = G.number_of_nodes()
            result["edges_created"] = G.number_of_edges()
            result["success"] = True
            return result

        # Persist to DB
        with SessionLocal() as session:
            node_map = {}
            nodes_new = 0
            nodes_reused = 0

            for node_name, attrs in G.nodes(data=True):
                existing = session.execute(
                    select(EntityNode).where(
                        EntityNode.tenant_id == tenant_id, EntityNode.name == node_name
                    )
                ).scalar_one_or_none()

                if not existing:
                    new_node = EntityNode(
                        tenant_id=tenant_id,
                        name=node_name,
                        type=attrs.get("type", "UNKNOWN"),
                        description=f"From conversation {conversation_id}",
                        properties=attrs.get("properties", {}),
                    )
                    session.add(new_node)
                    session.flush()
                    node_map[node_name] = new_node.node_id
                    nodes_new += 1
                else:
                    node_map[node_name] = existing.node_id
                    nodes_reused += 1

            edges_created = 0
            for src, dst, edge_attrs in G.edges(data=True):
                src_id = node_map.get(src)
                target_id = node_map.get(dst)

                if src_id and target_id:
                    edge = EntityEdge(
                        tenant_id=tenant_id,
                        source_id=src_id,
                        target_id=target_id,
                        relation=edge_attrs.get("relation", "RELATED_TO"),
                        description=edge_attrs.get("description", ""),
                        conversation_id=conversation_id,
                        weight=1.0,
                    )
                    session.add(edge)
                    edges_created += 1

            session.commit()

            result["nodes_created"] = nodes_new
            result["nodes_reused"] = nodes_reused
            result["edges_created"] = edges_created
            result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        logger.debug(f"Failed {conversation_id}: {e}")

    return result


def run_backfill(
    tenant_id: str | None = None,
    max_workers: int = 25,
    limit: int | None = None,
    dry_run: bool = False,
):
    """Run the graph backfill using summaries."""
    logger.info("Starting Fast Graph Backfill with Model Fallback...")
    logger.info(f"Workers: {max_workers}, Limit: {limit or 'all'}, Dry run: {dry_run}")
    logger.info(f"Models: {' -> '.join(MODELS)}")

    # Get conversations to process
    conversations = get_conversations_without_graphs(tenant_id, limit)
    total = len(conversations)
    logger.info(f"Found {total} conversations with summaries needing graphs")

    if total == 0:
        logger.info("Nothing to process!")
        return {"total": 0, "success": 0, "failed": 0}

    # Create extractor with fallback
    extractor = GraphExtractorWithFallback()

    # Results tracking
    results = {
        "total": total,
        "success": 0,
        "failed": 0,
        "nodes_created": 0,
        "nodes_reused": 0,
        "edges_created": 0,
    }

    # Process with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_conversation, conv_id, tid, summary, extractor, dry_run
            ): conv_id
            for conv_id, tid, summary in conversations
        }

        with tqdm(total=total, unit="conv", desc="Graph extraction") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result["success"]:
                        results["success"] += 1
                        results["nodes_created"] += result["nodes_created"]
                        results["nodes_reused"] += result["nodes_reused"]
                        results["edges_created"] += result["edges_created"]
                    else:
                        results["failed"] += 1
                except Exception as e:
                    results["failed"] += 1
                    logger.debug(f"Future failed: {e}")

                pbar.update(1)
                pbar.set_postfix(
                    ok=results["success"],
                    fail=results["failed"],
                    nodes=results["nodes_created"],
                )

    # Final report
    logger.info("\n" + "=" * 50)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total processed: {results['total']}")
    logger.info(f"Success: {results['success']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Nodes created: {results['nodes_created']}")
    logger.info(f"Nodes reused: {results['nodes_reused']}")
    logger.info(f"Edges created: {results['edges_created']}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fast Graph Backfill with Model Fallback"
    )
    parser.add_argument(
        "--tenant-id", type=str, help="Filter by tenant ID", default=None
    )
    parser.add_argument(
        "--workers", type=int, default=25, help="Number of parallel workers"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of conversations"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't write to DB, just report"
    )

    args = parser.parse_args()

    try:
        results = run_backfill(
            tenant_id=args.tenant_id,
            max_workers=args.workers,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        sys.exit(0 if results["failed"] == 0 else 1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
