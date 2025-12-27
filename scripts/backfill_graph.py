#!/usr/bin/env python3
"""
Backfill Knowledge Graph for existing conversations.

Uses existing cortex.intelligence.graph.GraphExtractor and
cortex.ingestion.writer.DBWriter for proper integration.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add backend to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
if BACKEND_SRC.exists():
    sys.path.insert(0, str(BACKEND_SRC))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from cortex.db.models import Chunk, Conversation  # noqa: E402
from cortex.db.session import SessionLocal  # noqa: E402
from cortex.intelligence.graph import GraphExtractor  # noqa: E402
from sqlalchemy import select  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "graph_backfill.log"),
    ],
)
logger = logging.getLogger("graph_backfill")


def get_all_conversations(tenant_id: str | None = None) -> list[tuple[uuid.UUID, str]]:
    """Get all conversation IDs and their tenant IDs from the database."""
    with SessionLocal() as session:
        stmt = select(Conversation.conversation_id, Conversation.tenant_id)
        if tenant_id:
            stmt = stmt.where(Conversation.tenant_id == tenant_id)
        results = session.execute(stmt).all()
        return [(row[0], row[1]) for row in results]


def get_conversation_context(session, conversation_id: uuid.UUID) -> str:
    """Reconstruct conversation text from chunks (body + attachments) - matches mailroom.py logic."""
    # Get body chunks
    body_stmt = (
        select(Chunk.text)
        .where(
            Chunk.conversation_id == conversation_id,
            Chunk.is_attachment == False,  # noqa: E712
        )
        .order_by(Chunk.position)
    )
    body_texts = session.execute(body_stmt).scalars().all()

    # Get attachment chunks
    att_stmt = (
        select(Chunk.text)
        .where(
            Chunk.conversation_id == conversation_id,
            Chunk.is_attachment == True,  # noqa: E712
        )
        .order_by(Chunk.position)
    )
    att_texts = session.execute(att_stmt).scalars().all()

    # Match the format from mailroom.py _process_intelligence()
    parts = []
    if body_texts:
        parts.append("--- Conversation Messages ---")
        parts.extend(body_texts)

    if att_texts:
        parts.append("\n--- Attachments ---")
        parts.extend(att_texts)

    return "\n\n".join(parts)


def process_conversation(
    conversation_id: uuid.UUID,
    tenant_id: str,
    extractor: GraphExtractor,
    dry_run: bool = False,
) -> dict:
    """Process a single conversation using existing extraction and writing code."""
    result = {
        "conversation_id": str(conversation_id),
        "nodes_created": 0,
        "nodes_reused": 0,
        "edges_created": 0,
        "success": False,
        "error": None,
    }

    try:
        with SessionLocal() as session:
            # 1. Get conversation context (matches mailroom.py approach)
            context = get_conversation_context(session, conversation_id)
            if not context or len(context.strip()) < 50:
                result["error"] = "Text too short"
                return result

            # 2. Extract graph using existing GraphExtractor
            G = extractor.extract_graph(context)
            if G.number_of_nodes() == 0:
                result["error"] = "No nodes extracted"
                return result

            # 3. Format graph data (matches mailroom.py _extract_graph())
            graph_data = {"nodes": [], "edges": []}
            for node_name, attrs in G.nodes(data=True):
                graph_data["nodes"].append(
                    {
                        "name": node_name,
                        "type": attrs.get("type", "UNKNOWN"),
                        "properties": {},
                    }
                )

            for src, dst, edge_attrs in G.edges(data=True):
                graph_data["edges"].append(
                    {
                        "source": src,
                        "target": dst,
                        "relation": edge_attrs.get("relation", "RELATED_TO"),
                        "description": edge_attrs.get("description", ""),
                    }
                )

            result["nodes_created"] = len(graph_data["nodes"])
            result["edges_created"] = len(graph_data["edges"])

            if dry_run:
                result["success"] = True
                return result

            # 4. Use existing DBWriter to persist graph
            # Note: We use direct DB operations here instead of writer for efficiency

            # Use DBWriter's graph writing logic directly
            from cortex.db.models import EntityEdge, EntityNode

            node_map = {}
            nodes_new = 0
            nodes_reused = 0

            for node_dict in graph_data["nodes"]:
                name = node_dict["name"]
                existing = session.execute(
                    select(EntityNode).where(
                        EntityNode.tenant_id == tenant_id, EntityNode.name == name
                    )
                ).scalar_one_or_none()

                if not existing:
                    new_node = EntityNode(
                        tenant_id=tenant_id,
                        name=name,
                        type=node_dict["type"],
                        description=f"Extracted from conversation {conversation_id}",
                        properties=node_dict.get("properties", {}),
                    )
                    session.add(new_node)
                    session.flush()
                    node_map[name] = new_node.node_id
                    nodes_new += 1
                else:
                    node_map[name] = existing.node_id
                    nodes_reused += 1

            edges_created = 0
            for edge_dict in graph_data["edges"]:
                src_id = node_map.get(edge_dict["source"])
                target_id = node_map.get(edge_dict["target"])

                if src_id and target_id:
                    edge = EntityEdge(
                        tenant_id=tenant_id,
                        source_id=src_id,
                        target_id=target_id,
                        relation=edge_dict["relation"],
                        description=edge_dict["description"],
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

            logger.info(
                f"Processed {conversation_id}: {nodes_new} new nodes, "
                f"{nodes_reused} reused, {edges_created} edges"
            )

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Failed to process {conversation_id}: {e}")

    return result


def run_backfill(
    tenant_id: str | None = None,
    max_workers: int = 5,
    limit: int | None = None,
    dry_run: bool = False,
    batch_size: int = 25,
):
    """Run the graph backfill for all conversations with batched processing."""
    import gc

    logger.info("Starting Knowledge Graph backfill...")

    # Get all conversations
    conversations = get_all_conversations(tenant_id)
    total = len(conversations)
    logger.info(f"Found {total} conversations to process")

    if limit:
        conversations = conversations[:limit]
        logger.info(f"Limited to {limit} conversations")

    if dry_run:
        logger.info("DRY RUN MODE - No data will be written")

    # Create extractor (thread-safe - creates per-call LLM clients)
    extractor = GraphExtractor()

    # Results tracking
    results = {
        "total": len(conversations),
        "success": 0,
        "failed": 0,
        "nodes_created": 0,
        "nodes_reused": 0,
        "edges_created": 0,
    }

    # Process in batches to prevent OOM
    num_batches = (len(conversations) + batch_size - 1) // batch_size
    logger.info(
        f"Processing in {num_batches} batches of up to {batch_size} conversations"
    )

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(conversations))
        batch = conversations[batch_start:batch_end]

        logger.info(
            f"Batch {batch_idx + 1}/{num_batches}: processing {len(batch)} conversations"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_conversation, conv_id, tid, extractor, dry_run
                ): (
                    conv_id,
                    tid,
                )
                for conv_id, tid in batch
            }

            for future in as_completed(futures):
                conv_id, tid = futures[future]
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
                    logger.error(f"Future failed for {conv_id}: {e}")

        # Progress report after each batch
        processed = batch_end
        logger.info(
            f"Progress: {processed}/{len(conversations)} "
            f"({results['success']} success, {results['failed']} failed)"
        )

        # Explicit cleanup between batches
        del futures
        gc.collect()

    # Final report
    logger.info("\n=== BACKFILL COMPLETE ===")
    logger.info(f"Total processed: {results['total']}")
    logger.info(f"Success: {results['success']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Nodes created: {results['nodes_created']}")
    logger.info(f"Nodes reused (deduped): {results['nodes_reused']}")
    logger.info(f"Edges created: {results['edges_created']}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Backfill Knowledge Graph for conversations"
    )
    parser.add_argument(
        "--tenant-id", type=str, help="Filter by tenant ID", default=None
    )
    parser.add_argument(
        "--workers", type=int, default=5, help="Number of parallel workers"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of conversations"
    )
    parser.add_argument(
        "--batch-size", type=int, default=25, help="Batch size for processing"
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
            batch_size=args.batch_size,
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
