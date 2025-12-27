"""
Reindexing Jobs.

Implements §7.3 of the Canonical Blueprint.

This module provides:
* ParallelEmbeddingIndexer - Simple parallel indexing for conversations
* MapReduceEmbeddingDriver - Full Map-Reduce driver for database chunks
* StreamingMapReduceDriver - Memory-efficient streaming for large datasets
* DistributedMapReduceDriver - Queue-based distributed processing
"""

from __future__ import annotations

import logging
import multiprocessing
import sys
from pathlib import Path
from typing import Any

# Add backend/src to sys.path
try:
    current_file = Path(__file__).resolve()
    # workers/src/cortex_workers/reindex_jobs/__init__.py -> ... -> workers -> root
    PROJECT_ROOT = current_file.parents[4]
    BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
    if BACKEND_SRC.exists():
        sys.path.insert(0, str(BACKEND_SRC))
except Exception:
    pass

from cortex.config.loader import get_config
from cortex.db.models import Chunk
from cortex.db.session import SessionLocal
from cortex.llm.client import embed_texts
from sqlalchemy import select

logger = logging.getLogger(__name__)

from .mapreduce_driver import (
    DistributedMapReduceDriver,
    JobState,
    MapReduceEmbeddingDriver,
    MapReduceJob,
    StreamingMapReduceDriver,
    get_mapreduce_driver,
)

# Re-export main classes
from .parallel_indexer import ParallelEmbeddingIndexer, parallel_index_conversations

__all__ = (
    "DistributedMapReduceDriver",
    "JobState",
    "MapReduceEmbeddingDriver",
    "MapReduceJob",
    "ParallelEmbeddingIndexer",
    "StreamingMapReduceDriver",
    "get_mapreduce_driver",
    "parallel_index_conversations",
    "process_reindex_job",
)


def process_reindex_job(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Process a reindexing job.

    Blueprint §7.3:
    * Incremental re-embed
    * Map-Reduce style driver

    Args:
        payload: Job payload with:
            - tenant_id: Required tenant ID
            - thread_ids: Optional list of specific threads
            - force: Optional force re-embedding
            - streaming: Optional use streaming for large datasets

    Returns:
        Job summary dict
    """
    tenant_id = payload.get("tenant_id")
    if not tenant_id:
        logger.error("Reindex job missing tenant_id")
        return {"error": "Missing tenant_id", "success": False}

    thread_ids = payload.get("thread_ids")
    force = payload.get("force", False)
    streaming = payload.get("streaming", False)

    logger.info(f"Starting reindex job for tenant {tenant_id}")

    try:
        config = get_config()

        # Choose driver based on payload hints
        if streaming:
            driver = StreamingMapReduceDriver(
                num_workers=config.processing.num_workers,
                batch_size=config.processing.batch_size,
            )
            result = driver.run_streaming_reindex(tenant_id, force=force)
        elif thread_ids:
            # Use simple indexer for specific threads
            indexer = ParallelEmbeddingIndexer(
                num_workers=config.processing.num_workers,
                batch_size=config.processing.batch_size,
            )
            results = []
            for thread_id in thread_ids:
                res = indexer.reindex_thread(tenant_id, thread_id)
                results.append(res)

            result = {
                "tenant_id": tenant_id,
                "threads_processed": len(thread_ids),
                "total_chunks": sum(r.get("chunks_processed", 0) for r in results),
                "success": all(r.get("status") == "success" for r in results),
            }
        else:
            # Use Map-Reduce driver for full tenant reindex
            driver = MapReduceEmbeddingDriver(
                num_workers=config.processing.num_workers,
                batch_size=config.processing.batch_size,
            )
            result = driver.run_tenant_reindex(tenant_id, force=force)

        logger.info(f"Reindex job completed: {result}")
        return {**result, "success": True}

    except Exception as e:
        logger.error(f"Reindex job failed: {e}")
        return {"error": str(e), "success": False, "tenant_id": tenant_id}
