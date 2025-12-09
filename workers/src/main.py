"""
Worker Entrypoint.

Implements §7.3 and §7.4 of the Canonical Blueprint.

Features:
* Multi-process worker pool
* Job type routing (ingest, reindex, embed_batch)
* Graceful shutdown
* Health monitoring
"""
from __future__ import annotations

import logging
import multiprocessing
import signal
import sys
import time
from pathlib import Path
from typing import Any

# Add backend/src to sys.path so we can import cortex
try:
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    backend_src = project_root / "backend" / "src"
    if backend_src.exists():
        sys.path.insert(0, str(backend_src))
except Exception:
    pass

from cortex.config.loader import get_config
from cortex.observability import init_observability
from cortex.queue import JobStatus, get_queue

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Job Handlers
# -----------------------------------------------------------------------------


def handle_ingest_job(payload: dict[str, Any]) -> bool:
    """
    Handle an ingest job.

    Args:
        payload: Job payload with ingest configuration

    Returns:
        True if successful
    """
    from cortex.ingestion.mailroom import process_job as run_ingest
    from cortex.ingestion.models import IngestJobRequest

    try:
        ingest_job = IngestJobRequest(**payload)
        summary = run_ingest(ingest_job)

        if summary.aborted_reason:
            logger.error(f"Ingest job aborted: {summary.aborted_reason}")
            return False

        logger.info(f"Ingest job completed: {summary.messages_ingested} messages")
        return True

    except Exception as e:
        logger.error(f"Ingest job failed: {e}", exc_info=True)
        return False


def handle_reindex_job(payload: dict[str, Any]) -> bool:
    """
    Handle a reindex job.

    Args:
        payload: Job payload with reindex configuration

    Returns:
        True if successful
    """
    from cortex_workers.reindex_jobs import process_reindex_job

    try:
        result = process_reindex_job(payload)
        return result.get("success", False)

    except Exception as e:
        logger.error(f"Reindex job failed: {e}", exc_info=True)
        return False


def handle_embed_batch_job(payload: dict[str, Any]) -> bool:
    """
    Handle a single embedding batch job (for distributed map-reduce).

    Args:
        payload: Job payload with batch data

    Returns:
        True if successful
    """
    from cortex.db.models import Chunk
    from cortex.db.session import SessionLocal
    from cortex.llm.client import embed_texts

    try:
        chunk_ids = payload.get("chunk_ids", [])
        texts = payload.get("texts", [])
        model_name = payload.get("model_name")

        if not chunk_ids or not texts:
            logger.warning("Empty batch job")
            return True

        # Embed texts
        embeddings = embed_texts(texts)

        # Update chunks
        with SessionLocal() as session:
            for chunk_id, embedding in zip(chunk_ids, embeddings, strict=False):
                chunk = session.query(Chunk).filter(Chunk.id == chunk_id).first()
                if chunk:
                    chunk.embedding = embedding.tolist()
                    if model_name:
                        chunk.embedding_model = model_name
            session.commit()

        logger.info(f"Embedded batch of {len(chunk_ids)} chunks")
        return True

    except Exception as e:
        logger.error(f"Embed batch job failed: {e}", exc_info=True)
        return False


def handle_check_distributed_completion(payload: dict[str, Any]) -> bool:
    """
    Check if all batches of a distributed job are complete.

    Args:
        payload: Job payload with batch job IDs

    Returns:
        True if all complete or should stop checking
    """
    queue = get_queue()

    parent_job_id = payload.get("parent_job_id")
    batch_job_ids = payload.get("batch_job_ids", [])

    completed = 0
    failed = 0
    pending = 0

    for job_id in batch_job_ids:
        status = queue.get_job_status(job_id)
        if status:
            if status.get("status") == JobStatus.COMPLETED:
                completed += 1
            elif status.get("status") in (JobStatus.FAILED, JobStatus.DEAD_LETTER):
                failed += 1
            else:
                pending += 1

    logger.info(
        f"Distributed job {parent_job_id}: "
        f"completed={completed}, failed={failed}, pending={pending}"
    )

    if pending > 0:
        # Re-enqueue check with delay
        time.sleep(5)
        queue.enqueue(
            "check_distributed_completion",
            payload,
            priority=-1,
        )
    else:
        logger.info(
            f"Distributed job {parent_job_id} complete: {completed} succeeded, {failed} failed"
        )

    return True


# Job type to handler mapping
JOB_HANDLERS = {
    "ingest": handle_ingest_job,
    "reindex": handle_reindex_job,
    "embed_batch": handle_embed_batch_job,
    "check_distributed_completion": handle_check_distributed_completion,
}


# -----------------------------------------------------------------------------
# Job Processing
# -----------------------------------------------------------------------------


def process_job(job: dict[str, Any]) -> bool:
    """
    Dispatch and process a single job.

    Args:
        job: Job dictionary with 'type' and 'payload'

    Returns:
        True if successful, False otherwise
    """
    job_type = job.get("type")
    job_id = job.get("id", "unknown")
    payload = job.get("payload", {})

    logger.info(f"Processing job {job_id} of type {job_type}")

    handler = JOB_HANDLERS.get(job_type)
    if not handler:
        logger.error(f"Unknown job type: {job_type}")
        return False

    try:
        return handler(payload)
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        return False


# -----------------------------------------------------------------------------
# Worker Process
# -----------------------------------------------------------------------------


class WorkerProcess:
    """
    Worker process that polls queue and processes jobs.

    Features:
    * Configurable job type filtering
    * Graceful shutdown on signal
    * Error backoff
    * Health reporting
    """

    def __init__(
        self,
        worker_id: int,
        job_types: list[str],
        poll_timeout: int = 5,
        error_backoff: int = 5,
    ):
        self.worker_id = worker_id
        self.job_types = job_types
        self.poll_timeout = poll_timeout
        self.error_backoff = error_backoff
        self._running = True
        self._jobs_processed = 0
        self._jobs_failed = 0
        self._last_job_time: float | None = None

    def run(self) -> None:
        """Main worker loop."""
        queue = get_queue()
        logger.info(f"Worker {self.worker_id} started, listening for: {self.job_types}")

        while self._running:
            try:
                job = queue.dequeue(self.job_types, timeout=self.poll_timeout)

                if job:
                    self._last_job_time = time.time()
                    success = process_job(job)

                    if success:
                        queue.ack(job["id"])
                        self._jobs_processed += 1
                    else:
                        error_msg = "Job processing returned False"
                        queue.nack(job["id"], error=error_msg)
                        self._jobs_failed += 1

            except KeyboardInterrupt:
                logger.info(f"Worker {self.worker_id} received interrupt")
                break
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}", exc_info=True)
                time.sleep(self.error_backoff)

        logger.info(
            f"Worker {self.worker_id} stopped: "
            f"processed={self._jobs_processed}, failed={self._jobs_failed}"
        )

    def stop(self) -> None:
        """Signal worker to stop."""
        self._running = False

    def get_stats(self) -> dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "jobs_processed": self._jobs_processed,
            "jobs_failed": self._jobs_failed,
            "last_job_time": self._last_job_time,
            "running": self._running,
        }


def worker_process_main(worker_id: int, job_types: list[str]) -> None:
    """
    Entry point for worker subprocess.

    Args:
        worker_id: Unique worker identifier
        job_types: List of job types to process
    """
    # Initialize in worker process
    try:
        from cortex.config.loader import get_config, reset_config

        reset_config()  # Fresh config in worker
        get_config()
    except Exception as e:
        logger.warning(f"Worker {worker_id} config init warning: {e}")

    worker = WorkerProcess(worker_id, job_types)

    # Handle signals in worker
    def signal_handler(_sig, _frame):
        worker.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    worker.run()


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def main(args: list[str] | None = None) -> None:
    """
    Main entry point for workers.

    Args:
        args: Command line arguments (unused for now)
    """
    if args is None:
        args = sys.argv[1:]

    # Initialize observability
    init_observability(service_name="cortex-worker")

    config = get_config()

    # Default settings
    concurrency = config.processing.num_workers if hasattr(config, "processing") else 2
    job_types = list(JOB_HANDLERS.keys())

    logger.info(f"Starting Cortex Workers (concurrency={concurrency})")
    logger.info(f"Registered job types: {job_types}")

    processes: list[multiprocessing.Process] = []

    for i in range(concurrency):
        p = multiprocessing.Process(
            target=worker_process_main,
            args=(i, job_types),
            daemon=True,
            name=f"cortex-worker-{i}",
        )
        p.start()
        processes.append(p)
        logger.info(f"Started worker process {i} (PID: {p.pid})")

    # Handle graceful shutdown
    shutdown_event = multiprocessing.Event()

    def signal_handler(_sig, _frame):
        logger.info("Shutting down workers...")
        shutdown_event.set()
        for p in processes:
            if p.is_alive():
                p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Monitor loop
    try:
        while not shutdown_event.is_set():
            # Check worker health
            alive_workers = sum(1 for p in processes if p.is_alive())
            if alive_workers < concurrency:
                dead_workers = [p for p in processes if not p.is_alive()]
                for p in dead_workers:
                    logger.warning(f"Worker {p.name} died, restarting...")
                    processes.remove(p)
                    new_id = int(p.name.split("-")[-1])
                    new_p = multiprocessing.Process(
                        target=worker_process_main,
                        args=(new_id, job_types),
                        daemon=True,
                        name=f"cortex-worker-{new_id}",
                    )
                    new_p.start()
                    processes.append(new_p)
                    logger.info(f"Restarted worker {new_id} (PID: {new_p.pid})")

            time.sleep(10)  # Health check interval

    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    # Set start method to spawn for safety (especially on Windows)
    multiprocessing.set_start_method("spawn", force=True)
    main()
