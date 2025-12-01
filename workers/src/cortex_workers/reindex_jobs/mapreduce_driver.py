"""
Map-Reduce Driver for Parallel Embedding Jobs.

Implements §7.3.1 of the Canonical Blueprint:
* Driver partitions work across workers
* Workers embed batches in parallel
* Driver merges results and commits to database

Architecture:
    Driver (this module)
        |
        +---> Worker 1 (embed batch 1)
        +---> Worker 2 (embed batch 2)
        +---> Worker N (embed batch N)
        |
        v
    Merge & Commit
"""
from __future__ import annotations

import logging
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

# Add backend/src to sys.path
try:
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[4]
    backend_src = project_root / "backend" / "src"
    if backend_src.exists() and str(backend_src) not in sys.path:
        sys.path.insert(0, str(backend_src))
except Exception:
    pass


logger = logging.getLogger(__name__)

T = TypeVar("T")


# -----------------------------------------------------------------------------
# Data Classes for Map-Reduce Jobs
# -----------------------------------------------------------------------------

class JobState(str, Enum):
    """State of a map-reduce job."""
    PENDING = "pending"
    MAPPING = "mapping"
    REDUCING = "reducing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MapTask:
    """Represents a single map task (worker unit of work)."""
    task_id: str
    batch_index: int
    chunk_ids: list[str]
    texts: list[str]
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())


@dataclass
class MapResult:
    """Result from a single map task."""
    task_id: str
    batch_index: int
    success: bool
    chunk_ids: list[str]
    embeddings: NDArray[np.float32] | None = None
    error: str | None = None
    processing_time_seconds: float = 0.0


@dataclass
class MapReduceJob:
    """Represents a complete map-reduce embedding job."""
    job_id: str
    tenant_id: str
    state: JobState = JobState.PENDING
    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    num_workers: int = 4
    batch_size: int = 64
    tasks: list[MapTask] = field(default_factory=list)
    results: list[MapResult] = field(default_factory=list)
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    
    def __post_init__(self):
        if not self.job_id:
            self.job_id = str(uuid.uuid4())


@dataclass
class EmbeddingBatch:
    """Batch of text to embed with metadata."""
    batch_id: int
    chunk_ids: list[str]
    texts: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Worker Functions (Run in Separate Processes)
# -----------------------------------------------------------------------------

def _init_worker():
    """Initialize worker process."""
    # Each worker needs to initialize its own config and LLM runtime
    try:
        from cortex.config.loader import get_config, reset_config
        reset_config()  # Reset singleton for fresh init in worker
        get_config()
    except Exception as e:
        logger.warning(f"Worker init warning: {e}")


def _embed_batch_worker(batch: EmbeddingBatch) -> tuple[int, bool, NDArray[np.float32] | None, str | None]:
    """
    Worker function to embed a batch of texts.
    
    Args:
        batch: EmbeddingBatch with texts to embed
        
    Returns:
        Tuple of (batch_id, success, embeddings, error_message)
    """
    try:
        from cortex.llm.client import embed_texts
        
        start_time = time.time()
        embeddings = embed_texts(batch.texts)
        elapsed = time.time() - start_time
        
        logger.debug(
            f"Batch {batch.batch_id}: Embedded {len(batch.texts)} texts in {elapsed:.2f}s"
        )
        
        return (batch.batch_id, True, embeddings, None)
        
    except Exception as e:
        logger.error(f"Batch {batch.batch_id} failed: {e}")
        return (batch.batch_id, False, None, str(e))


def _process_map_task(task: MapTask) -> MapResult:
    """
    Process a single map task (embed a batch).
    
    This is the main worker function for the map phase.
    """
    start_time = time.time()
    
    try:
        from cortex.llm.client import embed_texts
        
        embeddings = embed_texts(task.texts)
        
        return MapResult(
            task_id=task.task_id,
            batch_index=task.batch_index,
            success=True,
            chunk_ids=task.chunk_ids,
            embeddings=embeddings,
            processing_time_seconds=time.time() - start_time,
        )
        
    except Exception as e:
        logger.error(f"Map task {task.task_id} failed: {e}")
        return MapResult(
            task_id=task.task_id,
            batch_index=task.batch_index,
            success=False,
            chunk_ids=task.chunk_ids,
            error=str(e),
            processing_time_seconds=time.time() - start_time,
        )


# -----------------------------------------------------------------------------
# Map-Reduce Driver
# -----------------------------------------------------------------------------

class MapReduceEmbeddingDriver:
    """
    Driver for Map-Reduce parallel embedding jobs.
    
    Implements Blueprint §7.3.1:
    * Partition: Split chunks into batches for parallel processing
    * Map: Workers embed batches concurrently
    * Reduce: Merge results and update database
    
    Features:
    * Configurable parallelism
    * Progress tracking and reporting
    * Fault tolerance with partial retry
    * Memory-efficient streaming for large datasets
    """
    
    def __init__(
        self,
        num_workers: int = 4,
        batch_size: int = 64,
        use_processes: bool = True,
        max_retries: int = 2,
    ):
        """
        Initialize the driver.
        
        Args:
            num_workers: Number of parallel workers
            batch_size: Texts per batch
            use_processes: Use processes (True) or threads (False)
            max_retries: Max retries for failed batches
        """
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.use_processes = use_processes
        self.max_retries = max_retries
        self._lock = threading.Lock()
        
        # Get config-based defaults
        try:
            from cortex.config.loader import get_config
            config = get_config()
            self.num_workers = config.processing.num_workers
            self.batch_size = config.processing.batch_size
        except Exception:
            pass
        
        logger.info(
            f"MapReduceEmbeddingDriver initialized: "
            f"workers={self.num_workers}, batch_size={self.batch_size}"
        )

    def partition_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> Iterator[EmbeddingBatch]:
        """
        Partition chunks into batches for parallel processing.
        
        Args:
            chunks: List of chunk dicts with 'id' and 'text' keys
            
        Yields:
            EmbeddingBatch instances
        """
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            yield EmbeddingBatch(
                batch_id=i // self.batch_size,
                chunk_ids=[c["id"] for c in batch_chunks],
                texts=[c["text"] for c in batch_chunks],
            )

    def map_phase(
        self,
        batches: list[EmbeddingBatch],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[tuple[int, bool, NDArray[np.float32] | None, str | None]]:
        """
        Execute the map phase: embed batches in parallel.
        
        Args:
            batches: List of batches to embed
            progress_callback: Optional callback(completed, total)
            
        Returns:
            List of (batch_id, success, embeddings, error) tuples
        """
        results = []
        total = len(batches)
        completed = 0
        
        # Choose executor type
        if self.use_processes:
            executor_class = ProcessPoolExecutor
            executor_kwargs = {
                "max_workers": self.num_workers,
                "initializer": _init_worker,
            }
        else:
            executor_class = ThreadPoolExecutor
            executor_kwargs = {"max_workers": self.num_workers}
        
        with executor_class(**executor_kwargs) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(_embed_batch_worker, batch): batch
                for batch in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch {batch.batch_id} exception: {e}")
                    results.append((batch.batch_id, False, None, str(e)))
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
        
        return results

    def reduce_phase(
        self,
        batches: list[EmbeddingBatch],
        map_results: list[tuple[int, bool, NDArray[np.float32] | None, str | None]],
    ) -> tuple[dict[str, NDArray[np.float32]], list[str]]:
        """
        Execute the reduce phase: merge embeddings by chunk ID.
        
        Args:
            batches: Original batches
            map_results: Results from map phase
            
        Returns:
            Tuple of (chunk_id -> embedding dict, list of failed chunk IDs)
        """
        embeddings_by_id: dict[str, NDArray[np.float32]] = {}
        failed_ids: list[str] = []
        
        # Create batch_id -> batch lookup
        batch_map = {b.batch_id: b for b in batches}
        
        for batch_id, success, embeddings, error in map_results:
            batch = batch_map.get(batch_id)
            if not batch:
                continue
            
            if success and embeddings is not None:
                for chunk_id, embedding in zip(batch.chunk_ids, embeddings, strict=False):
                    embeddings_by_id[chunk_id] = embedding
            else:
                failed_ids.extend(batch.chunk_ids)
                logger.warning(f"Batch {batch_id} failed: {error}")
        
        return embeddings_by_id, failed_ids

    def run_job(
        self,
        job: MapReduceJob,
        fetch_chunks_fn: Callable[[str], list[dict[str, Any]]],
        update_chunks_fn: Callable[[dict[str, NDArray[np.float32]]], int],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> MapReduceJob:
        """
        Run a complete map-reduce embedding job.
        
        Args:
            job: Job configuration
            fetch_chunks_fn: Function to fetch chunks needing embedding
            update_chunks_fn: Function to update chunks with embeddings
            progress_callback: Optional progress callback
            
        Returns:
            Updated job with results
        """
        job.started_at = time.time()
        job.state = JobState.MAPPING
        
        try:
            # Fetch chunks
            logger.info(f"Job {job.job_id}: Fetching chunks for tenant {job.tenant_id}")
            chunks = fetch_chunks_fn(job.tenant_id)
            job.total_chunks = len(chunks)
            
            if not chunks:
                logger.info(f"Job {job.job_id}: No chunks to process")
                job.state = JobState.COMPLETED
                job.completed_at = time.time()
                return job
            
            # Partition into batches
            batches = list(self.partition_chunks(chunks))
            logger.info(f"Job {job.job_id}: Created {len(batches)} batches")
            
            # Map phase
            logger.info(f"Job {job.job_id}: Starting map phase with {self.num_workers} workers")
            map_results = self.map_phase(batches, progress_callback)
            
            # Reduce phase
            job.state = JobState.REDUCING
            logger.info(f"Job {job.job_id}: Starting reduce phase")
            embeddings_by_id, failed_ids = self.reduce_phase(batches, map_results)
            
            # Retry failed batches
            if failed_ids and self.max_retries > 0:
                logger.info(f"Job {job.job_id}: Retrying {len(failed_ids)} failed chunks")
                retry_chunks = [c for c in chunks if c["id"] in set(failed_ids)]
                
                for _attempt in range(self.max_retries):
                    if not retry_chunks:
                        break
                    
                    retry_batches = list(self.partition_chunks(retry_chunks))
                    retry_results = self.map_phase(retry_batches)
                    retry_embeddings, still_failed = self.reduce_phase(retry_batches, retry_results)
                    
                    embeddings_by_id.update(retry_embeddings)
                    retry_chunks = [c for c in retry_chunks if c["id"] in set(still_failed)]
                    
                    if not retry_chunks:
                        break
                
                failed_ids = [c["id"] for c in retry_chunks]
            
            # Update database
            if embeddings_by_id:
                logger.info(f"Job {job.job_id}: Updating {len(embeddings_by_id)} chunks")
                updated = update_chunks_fn(embeddings_by_id)
                job.processed_chunks = updated
            
            job.failed_chunks = len(failed_ids)
            job.state = JobState.COMPLETED
            job.completed_at = time.time()
            
            elapsed = job.completed_at - job.started_at
            logger.info(
                f"Job {job.job_id}: Completed in {elapsed:.2f}s - "
                f"processed={job.processed_chunks}, failed={job.failed_chunks}"
            )
            
            return job
            
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            job.state = JobState.FAILED
            job.error = str(e)
            job.completed_at = time.time()
            raise

    def run_tenant_reindex(
        self,
        tenant_id: str,
        force: bool = False,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Convenience method to reindex all chunks for a tenant.
        
        Args:
            tenant_id: Tenant ID to reindex
            force: Force re-embedding even if already embedded
            model_name: Target model name (uses config default if None)
            
        Returns:
            Job summary dict
        """
        from cortex.db.session import SessionLocal
        from cortex.db.models import Chunk
        from cortex.config.loader import get_config
        
        config = get_config()
        target_model = model_name or config.embedding.model_name
        
        def fetch_chunks(tid: str) -> list[dict[str, Any]]:
            with SessionLocal() as session:
                if force:
                    query = session.query(Chunk).filter(Chunk.tenant_id == tid)
                else:
                    query = session.query(Chunk).filter(
                        Chunk.tenant_id == tid,
                        (Chunk.embedding.is_(None)) | (Chunk.embedding_model != target_model)
                    )
                
                return [
                    {"id": str(c.id), "text": c.text}
                    for c in query.all()
                ]
        
        def update_chunks(embeddings: dict[str, NDArray[np.float32]]) -> int:
            with SessionLocal() as session:
                updated = 0
                for chunk_id, embedding in embeddings.items():
                    chunk = session.query(Chunk).filter(Chunk.id == chunk_id).first()
                    if chunk:
                        chunk.embedding = embedding.tolist()
                        chunk.embedding_model = target_model
                        updated += 1
                session.commit()
                return updated
        
        job = MapReduceJob(
            job_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )
        
        result = self.run_job(job, fetch_chunks, update_chunks)
        
        return {
            "job_id": result.job_id,
            "tenant_id": result.tenant_id,
            "state": result.state.value,
            "total_chunks": result.total_chunks,
            "processed_chunks": result.processed_chunks,
            "failed_chunks": result.failed_chunks,
            "duration_seconds": (result.completed_at or 0) - (result.started_at or 0),
            "error": result.error,
        }


# -----------------------------------------------------------------------------
# Streaming Map-Reduce for Large Datasets
# -----------------------------------------------------------------------------

class StreamingMapReduceDriver(MapReduceEmbeddingDriver):
    """
    Streaming variant for very large datasets that don't fit in memory.
    
    Uses database cursors and incremental updates to minimize memory usage.
    """
    
    def __init__(
        self,
        num_workers: int = 4,
        batch_size: int = 64,
        chunk_buffer_size: int = 1000,
        **kwargs,
    ):
        super().__init__(num_workers=num_workers, batch_size=batch_size, **kwargs)
        self.chunk_buffer_size = chunk_buffer_size

    def stream_chunks(
        self,
        tenant_id: str,
        model_name: str,
        force: bool = False,
    ) -> Iterator[list[dict[str, Any]]]:
        """
        Stream chunks in buffers to avoid loading all into memory.
        
        Yields:
            Buffers of chunk dicts
        """
        from cortex.db.session import SessionLocal
        from cortex.db.models import Chunk
        
        with SessionLocal() as session:
            if force:
                query = session.query(Chunk).filter(Chunk.tenant_id == tenant_id)
            else:
                query = session.query(Chunk).filter(
                    Chunk.tenant_id == tenant_id,
                    (Chunk.embedding.is_(None)) | (Chunk.embedding_model != model_name)
                )
            
            # Use yield_per for efficient streaming
            buffer = []
            for chunk in query.yield_per(100):
                buffer.append({"id": str(chunk.id), "text": chunk.text})
                if len(buffer) >= self.chunk_buffer_size:
                    yield buffer
                    buffer = []
            
            if buffer:
                yield buffer

    def run_streaming_reindex(
        self,
        tenant_id: str,
        force: bool = False,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Run reindex with streaming to handle large datasets.
        
        Args:
            tenant_id: Tenant to reindex
            force: Force re-embedding
            model_name: Target model
            
        Returns:
            Summary dict
        """
        from cortex.db.session import SessionLocal
        from cortex.db.models import Chunk
        from cortex.config.loader import get_config
        
        config = get_config()
        target_model = model_name or config.embedding.model_name
        
        start_time = time.time()
        total_processed = 0
        total_failed = 0
        total_chunks = 0
        
        for chunk_buffer in self.stream_chunks(tenant_id, target_model, force):
            total_chunks += len(chunk_buffer)
            
            # Process buffer
            batches = list(self.partition_chunks(chunk_buffer))
            map_results = self.map_phase(batches)
            embeddings_by_id, failed_ids = self.reduce_phase(batches, map_results)
            
            # Update incrementally
            if embeddings_by_id:
                with SessionLocal() as session:
                    for chunk_id, embedding in embeddings_by_id.items():
                        chunk = session.query(Chunk).filter(Chunk.id == chunk_id).first()
                        if chunk:
                            chunk.embedding = embedding.tolist()
                            chunk.embedding_model = target_model
                            total_processed += 1
                    session.commit()
            
            total_failed += len(failed_ids)
            
            logger.info(
                f"Streaming reindex progress: {total_processed}/{total_chunks} processed"
            )
        
        return {
            "tenant_id": tenant_id,
            "total_chunks": total_chunks,
            "processed_chunks": total_processed,
            "failed_chunks": total_failed,
            "duration_seconds": time.time() - start_time,
        }


# -----------------------------------------------------------------------------
# Distributed Map-Reduce (Using Queue)
# -----------------------------------------------------------------------------

class DistributedMapReduceDriver:
    """
    Distributed map-reduce that uses the job queue for work distribution.
    
    This variant submits individual batch jobs to the queue for processing
    by multiple worker nodes, enabling horizontal scaling.
    """
    
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size

    def submit_distributed_job(
        self,
        tenant_id: str,
        force: bool = False,
    ) -> str:
        """
        Submit a distributed reindex job.
        
        Creates sub-jobs for each batch and submits to queue.
        
        Returns:
            Parent job ID
        """
        from cortex.queue import get_queue
        from cortex.db.session import SessionLocal
        from cortex.db.models import Chunk
        from cortex.config.loader import get_config
        
        config = get_config()
        target_model = config.embedding.model_name
        queue = get_queue()
        
        parent_job_id = str(uuid.uuid4())
        
        with SessionLocal() as session:
            if force:
                query = session.query(Chunk.id, Chunk.text).filter(
                    Chunk.tenant_id == tenant_id
                )
            else:
                query = session.query(Chunk.id, Chunk.text).filter(
                    Chunk.tenant_id == tenant_id,
                    (Chunk.embedding.is_(None)) | (Chunk.embedding_model != target_model)
                )
            
            chunks = [(str(c.id), c.text) for c in query.all()]
        
        if not chunks:
            logger.info(f"No chunks to reindex for tenant {tenant_id}")
            return parent_job_id
        
        # Create batch jobs
        batch_jobs = []
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_job_id = queue.enqueue(
                "embed_batch",
                {
                    "parent_job_id": parent_job_id,
                    "tenant_id": tenant_id,
                    "batch_index": i // self.batch_size,
                    "chunk_ids": [c[0] for c in batch_chunks],
                    "texts": [c[1] for c in batch_chunks],
                    "model_name": target_model,
                },
            )
            batch_jobs.append(batch_job_id)
        
        # Enqueue a completion check job
        queue.enqueue(
            "check_distributed_completion",
            {
                "parent_job_id": parent_job_id,
                "tenant_id": tenant_id,
                "batch_job_ids": batch_jobs,
                "total_batches": len(batch_jobs),
            },
            priority=-1,  # Lower priority
        )
        
        logger.info(
            f"Submitted distributed reindex: parent={parent_job_id}, "
            f"batches={len(batch_jobs)}"
        )
        
        return parent_job_id


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------

def get_mapreduce_driver(
    distributed: bool = False,
    streaming: bool = False,
    **kwargs,
) -> MapReduceEmbeddingDriver:
    """
    Factory to get appropriate map-reduce driver.
    
    Args:
        distributed: Use distributed queue-based driver
        streaming: Use streaming driver for large datasets
        **kwargs: Additional driver kwargs
        
    Returns:
        Configured driver instance
    """
    if distributed:
        return DistributedMapReduceDriver(**kwargs)
    elif streaming:
        return StreamingMapReduceDriver(**kwargs)
    else:
        return MapReduceEmbeddingDriver(**kwargs)
