#!/usr/bin/env python3
"""
Unified Processing Module for EmailOps (revised)
- Safer multiprocessing (no bound-method targets; pure top-level worker)
- Robust monitoring with liveness checks and graceful shutdown
- Resume support for chunking (skips already-chunked docs when enabled)
- Functional embedding pipeline from chunk files (writes batch pickles)
- Safer imports and clearer error messages
- CLI: embed defaults to using chunk files
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import queue
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ChunkJob:
    """A chunking job for a single document"""
    doc_id: str
    doc_path: Path
    file_size: int
    priority: int = 0


@dataclass
class WorkerConfig:
    """Configuration for a chunking worker (serialized primitives only)"""
    worker_id: int
    jobs_assigned: list[tuple[str, str, int]]  # (doc_id, doc_path_str, file_size)
    chunk_config: dict[str, Any]               # kwargs for ChunkConfig


@dataclass
class WorkerStats:
    """Statistics for monitoring progress"""
    worker_id: int
    docs_processed: int
    docs_total: int
    chunks_created: int
    bytes_processed: int
    bytes_total: int
    start_time: float
    last_update: float
    errors: int
    status: str
    current_doc: str | None = None

    @property
    def progress_percent(self) -> float:
        if self.bytes_total <= 0:
            return 0.0
        p = (self.bytes_processed / max(1, self.bytes_total)) * 100.0
        return min(max(p, 0), 100)

    @property
    def estimated_time_remaining(self) -> timedelta | None:
        if self.bytes_processed <= 0:
            return None
        elapsed = max(1e-6, time.time() - self.start_time)
        rate = self.bytes_processed / elapsed
        remaining = max(0, self.bytes_total - self.bytes_processed)
        if rate > 0:
            return timedelta(seconds=remaining / rate)
        return None


@dataclass
class ProcessingStats:
    """Statistics for embedding operations"""
    worker_id: int
    project_id: str
    chunks_processed: int
    chunks_total: int
    start_time: float
    last_update: float
    errors: int
    status: str
    account_group: int

    @property
    def progress_percent(self) -> float:
        if self.chunks_total == 0:
            return 0.0
        return (self.chunks_processed / self.chunks_total) * 100

    @property
    def estimated_time_remaining(self) -> timedelta | None:
        if self.chunks_processed == 0:
            return None
        elapsed = time.time() - self.start_time
        rate = self.chunks_processed / max(elapsed, 1e-6)
        remaining = self.chunks_total - self.chunks_processed
        if rate > 0:
            return timedelta(seconds=remaining / rate)
        return None


# =============================================================================
# Helpers
# =============================================================================

def _safe_filename_for_doc(doc_id: str) -> str:
    """Map a doc_id (relative path) to a stable, filesystem-safe filename."""
    # Replace anything except alnum, dot, underscore, space and dash
    safe_id = re.sub(r"[^A-Za-z0-9._ \-]+", "_", doc_id)
    # guard extremely long names
    if len(safe_id) > 128:
        safe_id = safe_id[:128]
    h = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:8]
    return f"{safe_id}.{h}.json"


def _save_chunks_to_path(chunks_dir: Path, doc_id: str, chunks: list[dict], file_size: int) -> Path:
    """Save chunks to JSON file; returns the output path."""
    output_file = chunks_dir / _safe_filename_for_doc(doc_id)
    chunk_data = {
        "doc_id": doc_id,
        "num_chunks": len(chunks),
        "chunks": chunks,
        "metadata": {
            "chunked_at": datetime.now().isoformat(),
            "original_size": file_size,
        },
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)
    return output_file


def _initialize_gcp_credentials() -> str | None:
    """
    Initialize GCP credentials using the centralized config module.
    Returns the path to the credentials file if successful, None otherwise.
    """
    try:
        # Import here to avoid circular dependencies
        from emailops.config import get_config

        config = get_config()

        # Check if credentials are already set in environment
        env_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if env_creds:
            creds_path = Path(env_creds)
            if creds_path.exists():
                logging.info("Using existing GCP credentials from environment: %s", env_creds)
                return str(creds_path)

        # Use config to find credential file
        cred_file = config.get_credential_file()
        if not cred_file:
            secrets_dir = config.get_secrets_dir()
            logging.error("No valid GCP credentials found in secrets directory: %s", secrets_dir)
            logging.info("Checked for credential files in priority order")
            return None

        # Validate and set up credentials
        try:
            with cred_file.open() as f:
                creds_data = json.load(f)
                project_id = creds_data.get("project_id")

            # Set environment variables for GCP authentication
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_file)
            if project_id:
                os.environ["GCP_PROJECT"] = project_id
                os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
                os.environ["VERTEX_PROJECT"] = project_id

            # Set default location if not already set
            os.environ.setdefault("GCP_REGION", config.GCP_REGION)
            os.environ.setdefault("VERTEX_LOCATION", config.VERTEX_LOCATION)

            logging.info("Initialized GCP credentials from %s (project: %s)", cred_file.name, project_id)
            return str(cred_file)

        except json.JSONDecodeError as e:
            logging.error("Invalid JSON in credentials file %s: %s", cred_file, e)
            return None
        except Exception as e:
            logging.error("Error reading credentials file %s: %s", cred_file, e)
            return None

    except ImportError as e:
        logging.error("Failed to import config module: %s", e)
        logging.error("Ensure emailops.config is available")
        return None


# =============================================================================
# Top-level worker entry (must be picklable under 'spawn')
# =============================================================================

def _chunk_worker_entry(worker_id: int,
                        jobs_assigned: list[tuple[str, str, int]],
                        chunk_config: dict[str, Any],
                        chunks_dir: str,
                        stats_queue,
                        control_queue,
                        log_level: int):
    """
    Worker process for chunking (top-level function so it can be pickled).
    """
    # Minimal, worker-local logging (avoid reusing parent's logger object)
    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s - Worker{worker_id} - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger(f"ChunkWorker.{worker_id}")

    try:
        from emailops.text_chunker import ChunkConfig, TextChunker
        from emailops.utils import read_text_file
    except Exception as e:
        # Report an error once and exit
        stats = WorkerStats(
            worker_id=worker_id,
            docs_processed=0,
            docs_total=len(jobs_assigned),
            chunks_created=0,
            bytes_processed=0,
            bytes_total=sum(sz for _, _, sz in jobs_assigned),
            start_time=time.time(),
            last_update=time.time(),
            errors=1,
            status=f"failed: {e}",
        )
        stats_queue.put(stats)
        return

    chunker = TextChunker(ChunkConfig(**chunk_config))
    stats = WorkerStats(
        worker_id=worker_id,
        docs_processed=0,
        docs_total=len(jobs_assigned),
        chunks_created=0,
        bytes_processed=0,
        bytes_total=sum(job[2] for job in jobs_assigned),
        start_time=time.time(),
        last_update=time.time(),
        errors=0,
        status="running",
    )
    stats_queue.put(stats)

    chunks_dir_path = Path(chunks_dir)
    for (doc_id, path_str, file_size) in jobs_assigned:
        # Check for shutdown (non-blocking)
        try:
            while True:
                cmd = control_queue.get_nowait()
                if cmd == "SHUTDOWN":
                    stats.status = "stopped"
                    stats_queue.put(stats)
                    return
        except queue.Empty:
            pass

        stats.current_doc = doc_id
        try:
            text = read_text_file(Path(path_str))
            chunks = chunker.chunk_text(text, metadata={"doc_id": doc_id, "doc_path": path_str})
            if chunks:
                _save_chunks_to_path(chunks_dir_path, doc_id, chunks, file_size)
                stats.chunks_created += len(chunks)
            stats.docs_processed += 1
            stats.bytes_processed += file_size
        except Exception as e:
            logger.error("Error processing %s: %s", path_str, e)
            stats.errors += 1

        stats.last_update = time.time()
        stats_queue.put(stats)

    stats.status = "completed"
    stats_queue.put(stats)


# =============================================================================
# Main Processor Class
# =============================================================================

class UnifiedProcessor:
    """Unified processor for text and embedding operations"""

    def __init__(
        self,
        root_dir: str,
        mode: str = "chunk",  # "chunk", "embed", "repair", "fix"
        num_workers: int | None = None,
        batch_size: int | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        resume: bool = True,
        test_mode: bool = False,
        log_level: str = "INFO",
    ):
        # Import config at instance level to get defaults
        try:
            from emailops.config import get_config
            config = get_config()
        except ImportError:
            # Fallback defaults if config not available
            class FallbackConfig:
                DEFAULT_BATCH_SIZE = 64
                DEFAULT_CHUNK_SIZE = 1600
                DEFAULT_CHUNK_OVERLAP = 200
                DEFAULT_NUM_WORKERS = os.cpu_count() or 4
                CHUNK_DIRNAME = "_chunks"
                INDEX_DIRNAME = "_index"
            config = FallbackConfig()
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.mode = mode
        self.num_workers = num_workers or config.DEFAULT_NUM_WORKERS
        self.batch_size = batch_size or config.DEFAULT_BATCH_SIZE
        self.chunk_size = chunk_size or config.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.DEFAULT_CHUNK_OVERLAP
        self.resume = resume
        self.test_mode = test_mode
        self._log_level = getattr(logging, log_level.upper(), logging.INFO)

        # Setup directories based on mode using config
        if mode == "chunk":
            self.output_dir = self.root_dir / config.CHUNK_DIRNAME
            self.state_dir = self.output_dir / "_chunker_state"
            self.chunks_dir = self.output_dir / "chunks"
        else:  # embedding/repair/fix modes
            self.index_dir = self.root_dir / config.INDEX_DIRNAME
            self.index_dir.mkdir(parents=True, exist_ok=True)

        self.setup_logging()

        # Multiprocessing setup — prefer 'spawn' on all platforms for safety
        self.ctx = mp.get_context("spawn")
        self._shutdown_initiated = False

    # ---------------------------------------------------------------------
    def setup_logging(self):
        """Configure logging"""
        log_dir = (
            self.state_dir
            if hasattr(self, "state_dir")
            else self.index_dir
            if hasattr(self, "index_dir")
            else Path.cwd()
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{self.mode}_{datetime.now():%Y%m%d_%H%M%S}.log"

        logging.basicConfig(
            level=self._log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
            force=True,
        )
        self.logger = logging.getLogger(f"UnifiedProcessor.{self.mode}")

    def close(self):
        """Close logging handlers to release file locks."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    # ---------------------------------------------------------------------
    # Text Chunking Operations
    # ---------------------------------------------------------------------

    def _build_chunk_config_kwargs(self) -> dict[str, Any]:
        """Return kwargs for constructing ChunkConfig in any process."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "respect_sentences": True,
            "respect_paragraphs": True,
            "progressive_scaling": True,
        }

    def chunk_documents(self, input_dir: str, file_pattern: str = "*.txt") -> None:
        """Process documents into chunks"""
        self.input_dir = Path(input_dir)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Find documents to process
        jobs = self._find_documents(file_pattern)
        if not jobs:
            self.logger.info("No new documents to process")
            return

        self.logger.info(f"Found {len(jobs)} documents to process")

        chunk_cfg = self._build_chunk_config_kwargs()
        # Process documents
        if self.num_workers > 1:
            self._parallel_chunk(jobs, chunk_cfg)
        else:
            self._sequential_chunk(jobs, chunk_cfg)

    def _existing_chunk_path(self, doc_id: str) -> Path:
        """Return expected output path for a given doc_id."""
        return self.chunks_dir / _safe_filename_for_doc(doc_id)

    def _find_documents(self, file_pattern: str) -> list[ChunkJob]:
        """Find documents to process"""
        jobs: list[ChunkJob] = []
        skipped = 0
        for file_path in self.input_dir.rglob(file_pattern):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(self.input_dir).as_posix()
            doc_id = rel
            try:
                st = file_path.stat()
                if st.st_size == 0:
                    continue
                # Resume: skip if chunks already exist
                if self.resume and self._existing_chunk_path(doc_id).exists():
                    skipped += 1
                    continue
                jobs.append(ChunkJob(doc_id=doc_id, doc_path=file_path, file_size=st.st_size))
            except Exception as e:
                self.logger.warning(f"Could not stat file {file_path}: {e}")

        jobs.sort(key=lambda j: -j.file_size)  # Process larger files first
        if self.test_mode:
            jobs = jobs[:10]  # Limit to 10 files in test mode
        if skipped:
            self.logger.info("Resume enabled: skipped %d already-chunked files", skipped)
        return jobs

    def _parallel_chunk(self, jobs: list[ChunkJob], chunk_config: dict[str, Any]):
        """Process chunks in parallel"""
        # Distribute work
        worker_configs = self._distribute_chunking_work(jobs, chunk_config)

        # Create queues
        stats_queue = self.ctx.Queue()
        control_queue = self.ctx.Queue()
        workers = []

        # Start workers (top-level function target, not bound method)
        for config in worker_configs:
            worker = self.ctx.Process(
                target=_chunk_worker_entry,
                args=(
                    config.worker_id,
                    config.jobs_assigned,
                    config.chunk_config,
                    str(self.chunks_dir),
                    stats_queue,
                    control_queue,
                    self._log_level,
                ),
            )
            worker.start()
            workers.append(worker)

        # Monitor progress
        try:
            self._monitor_workers(workers, stats_queue, control_queue)
        finally:
            # Ensure workers are collected
            for w in workers:
                w.join(timeout=2)
                if w.is_alive():
                    w.terminate()
                    w.join(timeout=1)

    def _sequential_chunk(self, jobs: list[ChunkJob], chunk_config: dict[str, Any]):
        """Process chunks sequentially"""
        try:
            from emailops.text_chunker import ChunkConfig, TextChunker
            from emailops.utils import read_text_file
        except ImportError as e:
            self.logger.error("Missing dependencies: %s", e)
            self.logger.error("Please ensure the 'emailops' package is installed and available")
            return

        chunker = TextChunker(ChunkConfig(**chunk_config))

        for i, job in enumerate(jobs):
            self.logger.info(f"Processing {i + 1}/{len(jobs)}: {job.doc_id}")
            try:
                text = read_text_file(job.doc_path)
                chunks = chunker.chunk_text(
                    text, metadata={"doc_id": job.doc_id, "doc_path": str(job.doc_path)}
                )
                if chunks:
                    _save_chunks_to_path(self.chunks_dir, job.doc_id, chunks, job.file_size)
            except Exception as e:
                self.logger.error(f"Error processing {job.doc_path}: {e}")

    def _distribute_chunking_work(self, jobs: list[ChunkJob], chunk_config: dict[str, Any]) -> list[WorkerConfig]:
        """Distribute jobs across workers (balanced by file size)"""
        n_workers = min(self.num_workers, len(jobs))
        worker_bins: list[list[ChunkJob]] = [[] for _ in range(n_workers)]
        worker_sizes = [0] * n_workers
        for job in jobs:
            idx = worker_sizes.index(min(worker_sizes))
            worker_bins[idx].append(job)
            worker_sizes[idx] += job.file_size

        configs: list[WorkerConfig] = []
        for i, worker_jobs in enumerate(worker_bins):
            if worker_jobs:
                configs.append(
                    WorkerConfig(
                        worker_id=i,
                        jobs_assigned=[(j.doc_id, str(j.doc_path), j.file_size) for j in worker_jobs],
                        chunk_config=chunk_config,
                    )
                )
        return configs

    # ---------------------------------------------------------------------
    # Embedding Operations
    # ---------------------------------------------------------------------

    def create_embeddings(self, use_chunked_files: bool = True):
        """Create embeddings from chunks or documents"""
        # Initialize GCP credentials first (only needed for Vertex provider)
        provider = os.getenv("EMBED_PROVIDER", "vertex")
        if provider.lower() == "vertex":
            creds_path = _initialize_gcp_credentials()
            if not creds_path:
                self.logger.error("Failed to initialize GCP credentials for Vertex AI")
                self.logger.info("Please ensure valid service account JSON files exist in the 'secrets' directory")
                return

        # Verify dependencies early
        try:
            # 'load_validated_accounts' may be used by downstream code (optional)
            from emailops.llm_runtime import load_validated_accounts  # noqa: F401
        except ImportError:
            # Not fatal; embedding can still proceed if we can call embed_texts
            self.logger.warning("emailops.llm_runtime not found; proceeding without validated accounts")

        try:
            from emailops.llm_client import embed_texts
        except ImportError as e:
            self.logger.error("Missing dependencies: %s", e)
            self.logger.error("Please ensure emailops package is properly installed")
            return

        if use_chunked_files:
            self._embed_from_chunks(embed_texts, provider)
        else:
            self._embed_from_documents(embed_texts, provider)

    def _embed_from_chunks(self, embed_fn, provider: str):
        """Create embeddings from chunk files (single-process, batched)."""
        # Use config for chunk directory name
        try:
            from emailops.config import get_config
            config = get_config()
            chunk_dirname = config.CHUNK_DIRNAME
        except ImportError:
            chunk_dirname = "_chunks"

        chunks_dir = self.root_dir / chunk_dirname / "chunks"
        if not chunks_dir.exists():
            self.logger.error("No chunks directory found at: %s", chunks_dir)
            return

        emb_dir = self.index_dir / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)

        chunk_files = sorted(chunks_dir.glob("*.json"))
        if not chunk_files:
            self.logger.error("No chunk files found in %s", chunks_dir)
            return

        self.logger.info("Embedding %d chunk files (batch size=%d)", len(chunk_files), self.batch_size)
        batch_id = 0
        total_chunks = 0

        def flush_batch(batch_chunks: list[dict[str, Any]]):
            nonlocal batch_id, total_chunks
            if not batch_chunks:
                return

            texts = [c["text"] for c in batch_chunks]
            try:
                embs = embed_fn(texts, provider=provider)
            except Exception as e:
                self.logger.error("Embedding failed for batch %d: %s", batch_id, e)
                return

            embs = np.asarray(embs, dtype="float32")
            data = {"chunks": batch_chunks, "embeddings": embs}
            out_path = emb_dir / f"worker_0_batch_{batch_id:05d}.pkl"
            with out_path.open("wb") as f:
                pickle.dump(data, f)
            self.logger.info("Wrote %s (chunks=%d, dim=%s)", out_path.name, len(batch_chunks), embs.shape[1] if embs.ndim == 2 else "?")
            total_chunks += len(batch_chunks)
            batch_id += 1

        batch: list[dict[str, Any]] = []
        for jf in chunk_files:
            try:
                with jf.open(encoding="utf-8") as f:
                    data = json.load(f)
                doc_id = data.get("doc_id", jf.name)
                for ch in data.get("chunks", []):
                    # Ensure minimal schema
                    if "text" not in ch:
                        continue
                    batch.append({"text": ch["text"], "doc_id": doc_id, **{k: v for k, v in ch.items() if k != "text"}})
                    if len(batch) >= self.batch_size:
                        flush_batch(batch)
                        batch = []
            except Exception as e:
                self.logger.error("Failed to read %s: %s", jf.name, e)

        # Flush any remainder
        flush_batch(batch)
        self.logger.info("Embedding complete. Total chunks embedded: %d", total_chunks)

    def _embed_from_documents(self, embed_fn, provider: str):
        """Create embeddings directly from documents by chunking on the fly."""
        # We'll mirror the chunking config and chunk before embedding.
        try:
            from emailops.text_chunker import ChunkConfig, TextChunker
            from emailops.utils import read_text_file
        except ImportError as e:
            self.logger.error("Missing dependencies for document embedding: %s", e)
            return

        # Find input documents under root_dir
        patterns = ["*.txt", "*.md"]
        doc_paths: list[Path] = []
        for pat in patterns:
            doc_paths.extend(self.root_dir.rglob(pat))
        doc_paths = [p for p in doc_paths if p.is_file()]

        if not doc_paths:
            self.logger.error("No documents found under %s matching %s", self.root_dir, patterns)
            return

        emb_dir = self.index_dir / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)

        chunker = TextChunker(ChunkConfig(**self._build_chunk_config_kwargs()))
        batch: list[dict[str, Any]] = []
        batch_id = 0
        total_chunks = 0

        def flush_batch():
            nonlocal batch, batch_id, total_chunks
            if not batch:
                return
            texts = [c["text"] for c in batch]
            try:
                embs = embed_fn(texts, provider=provider)
            except Exception as e:
                self.logger.error("Embedding failed for batch %d: %s", batch_id, e)
                return

            embs = np.asarray(embs, dtype="float32")
            data = {"chunks": batch, "embeddings": embs}
            out_path = emb_dir / f"worker_0_batch_{batch_id:05d}.pkl"
            with out_path.open("wb") as f:
                pickle.dump(data, f)
            self.logger.info("Wrote %s (chunks=%d, dim=%s)", out_path.name, len(batch), embs.shape[1] if embs.ndim == 2 else "?")
            total_chunks += len(batch)
            batch = []
            batch_id += 1

        for p in sorted(doc_paths):
            try:
                text = read_text_file(p)
                doc_id = p.relative_to(self.root_dir).as_posix()
                chunks = chunker.chunk_text(text, metadata={"doc_id": doc_id, "doc_path": str(p)})
                for ch in chunks:
                    if "text" not in ch:
                        continue
                    batch.append({"text": ch["text"], "doc_id": doc_id, **{k: v for k, v in ch.items() if k != "text"}})
                    if len(batch) >= self.batch_size:
                        flush_batch()
            except Exception as e:
                self.logger.error("Failed to process %s: %s", p, e)

        flush_batch()
        self.logger.info("Document embedding complete. Total chunks embedded: %d", total_chunks)

    # ---------------------------------------------------------------------
    # Monitoring
    # ---------------------------------------------------------------------

    def _monitor_workers(self, workers: list[mp.Process], stats_queue, control_queue):
        """Monitor worker progress with liveness checks and graceful shutdown."""
        worker_states: dict[int, WorkerStats] = {}
        last_stats_time = time.time()

        try:
            while True:
                # Drain stats queue
                drained = False
                while True:
                    try:
                        stats: WorkerStats = stats_queue.get_nowait()
                        worker_states[stats.worker_id] = stats
                        drained = True
                        last_stats_time = time.time()
                    except queue.Empty:
                        break

                # Display progress if we have updates
                if drained:
                    self._display_progress(worker_states)

                # Exit when all workers have exited
                if all(not w.is_alive() for w in workers):
                    break

                # Safety: if no stats for a while but all workers idle, still show something
                if time.time() - last_stats_time > 30:
                    self._display_progress(worker_states)
                    last_stats_time = time.time()

                time.sleep(0.5)

        except KeyboardInterrupt:
            self.logger.warning("Interrupt received. Attempting graceful shutdown...")
            # Signal shutdown to workers
            for _ in workers:
                with contextlib.suppress(Exception):
                    control_queue.put_nowait("SHUTDOWN")
        finally:
            self._print_summary(worker_states)

    def _display_progress(self, worker_states: dict[int, WorkerStats]):
        """Display progress"""
        if sys.stdout.isatty():
            os.system("cls" if os.name == "nt" else "clear")

        print("=" * 80)
        print(f"PROCESSING - {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 80)

        for wid in sorted(worker_states.keys()):
            stats = worker_states[wid]
            eta = stats.estimated_time_remaining
            eta_str = f" ~ETA {str(eta).split('.')[0]}" if eta else ""
            print(f"\nWorker {wid}: [{stats.progress_percent:5.1f}%] {stats.status}{eta_str}")

            if hasattr(stats, "docs_processed"):
                print(f"  Docs: {stats.docs_processed}/{stats.docs_total}")
                print(f"  Chunks: {stats.chunks_created}")
            elif hasattr(stats, "chunks_processed") and hasattr(stats, "chunks_total"):
                print(f"  Chunks: {stats.chunks_processed}/{stats.chunks_total}")

            if stats.errors > 0:
                print(f"  Errors: {stats.errors}")
            if stats.current_doc:
                print(f"  Current: {stats.current_doc}")

    def _print_summary(self, worker_states: dict[int, WorkerStats]):
        """Print final summary"""
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETED")
        print("=" * 80)

        total_processed = 0
        total_errors = 0

        for stats in worker_states.values():
            if hasattr(stats, "docs_processed"):
                total_processed += stats.docs_processed
            elif hasattr(stats, "chunks_processed"):
                total_processed += stats.chunks_processed
            total_errors += stats.errors

        print(f"Total processed: {total_processed}")
        if total_errors > 0:
            print(f"Total errors: {total_errors}")

    # ---------------------------------------------------------------------
    # Index Repair & Fix
    # ---------------------------------------------------------------------

    def repair_index(self, remove_batches: bool = False):
        """Repair/merge batch pickles into final index"""
        emb_dir = self.index_dir / "embeddings"
        if not emb_dir.exists():
            self.logger.error(f"No batch directory found at: {emb_dir}")
            return

        pkl_files = sorted(emb_dir.glob("worker_*_batch_*.pkl"))
        if not pkl_files:
            self.logger.error("No batch pickle files found")
            return

        self.logger.info(f"Merging {len(pkl_files)} batch files...")

        all_chunks = []
        all_embeddings = []

        for pkl_file in pkl_files:
            try:
                with pkl_file.open("rb") as f:
                    data = pickle.load(f)

                chunks = data.get("chunks", [])
                embeddings = np.asarray(data.get("embeddings", []), dtype="float32")

                if len(chunks) == embeddings.shape[0] and embeddings.ndim == 2:
                    all_chunks.extend(chunks)
                    all_embeddings.append(embeddings)
                else:
                    self.logger.warning(f"Shape mismatch in {pkl_file.name}: chunks={len(chunks)} emb={embeddings.shape}")
            except Exception as e:
                self.logger.error(f"Failed to load {pkl_file}: {e}")

        if not all_embeddings:
            self.logger.error("No valid embeddings found")
            return

        # Merge and save
        merged_embeddings = np.vstack(all_embeddings)

        # Save embeddings
        np.save(self.index_dir / "embeddings.npy", merged_embeddings.astype("float32"))

        # Save mapping
        mapping_file = self.index_dir / "mapping.json"
        with mapping_file.open("w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Saved {len(all_chunks)} chunks with embeddings")

        # Try to create FAISS index
        try:
            import faiss
            index = faiss.IndexFlatIP(merged_embeddings.shape[1])
            # Ensure the array is contiguous and float32 before adding
            embeddings_for_index = np.ascontiguousarray(merged_embeddings, dtype=np.float32)
            index.add(embeddings_for_index)
            faiss.write_index(index, str(self.index_dir / "index.faiss"))
            self.logger.info("Created FAISS index")
        except Exception as e:
            self.logger.warning("Could not create FAISS index: %s", e)

        # Clean up batches if requested
        if remove_batches:
            for pkl_file in pkl_files:
                with contextlib.suppress(Exception):
                    pkl_file.unlink()
            self.logger.info(f"Removed {len(pkl_files)} batch files")

    def fix_failed_embeddings(self):
        """Fix chunks with zero vectors"""
        provider = os.getenv("EMBED_PROVIDER", "vertex")
        # Initialize GCP credentials first if using Vertex
        if provider.lower() == "vertex":
            creds_path = _initialize_gcp_credentials()
            if not creds_path:
                self.logger.error("Failed to initialize GCP credentials for re-embedding")
                return

        try:
            from emailops.llm_client import embed_texts
        except ImportError as e:
            self.logger.error("Missing dependencies: %s", e)
            return

        emb_dir = self.index_dir / "embeddings"
        if not emb_dir.exists():
            self.logger.error("No embeddings directory found")
            return

        pkl_files = list(emb_dir.glob("*.pkl"))
        self.logger.info(f"Scanning {len(pkl_files)} files for zero vectors...")

        total_fixed = 0

        for pkl_file in pkl_files:
            try:
                with pkl_file.open("rb") as f:
                    data = pickle.load(f)

                embeddings = np.array(data.get("embeddings", []), dtype="float32")
                chunks = data.get("chunks", [])

                if embeddings.ndim != 2 or len(chunks) != embeddings.shape[0]:
                    self.logger.warning("Skipping %s due to shape mismatch", pkl_file.name)
                    continue

                # Find zero vectors
                zero_mask = np.all(embeddings == 0, axis=1)
                num_zeros = int(np.sum(zero_mask))

                if num_zeros > 0:
                    self.logger.info(f"Found {num_zeros} zero vectors in {pkl_file.name}")
                    failed_indices = np.where(zero_mask)[0]
                    texts = [chunks[i]["text"] for i in failed_indices if "text" in chunks[i]]

                    try:
                        new_embeddings = embed_texts(texts, provider=provider)
                        new_embs = np.array(new_embeddings, dtype="float32")

                        # Validate dimensions
                        if new_embs.ndim != 2 or new_embs.shape[0] != len(texts):
                            self.logger.error("Re-embedding produced unexpected shape %s", new_embs.shape)
                            continue

                        # Replace zeros
                        for idx, failed_idx in enumerate(failed_indices):
                            embeddings[failed_idx] = new_embs[idx]

                        # Save updated
                        data["embeddings"] = embeddings
                        with pkl_file.open("wb") as f:
                            pickle.dump(data, f)

                        self.logger.info(f"Fixed {num_zeros} vectors in {pkl_file.name}")
                        total_fixed += num_zeros
                    except Exception as e:
                        self.logger.error(f"Failed to re-embed: {e}")

            except Exception as e:
                self.logger.error(f"Error processing {pkl_file.name}: {e}")

        self.logger.info(f"Fixed {total_fixed} zero vectors total")

    # =============================================================================
    # CLI Helpers
    # =============================================================================

# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Unified processor for text and embedding operations"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Chunk command
    chunk_parser = subparsers.add_parser("chunk", help="Process documents into chunks")
    chunk_parser.add_argument("--input", required=True, help="Input directory")
    chunk_parser.add_argument("--output", required=True, help="Output directory")
    chunk_parser.add_argument("--workers", type=int, help="Number of workers")
    # Use config defaults if available
    try:
        from emailops.config import get_config
        config = get_config()
        default_chunk_size = config.DEFAULT_CHUNK_SIZE
        default_chunk_overlap = config.DEFAULT_CHUNK_OVERLAP
        default_batch_size = config.DEFAULT_BATCH_SIZE
    except ImportError:
        default_chunk_size = 1600
        default_chunk_overlap = 200
        default_batch_size = 64

    chunk_parser.add_argument("--chunk-size", type=int, default=default_chunk_size)
    chunk_parser.add_argument("--chunk-overlap", type=int, default=default_chunk_overlap)
    chunk_parser.add_argument("--pattern", default="*.txt", help="File pattern (e.g., '*.txt' or '*.md')")
    chunk_parser.add_argument("--test", action="store_true", help="Test mode")
    chunk_parser.add_argument("--no-resume", action="store_true", help="Disable resume (process even if chunks exist)")

    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Create embeddings")
    embed_parser.add_argument("--root", required=True, help="Project root directory")
    embed_parser.add_argument("--batch-size", type=int, default=default_batch_size)
    embed_parser.add_argument("--workers", type=int, help="Number of workers (not used in this impl)")
    # Default to using chunk files; allow opt-out
    embed_parser.add_argument("--from-docs", action="store_true", help="Embed by chunking docs on the fly (skip _chunks)")
    embed_parser.add_argument("--provider", default="vertex", help="Embedding provider (vertex, openai, etc.)")

    # Repair command
    repair_parser = subparsers.add_parser("repair", help="Repair/merge index")
    repair_parser.add_argument("--root", required=True, help="Root directory")
    repair_parser.add_argument("--remove-batches", action="store_true")

    # Fix command
    fix_parser = subparsers.add_parser("fix", help="Fix failed embeddings")
    fix_parser.add_argument("--root", required=True, help="Root directory")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "chunk":
        processor = UnifiedProcessor(
            root_dir=args.output,
            mode="chunk",
            num_workers=args.workers,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            resume=not args.no_resume,
            test_mode=args.test,
        )
        processor.chunk_documents(args.input, args.pattern)

    elif args.command == "embed":
        # Set the embedding provider
        os.environ["EMBED_PROVIDER"] = args.provider
        processor = UnifiedProcessor(
            root_dir=args.root,
            mode="embed",
            num_workers=args.workers,
            batch_size=args.batch_size,
        )
        processor.create_embeddings(use_chunked_files=not args.from_docs)

    elif args.command == "repair":
        processor = UnifiedProcessor(root_dir=args.root, mode="repair")
        processor.repair_index(remove_batches=args.remove_batches)

    elif args.command == "fix":
        processor = UnifiedProcessor(root_dir=args.root, mode="fix")
        processor.fix_failed_embeddings()

    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
