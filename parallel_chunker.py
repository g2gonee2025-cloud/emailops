#!/usr/bin/env python3
"""
Parallel Text Chunker with Multi-Worker Support (production-hardened)

Features:
- Parallel processing across multiple workers
- Smart load balancing for documents of varying sizes
- Resume capability using processed_index.json AND output mtimes + CONFIG FINGERPRINT
- Path-preserving, collision-safe chunk output layout
- Atomic writes for all JSON outputs
- Per-worker logging, robust monitoring, and graceful shutdown
- Configurable start method, log level, and per-file read cap (max chars)

Version: 2025-10-06
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import multiprocessing as mp
import re
import hashlib
import queue
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from contextlib import suppress

# Core dependencies from this project
from emailops.text_chunker import TextChunker, ChunkConfig  # chunk algorithm & config
from emailops import text_chunker as text_chunker_module   # to fingerprint algorithm file
from emailops.utils import read_text_file                  # robust text reader

# -------------------------
# Data classes (pickle-safe)
# -------------------------

@dataclass
class ChunkJob:
    """A chunking job for a single document"""
    doc_id: str
    doc_path: Path
    file_size: int
    priority: int = 0  # Lower number = higher priority

@dataclass
class WorkerConfig:
    """Configuration for a chunking worker"""
    worker_id: int
    jobs_assigned: List[ChunkJob]
    chunk_config: ChunkConfig

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
    status: str  # 'running', 'completed', 'failed', 'stopped'
    current_doc: Optional[str] = None

    @property
    def progress_percent(self) -> float:
        if self.bytes_total <= 0:
            return 0.0
        p = (self.bytes_processed / max(1, self.bytes_total)) * 100.0
        # clamp to [0, 100]
        return 0.0 if p < 0 else (100.0 if p > 100.0 else p)

    @property
    def estimated_time_remaining(self) -> Optional[timedelta]:
        if self.bytes_processed <= 0:
            return None
        elapsed = max(1e-6, time.time() - self.start_time)
        rate = self.bytes_processed / elapsed
        remaining = max(0, self.bytes_total - self.bytes_processed)
        if rate > 0:
            return timedelta(seconds=remaining / rate)
        return None


# -------------------------
# Helpers
# -------------------------

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._ -]+")


def _sanitize_parts(rel: str) -> Path:
    """
    Sanitize each component of a relative path so we can preserve
    the directory structure safely under the chunks/ directory.
    """
    p = Path(rel)
    safe_parts = []
    for part in p.parts:
        safe = _SANITIZE_RE.sub("_", part).strip(" .")
        safe = safe if safe else "_"
        safe_parts.append(safe)
    return Path(*safe_parts)


def _hashed_output_path(chunks_dir: Path, doc_id: str) -> Path:
    """
    Create a collision-safe, path-preserving output path:
      chunks/<sanitized/relative/path>.{hash8}.json
    """
    safe_rel = _sanitize_parts(doc_id)
    h = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:8]
    return (chunks_dir / safe_rel).with_name(safe_rel.name + f".{h}.json")


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically to avoid partial/corrupt files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _configure_worker_logging(worker_id: int, state_dir: Path, log_level: int) -> logging.Logger:
    """Set up separate logging for each worker process."""
    log_file = state_dir / f"worker_{worker_id}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s - Worker{worker_id} - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger(f"Worker{worker_id}")


def _chunker_module_mtime() -> int:
    """Fingerprint the algorithm implementation via module mtime."""
    with suppress(Exception):
        f = Path(text_chunker_module.__file__)
        return int(f.stat().st_mtime)
    return 0


def _fingerprint_chunk_config(cfg: ChunkConfig) -> str:
    """
    Stable SHA-1 fingerprint of the chunking configuration + algorithm mtime.
    Ensures resume will reprocess if config or algorithm changes.
    """
    payload = {
        "cfg": asdict(cfg),
        "algo_mtime": _chunker_module_mtime(),
        "algo_name": text_chunker_module.__name__,
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _load_chunk_fingerprint(path: Path) -> Optional[str]:
    """Read prior chunk file and return its config fingerprint (if present)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        return meta.get("chunk_config_fingerprint")
    except Exception:
        return None


# -------------------------
# Coordinator
# -------------------------

class ParallelChunker:
    """Coordinator for parallel text chunking"""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        num_workers: Optional[int] = None,
        chunk_size: int = 1600,
        chunk_overlap: int = 200,
        file_pattern: str = "*.txt",
        resume: bool = True,
        test_mode: bool = False,
        test_files: int = 100,
        start_method: Optional[str] = None,
        clear_screen: bool = True,
        log_level: str = "INFO",
        max_chars: Optional[int] = None,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers or os.cpu_count() or 1
        self.file_pattern = file_pattern
        self.resume = resume
        self.test_mode = test_mode
        self.test_files = test_files
        self.clear_screen = clear_screen
        self.max_chars = max_chars if (isinstance(max_chars, int) and max_chars > 0) else None
        self._log_level_no = getattr(logging, (log_level or "INFO").upper(), logging.INFO)

        self.chunk_config = ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            respect_sentences=True,
            respect_paragraphs=True,
            progressive_scaling=True,
        )
        self._config_fp = _fingerprint_chunk_config(self.chunk_config)

        self.state_dir = self.output_dir / "_chunker_state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.processed_index_path = self.state_dir / "processed_index.json"
        self.chunks_dir = self.output_dir / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        # Multiprocessing context selection
        chosen = start_method or os.getenv("CHUNKER_START_METHOD")
        if chosen:
            # Validate explicitly requested method
            if chosen not in mp.get_all_start_methods():
                raise ValueError(f"Unsupported start method: {chosen}")
            ctx_method = chosen
        else:
            # Safe default: spawn on Windows; forkserver if available on POSIX; else spawn
            if os.name == "nt":
                ctx_method = "spawn"
            else:
                ctx_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        self.ctx = mp.get_context(ctx_method)

        self._io_lock = self.ctx.Lock()
        self.stats_queue: mp.Queue = self.ctx.Queue()
        self.control_queue: mp.Queue = self.ctx.Queue()
        self.workers: List[mp.Process] = []

        self._shutdown_initiated = False

        self.setup_logging()
        self._install_signal_handlers()

    def setup_logging(self):
        """Configure coordinator logging."""
        log_file = self.state_dir / f"chunker_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=self._log_level_no,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
            force=True,
        )
        self.logger = logging.getLogger("ParallelChunker")
        self.logger.info("Start method: %s", self.ctx.get_start_method())

    def _install_signal_handlers(self) -> None:
        """Trap SIGINT/SIGTERM for graceful shutdown."""
        def _handler(sig, _frame):
            if not self._shutdown_initiated:
                self._shutdown_initiated = True
                self.logger.info("Received signal %s — initiating graceful shutdown...", sig)
                self.shutdown()
        with suppress(Exception):
            signal.signal(signal.SIGINT, _handler)
        with suppress(Exception):
            signal.signal(signal.SIGTERM, _handler)

    # ---------- Resume / discovery ----------

    def _output_path_for(self, doc_id: str) -> Path:
        """Mirror the worker's output path logic for consistent resume checks."""
        return _hashed_output_path(self.chunks_dir, doc_id)

    def _load_processed_index(self) -> Dict[str, float]:
        """Load the index of processed files and their modification times."""
        if self.processed_index_path.exists():
            try:
                with self._io_lock:
                    txt = self.processed_index_path.read_text(encoding="utf-8")
                return json.loads(txt)
            except Exception:
                self.logger.warning("Could not load processed index, creating a new one.")
        return {}

    def find_documents(self) -> List[ChunkJob]:
        """
        Find all documents to process. When resuming, we skip files **only if**:
          - processed_index says it's up-to-date, AND
          - output chunk file is newer than the source, AND
          - output chunk file's config fingerprint matches current config.
        Otherwise we reprocess to guarantee correctness on config/algorithm changes.
        """
        jobs: List[ChunkJob] = []
        processed_index = self._load_processed_index()

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        for file_path in self.input_dir.rglob(self.file_pattern):
            if not file_path.is_file():
                continue

            # POSIX-normalized doc_id for portability across OSes
            rel = file_path.relative_to(self.input_dir).as_posix()
            doc_id = rel
            try:
                st = file_path.stat()
                mtime = st.st_mtime
                file_size = st.st_size
                if file_size == 0:
                    continue

                if self.resume:
                    fresh_in_index = processed_index.get(doc_id, 0) >= mtime

                    out_file = self._output_path_for(doc_id)
                    fresh_output = out_file.exists() and out_file.stat().st_mtime >= mtime
                    config_match = False
                    if fresh_output:
                        prior_fp = _load_chunk_fingerprint(out_file)
                        config_match = (prior_fp is not None and prior_fp == self._config_fp)

                    if fresh_in_index and fresh_output and config_match:
                        # Truly up-to-date: skip
                        continue

                jobs.append(ChunkJob(doc_id=doc_id, doc_path=file_path, file_size=file_size))
            except Exception as e:
                self.logger.warning(f"Could not stat file {file_path}: {e}")

        # Greedy bin packing by size (descending), respecting potential priority
        jobs.sort(key=lambda j: (j.priority, -j.file_size))

        if self.test_mode:
            jobs = jobs[: self.test_files]
            self.logger.info(f"TEST MODE: Processing only first {len(jobs)} files")

        self.logger.info(f"Found {len(jobs)} new or modified documents to process")
        return jobs

    # ---------- Work distribution ----------

    def distribute_workload(self, jobs: List[ChunkJob]) -> List[WorkerConfig]:
        """Distribute jobs across workers using a greedy bin packing by bytes."""
        worker_count = min(self.num_workers, max(1, len(jobs)))
        worker_bins: List[List[ChunkJob]] = [[] for _ in range(worker_count)]
        worker_sizes = [0] * worker_count

        for job in jobs:
            idx = worker_sizes.index(min(worker_sizes))
            worker_bins[idx].append(job)
            worker_sizes[idx] += job.file_size

        configs: List[WorkerConfig] = []
        for i, worker_jobs in enumerate(worker_bins):
            if worker_jobs:
                configs.append(
                    WorkerConfig(
                        worker_id=i,
                        jobs_assigned=worker_jobs,
                        chunk_config=self.chunk_config,
                    )
                )
                self.logger.info(
                    f"Worker {i}: {len(worker_jobs)} documents, {worker_sizes[i] / 1024 / 1024:.1f} MB total"
                )

        return configs

    # ---------- Run / Monitor ----------

    def run(self):
        """Main execution method."""
        self.logger.info(
            "Starting parallel chunking with up to %s workers (pattern: %s)",
            self.num_workers, self.file_pattern
        )
        try:
            jobs = self.find_documents()
            if not jobs:
                self.logger.info("No new or modified documents to process.")
                return

            worker_configs = self.distribute_workload(jobs)

            # Start workers
            for config in worker_configs:
                worker = self.ctx.Process(
                    target=chunk_worker,
                    name=f"chunk-worker-{config.worker_id}",
                    args=(
                        config,
                        self.stats_queue,
                        self.control_queue,
                        self.chunks_dir,
                        self.state_dir,
                        self._io_lock,
                        self._log_level_no,
                        self.max_chars,
                        self._config_fp,
                    ),
                )
                worker.daemon = False
                worker.start()
                self.workers.append(worker)

            self.monitor_workers(self.workers)

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
            self.shutdown()
        except Exception as e:
            self.logger.error(f"Coordinator error: {e}", exc_info=True)
            self.shutdown()
        finally:
            # Best-effort queue cleanup
            with suppress(Exception):
                self.stats_queue.close()
                self.stats_queue.join_thread()
            with suppress(Exception):
                self.control_queue.close()
                self.control_queue.join_thread()

    def monitor_workers(self, workers: List[mp.Process]):
        """Monitor and display progress without hanging if a worker never reports."""
        worker_states: Dict[int, WorkerStats] = {}
        last_render = 0.0

        while True:
            # Drain the stats queue without blocking the whole loop
            drained = False
            while True:
                try:
                    stats: WorkerStats = self.stats_queue.get_nowait()
                    worker_states[stats.worker_id] = stats
                    drained = True
                except queue.Empty:
                    break
                except Exception:
                    break  # tolerate queue errors during shutdown

            # Render at ~1 Hz (or on first stats received)
            now = time.time()
            if drained or (now - last_render > 1.0):
                self.display_progress(worker_states)
                last_render = now

            # Stop when all workers are no longer alive and queue is drained
            if not any(p.is_alive() for p in workers):
                # One last drain in case of final messages en route
                while True:
                    try:
                        stats = self.stats_queue.get_nowait()
                        worker_states[stats.worker_id] = stats
                    except queue.Empty:
                        break
                    except Exception:
                        break
                break

            time.sleep(0.1)

        # Join workers
        for p in workers:
            p.join()

        self.print_summary(worker_states)

    def display_progress(self, worker_states: Dict[int, WorkerStats]):
        """Display progress from all workers."""
        if self.clear_screen and sys.stdout.isatty():
            os.system("cls" if os.name == "nt" else "clear")

        print("=" * 80)
        print(f"PARALLEL TEXT CHUNKING - {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 80)

        for wid in sorted(worker_states.keys()):
            self._display_worker_stats(worker_states[wid])

        total_docs = sum(s.docs_processed for s in worker_states.values())
        total_chunks = sum(s.chunks_created for s in worker_states.values())
        total_bytes = sum(s.bytes_processed for s in worker_states.values())
        total_errors = sum(s.errors for s in worker_states.values())

        print("\n" + "=" * 80)
        print(
            f"OVERALL: {total_docs} documents, {total_chunks} chunks, "
            f"{total_bytes / 1024 / 1024:.1f} MB processed"
        )
        if total_errors > 0:
            print(f"Total Errors: {total_errors}")
        print("=" * 80)

    def _display_worker_stats(self, stats: WorkerStats):
        """Display stats for a single worker."""
        status_symbol = {
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
            "stopped": "🛑",
        }.get(stats.status, "❓")

        progress_bar = self.create_progress_bar(stats.progress_percent, 30)

        eta = stats.estimated_time_remaining
        eta_str = f" | ⏳ ETA: {str(eta).split('.')[0]}" if eta else ""

        print(f"\n{status_symbol} Worker {stats.worker_id}: {progress_bar} {stats.progress_percent:5.1f}%{eta_str}")
        print(
            f"   📄 {stats.docs_processed}/{stats.docs_total} docs | "
            f"📦 {stats.chunks_created} chunks | "
            f"💾 {stats.bytes_processed / 1024 / 1024:.1f} MB"
        )

        if stats.current_doc:
            print(f"   🔨 Processing: {Path(stats.current_doc).name}")

        if stats.errors > 0:
            print(f"   ⚠️ Errors: {stats.errors}")

    @staticmethod
    def create_progress_bar(percent: float, width: int = 30) -> str:
        """Create a text progress bar."""
        pct = 0 if percent < 0 else (100 if percent > 100 else percent)
        filled = int(round(width * pct / 100))
        filled = min(max(filled, 0), width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def print_summary(self, worker_states: Dict[int, WorkerStats]):
        """Print final summary."""
        print("\n" + "=" * 80)
        print("CHUNKING COMPLETED")
        print("=" * 80)

        total_docs = sum(s.docs_processed for s in worker_states.values())
        total_chunks = sum(s.chunks_created for s in worker_states.values())
        total_bytes = sum(s.bytes_processed for s in worker_states.values())
        total_errors = sum(s.errors for s in worker_states.values())

        print(f"\nTotal Documents Processed: {total_docs}")
        print(f"Total Chunks Created: {total_chunks}")
        print(f"Total Data Processed: {total_bytes / 1024 / 1024:.1f} MB")

        if total_docs > 0 and total_chunks > 0:
            print(f"Average Chunks per Document: {total_chunks / total_docs:.1f}")

        if total_errors > 0:
            print(f"\nTotal Errors: {total_errors}")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_docs": total_docs,
            "total_chunks": total_chunks,
            "total_bytes": total_bytes,
            "total_errors": total_errors,
            "workers": len(worker_states),
        }

        summary_file = self.state_dir / "chunking_summary.json"
        _atomic_write_json(summary_file, summary)

    def shutdown(self):
        """Gracefully shutdown all workers."""
        self.logger.info("Shutting down workers...")
        # Push exactly one token per started worker
        for _ in self.workers:
            with suppress(Exception):
                self.control_queue.put_nowait("SHUTDOWN")

        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                self.logger.warning(f"Force terminating worker {worker.pid}")
                worker.terminate()


# -------------------------
# Worker
# -------------------------

def chunk_worker(
    config: WorkerConfig,
    stats_queue: mp.Queue,
    control_queue: mp.Queue,
    chunks_dir: Path,
    state_dir: Path,
    io_lock: mp.Lock,
    log_level: int,
    max_chars: Optional[int],
    config_fingerprint: str,
):
    """
    Worker process that handles chunking for assigned documents.
    """
    logger = _configure_worker_logging(config.worker_id, state_dir, log_level)
    chunker = TextChunker(config.chunk_config)

    total_bytes = sum(job.file_size for job in config.jobs_assigned)
    stats = WorkerStats(
        worker_id=config.worker_id,
        docs_processed=0,
        docs_total=len(config.jobs_assigned),
        chunks_created=0,
        bytes_processed=0,
        bytes_total=total_bytes,
        start_time=time.time(),
        last_update=time.time(),
        errors=0,
        status="running",
    )

    stats_queue.put(stats)
    last_sent_ts = time.time()

    logger.info(f"Worker {config.worker_id} started with {len(config.jobs_assigned)} documents")

    for job in config.jobs_assigned:
        # Non-blocking control channel poll (tolerant to queue state)
        cmd = None
        try:
            cmd = control_queue.get_nowait()
        except queue.Empty:
            pass
        except Exception:
            pass

        if cmd == "SHUTDOWN":
            logger.info("Received shutdown signal")
            stats.status = "stopped"
            stats_queue.put(stats)
            return

        stats.current_doc = job.doc_id
        stats_queue.put(stats)

        try:
            text = read_text_file(job.doc_path, max_chars=max_chars)
            chunks = chunker.chunk_text(
                text, metadata={"doc_id": job.doc_id, "doc_path": str(job.doc_path)}
            )

            if chunks:
                chunk_data = {
                    "doc_id": job.doc_id,
                    "doc_path": str(job.doc_path),
                    "num_chunks": len(chunks),
                    "chunks": chunks,
                    "metadata": {
                        "chunked_at": datetime.now().isoformat(),
                        "chunk_config": asdict(config.chunk_config),
                        "chunk_config_fingerprint": config_fingerprint,
                        "original_size": job.file_size,
                    },
                }

                output_file = _hashed_output_path(chunks_dir, job.doc_id)
                _atomic_write_json(output_file, chunk_data)

                stats.chunks_created += len(chunks)

            stats.docs_processed += 1
            stats.bytes_processed += job.file_size
            stats.last_update = time.time()

            # Update resume index atomically
            try:
                with io_lock:
                    processed_index_path = state_dir / "processed_index.json"
                    idx: Dict[str, float] = {}
                    if processed_index_path.exists():
                        try:
                            idx = json.loads(processed_index_path.read_text(encoding="utf-8"))
                        except Exception:
                            logger.warning("Could not decode processed_index.json, starting fresh.")
                    idx[job.doc_id] = job.doc_path.stat().st_mtime
                    tmp = processed_index_path.with_suffix(processed_index_path.suffix + ".tmp")
                    tmp.write_text(json.dumps(idx, indent=2), encoding="utf-8")
                    os.replace(tmp, processed_index_path)
            except Exception as e:
                logger.warning(f"Failed to update processed index for {job.doc_id}: {e}")

            if (stats.docs_processed % 10 == 0) or (time.time() - last_sent_ts > 5):
                stats_queue.put(stats)
                last_sent_ts = time.time()

        except Exception as e:
            logger.error(f"Error processing {job.doc_path}: {e}", exc_info=True)
            stats.errors += 1

    stats.status = "completed"
    stats.current_doc = None
    stats_queue.put(stats)
    logger.info(f"Worker {config.worker_id} completed")


# -------------------------
# CLI
# -------------------------

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parallel Text Chunker for processing multiple documents"
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing documents to chunk")
    parser.add_argument("--output-dir", required=True, help="Directory to save chunked output")
    parser.add_argument("--workers", type=int, help="Number of worker processes (default: CPU count)")
    parser.add_argument("--chunk-size", type=int, default=1600, help="Size of each chunk in characters")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument("--file-pattern", default="*.txt", help="Glob pattern for files to process")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, don't resume from previous state")
    parser.add_argument("--test-mode", action="store_true", help="Test mode: process only first N files")
    parser.add_argument("--test-files", type=int, default=100, help="Number of files to process in test mode")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the console while running")
    parser.add_argument("--start-method", choices=["spawn", "forkserver", "fork"], help="Multiprocessing start method")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--max-chars", type=int, default=int(os.getenv("MAX_DOC_CHARS", "0")),
                        help="Hard cap on characters read per file (0 = unlimited)")
    args = parser.parse_args()

    chunker = ParallelChunker(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        file_pattern=args.file_pattern,
        resume=not args.no_resume,
        test_mode=args.test_mode,
        test_files=args.test_files,
        clear_screen=not args.no_clear,
        start_method=args.start_method,
        log_level=args.log_level,
        max_chars=(args.max_chars if args.max_chars > 0 else None),
    )
    chunker.run()


if __name__ == "__main__":
    mp.freeze_support()
    main()
