#!/usr/bin/env python3
"""
Unified Text Processing Module for EmailOps
Combines chunking and text processing functionality
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
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from contextlib import suppress

# Core dependencies
from emailops.text_chunker import TextChunker, ChunkConfig
from emailops import text_chunker as text_chunker_module
from emailops.utils import read_text_file

# -------------------------
# Data classes
# -------------------------

@dataclass
class ChunkJob:
    """A chunking job for a single document"""
    doc_id: str
    doc_path: Path
    file_size: int
    priority: int = 0

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
    status: str
    current_doc: Optional[str] = None

    @property
    def progress_percent(self) -> float:
        if self.bytes_total <= 0:
            return 0.0
        p = (self.bytes_processed / max(1, self.bytes_total)) * 100.0
        return min(max(p, 0), 100)

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
# Main Chunker Class
# -------------------------

class TextProcessor:
    """Unified text processing coordinator"""

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
        self._config_fp = self._fingerprint_chunk_config(self.chunk_config)

        self.state_dir = self.output_dir / "_chunker_state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.processed_index_path = self.state_dir / "processed_index.json"
        self.chunks_dir = self.output_dir / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        # Multiprocessing context
        chosen = start_method or os.getenv("CHUNKER_START_METHOD")
        if chosen and chosen in mp.get_all_start_methods():
            ctx_method = chosen
        else:
            ctx_method = "spawn" if os.name == "nt" else ("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
        self.ctx = mp.get_context(ctx_method)

        self._io_lock = self.ctx.Lock()
        self.stats_queue: mp.Queue = self.ctx.Queue()
        self.control_queue: mp.Queue = self.ctx.Queue()
        self.workers: List[mp.Process] = []
        self._shutdown_initiated = False

        self.setup_logging()
        self._install_signal_handlers()

    def setup_logging(self):
        """Configure logging"""
        log_file = self.state_dir / f"chunker_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=self._log_level_no,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
            force=True,
        )
        self.logger = logging.getLogger("TextProcessor")
        self.logger.info("Start method: %s", self.ctx.get_start_method())

    def _install_signal_handlers(self) -> None:
        """Setup graceful shutdown handlers"""
        def _handler(sig, _frame):
            if not self._shutdown_initiated:
                self._shutdown_initiated = True
                self.logger.info("Received signal %s — initiating graceful shutdown...", sig)
                self.shutdown()
        with suppress(Exception):
            signal.signal(signal.SIGINT, _handler)
        with suppress(Exception):
            signal.signal(signal.SIGTERM, _handler)

    # ---- Helper methods ----

    def _fingerprint_chunk_config(self, cfg: ChunkConfig) -> str:
        """Create fingerprint of chunking configuration"""
        try:
            f = Path(text_chunker_module.__file__)
            mtime = int(f.stat().st_mtime)
        except:
            mtime = 0
        
        payload = {
            "cfg": asdict(cfg),
            "algo_mtime": mtime,
            "algo_name": text_chunker_module.__name__,
        }
        s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    def _sanitize_parts(self, rel: str) -> Path:
        """Sanitize path components for safe storage"""
        _SANITIZE_RE = re.compile(r"[^A-Za-z0-9._ -]+")
        p = Path(rel)
        safe_parts = []
        for part in p.parts:
            safe = _SANITIZE_RE.sub("_", part).strip(" .")
            safe = safe if safe else "_"
            safe_parts.append(safe)
        return Path(*safe_parts)

    def _hashed_output_path(self, doc_id: str) -> Path:
        """Create collision-safe output path"""
        safe_rel = self._sanitize_parts(doc_id)
        h = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:8]
        return (self.chunks_dir / safe_rel).with_name(safe_rel.name + f".{h}.json")

    def _atomic_write_json(self, path: Path, payload: Any) -> None:
        """Write JSON atomically"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _load_processed_index(self) -> Dict[str, float]:
        """Load index of processed files"""
        if self.processed_index_path.exists():
            try:
                with self._io_lock:
                    txt = self.processed_index_path.read_text(encoding="utf-8")
                return json.loads(txt)
            except Exception:
                self.logger.warning("Could not load processed index, creating new one")
        return {}

    def _load_chunk_fingerprint(self, path: Path) -> Optional[str]:
        """Read chunk file's config fingerprint"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("metadata", {})
            return meta.get("chunk_config_fingerprint")
        except Exception:
            return None

    # ---- Main workflow methods ----

    def find_documents(self) -> List[ChunkJob]:
        """Find all documents to process"""
        jobs: List[ChunkJob] = []
        processed_index = self._load_processed_index()

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        for file_path in self.input_dir.rglob(self.file_pattern):
            if not file_path.is_file():
                continue

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
                    out_file = self._hashed_output_path(doc_id)
                    fresh_output = out_file.exists() and out_file.stat().st_mtime >= mtime
                    config_match = False
                    if fresh_output:
                        prior_fp = self._load_chunk_fingerprint(out_file)
                        config_match = (prior_fp is not None and prior_fp == self._config_fp)

                    if fresh_in_index and fresh_output and config_match:
                        continue

                jobs.append(ChunkJob(doc_id=doc_id, doc_path=file_path, file_size=file_size))
            except Exception as e:
                self.logger.warning(f"Could not stat file {file_path}: {e}")

        jobs.sort(key=lambda j: (j.priority, -j.file_size))

        if self.test_mode:
            jobs = jobs[:self.test_files]
            self.logger.info(f"TEST MODE: Processing only first {len(jobs)} files")

        self.logger.info(f"Found {len(jobs)} new or modified documents to process")
        return jobs

    def distribute_workload(self, jobs: List[ChunkJob]) -> List[WorkerConfig]:
        """Distribute jobs across workers"""
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

    def run(self):
        """Main execution"""
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
            with suppress(Exception):
                self.stats_queue.close()
                self.stats_queue.join_thread()
            with suppress(Exception):
                self.control_queue.close()
                self.control_queue.join_thread()

    def monitor_workers(self, workers: List[mp.Process]):
        """Monitor worker progress"""
        worker_states: Dict[int, WorkerStats] = {}
        last_render = 0.0

        while True:
            # Drain stats queue
            drained = False
            while True:
                try:
                    stats: WorkerStats = self.stats_queue.get_nowait()
                    worker_states[stats.worker_id] = stats
                    drained = True
                except queue.Empty:
                    break
                except Exception:
                    break

            # Render progress
            now = time.time()
            if drained or (now - last_render > 1.0):
                self.display_progress(worker_states)
                last_render = now

            # Check if all workers done
            if not any(p.is_alive() for p in workers):
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
        """Display progress"""
        if self.clear_screen and sys.stdout.isatty():
            os.system("cls" if os.name == "nt" else "clear")

        print("=" * 80)
        print(f"TEXT PROCESSING - {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 80)

        for wid in sorted(worker_states.keys()):
            stats = worker_states[wid]
            status_symbol = {
                "running": "🔄",
                "completed": "✅",
                "failed": "❌",
                "stopped": "🛑",
            }.get(stats.status, "❓")

            progress_bar = self.create_progress_bar(stats.progress_percent, 30)
            eta = stats.estimated_time_remaining
            eta_str = f" | ETA: {str(eta).split('.')[0]}" if eta else ""

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

    @staticmethod
    def create_progress_bar(percent: float, width: int = 30) -> str:
        """Create text progress bar"""
        pct = min(max(percent, 0), 100)
        filled = int(round(width * pct / 100))
        filled = min(max(filled, 0), width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def print_summary(self, worker_states: Dict[int, WorkerStats]):
        """Print final summary"""
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
        self._atomic_write_json(summary_file, summary)

    def shutdown(self):
        """Gracefully shutdown workers"""
        self.logger.info("Shutting down workers...")
        for _ in self.workers:
            with suppress(Exception):
                self.control_queue.put_nowait("SHUTDOWN")

        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                self.logger.warning(f"Force terminating worker {worker.pid}")
                worker.terminate()


# -------------------------
# Worker Function
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
    """Worker process for chunking documents"""
    # Configure worker logging
    log_file = state_dir / f"worker_{config.worker_id}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s - Worker{config.worker_id} - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger(f"Worker{config.worker_id}")
    
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

    _SANITIZE_RE = re.compile(r"[^A-Za-z0-9._ -]+")

    def _sanitize_parts(rel: str) -> Path:
        p = Path(rel)
        safe_parts = []
        for part in p.parts:
            safe = _SANITIZE_RE.sub("_", part).strip(" .")
            safe = safe if safe else "_"
            safe_parts.append(safe)
        return Path(*safe_parts)

    def _hashed_output_path(doc_id: str) -> Path:
        safe_rel = _sanitize_parts(doc_id)
        h = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:8]
        return (chunks_dir / safe_rel).with_name(safe_rel.name + f".{h}.json")

    def _atomic_write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    for job in config.jobs_assigned:
        # Check for shutdown
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

                output_file = _hashed_output_path(job.doc_id)
                _atomic_write_json(output_file, chunk_data)
                stats.chunks_created += len(chunks)

            stats.docs_processed += 1
            stats.bytes_processed += job.file_size
            stats.last_update = time.time()

            # Update processed index
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
# CLI Entry Point
# -------------------------

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Unified text processor for chunking and analysis"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Chunk command
    chunk_parser = subparsers.add_parser("chunk", help="Process documents into chunks")
    chunk_parser.add_argument("--input-dir", required=True, help="Directory containing documents")
    chunk_parser.add_argument("--output-dir", required=True, help="Directory to save output")
    chunk_parser.add_argument("--workers", type=int, help="Number of worker processes")
    chunk_parser.add_argument("--chunk-size", type=int, default=1600, help="Size of each chunk")
    chunk_parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    chunk_parser.add_argument("--file-pattern", default="*.txt", help="File pattern to process")
    chunk_parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    chunk_parser.add_argument("--test-mode", action="store_true", help="Process only first N files")
    chunk_parser.add_argument("--test-files", type=int, default=100, help="Number of test files")
    chunk_parser.add_argument("--no-clear", action="store_true", help="Don't clear console")
    chunk_parser.add_argument("--start-method", choices=["spawn", "forkserver", "fork"])
    chunk_parser.add_argument("--log-level", default="INFO", help="Logging level")
    chunk_parser.add_argument("--max-chars", type=int, default=0, help="Max chars per file (0=unlimited)")
    
    args = parser.parse_args()
    
    if args.command == "chunk":
        processor = TextProcessor(
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
        processor.run()
    else:
        parser.print_help()


if __name__ == "__main__":
    mp.freeze_support()
    main()