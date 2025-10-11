#!/usr/bin/env python3
"""
Unified Processing Module for EmailOps (patched, critical fixes only)

Fixes included here (focus: critical issues):
- 🛠️ Multiprocessing safety: pass pure-data dicts to workers; reconstruct ChunkConfig in worker.
- 🧠 Robust embedding dim probe: no silent 768 fallback. Probe with "x"; optional EMBED_DIM override; fail-fast on probe errors.
- 🧩 Zero-vector repair ID parity: when chunk_index is absent in chunk JSONs, use a per-doc counter (matching _chunks_to_mapping) so `fix` can actually repair.
- 🪟 Windows memmap safety: close/del memmaps before os.replace() during truncation to avoid PermissionError on Windows.
- 🧭 Monitor idle timeout: configurable via EMAILOPS_MONITOR_IDLE_TIMEOUT; on timeout, politely SHUTDOWN chunk workers.
- 🧮 FAISS guard: skip index build when too many zero vectors (>5%) to avoid degenerate cosine/IP behavior.
- 🔁 Deterministic-ish inputs: load chunk JSON files in sorted order to reduce non-determinism.

Nice-to-haves (ordering stability, absolute-path redaction, long-hash filenames, etc.) are intentionally left for later.

Artifacts emitted (compatible with downstream tools):
  <root>/_index/
    - embeddings.npy      (float32, row-aligned to mapping.json)
    - mapping.json        (id/path/snippet/subject/date/... schema)
    - index.faiss         (optional)
    - meta.json           (via emailops.index_metadata)
  <root>/_chunks/chunks/  (intermediate chunk JSONs)
"""

from __future__ import annotations

import os
import sys
import json
import time
import pickle
import logging
import argparse
import multiprocessing as mp
import numpy as np
import re
import hashlib
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import queue
import shutil  # for --force rechunk
from collections import defaultdict

# ----------------------------------------------------------------------------- #
# Dataclasses
# ----------------------------------------------------------------------------- #

@dataclass
class ChunkJob:
    """A chunking job for a single document."""
    doc_id: str
    doc_path: Path
    file_size: int
    priority: int = 0

@dataclass
class WorkerConfig:
    """Configuration for a chunking worker (pure-data for safe pickling)."""
    worker_id: int
    jobs_assigned: List[ChunkJob]
    chunks_dir: str
    chunk_config: Dict[str, Any]  # pure-data dict; reconstructed in worker
    resume: bool = True

@dataclass
class WorkerStats:
    """Statistics for monitoring chunking progress."""
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
        return min(100.0, max(0.0, (self.bytes_processed / max(1, self.bytes_total)) * 100.0))

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

@dataclass
class ProcessingStats:
    """Statistics for monitoring embedding progress."""
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
        return (self.chunks_processed / self.chunks_total) * 100.0

    @property
    def estimated_time_remaining(self) -> Optional[timedelta]:
        if self.chunks_processed == 0:
            return None
        elapsed = time.time() - self.start_time
        rate = self.chunks_processed / max(elapsed, 1e-6)
        remaining = self.chunks_total - self.chunks_processed
        if rate > 0:
            return timedelta(seconds=remaining / rate)
        return None

# ----------------------------------------------------------------------------- #
# Small helpers (module-level for picklability)
# ----------------------------------------------------------------------------- #

def _safe_logger(name: str) -> logging.Logger:
    """Get a logger without reconfiguring global handlers (works in workers)."""
    return logging.getLogger(name)

def _sanitize_doc_id(doc_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ \-\/]+", "_", doc_id)

def _chunk_output_path(chunks_dir: Path, doc_id: str) -> Path:
    """
    Use a short, hash-based filename to avoid long path issues.
    Note: 16 hex prefix is kept for backward compatibility.
    """
    h = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()
    return chunks_dir / f"{h[:16]}.json"

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _utc_iso(dt: float) -> str:
    return datetime.fromtimestamp(dt, tz=timezone.utc).isoformat()

def _file_mtime_iso(p: Path) -> Optional[str]:
    try:
        return _utc_iso(p.stat().st_mtime)
    except Exception:
        return None

def _load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _atomic_write_json(out_path: Path, record: dict, retries: int = 5, delay: float = 0.1) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    _ensure_dir(out_path.parent)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    for _ in range(retries):
        try:
            tmp.replace(out_path)
            return
        except Exception:
            time.sleep(delay)
    # Last attempt: raise
    tmp.replace(out_path)

def _save_chunks_to_dir(chunks_dir: Path, doc_id: str, chunks: List[Dict[str, Any]], file_size: int) -> Path:
    """Write one chunk JSON file atomically with retry."""
    out_path = _chunk_output_path(chunks_dir, doc_id)
    record = {
        "doc_id": doc_id,
        "num_chunks": len(chunks),
        "chunks": chunks,
        "metadata": {
            "chunked_at": datetime.now(timezone.utc).isoformat(),
            "original_size": int(file_size),
        },
    }
    _atomic_write_json(out_path, record)
    return out_path

def _should_skip_as_up_to_date(chunks_dir: Path, job: ChunkJob) -> bool:
    """Return True if an existing chunk file looks up-to-date for this job."""
    out_path = _chunk_output_path(chunks_dir, job.doc_id)
    if not out_path.exists():
        return False
    data = _load_json(out_path) or {}
    meta = data.get("metadata") or {}
    try:
        chunk_ts = datetime.fromisoformat(str(meta.get("chunked_at", "")))
        if chunk_ts.tzinfo is None:
            chunk_ts = chunk_ts.replace(tzinfo=timezone.utc)
    except Exception:
        return False
    try:
        src_mtime = job.doc_path.stat().st_mtime
        src_dt = datetime.fromtimestamp(src_mtime, tz=timezone.utc)
    except Exception:
        return False
    if int(meta.get("original_size", -1)) == job.file_size and chunk_ts >= src_dt:
        return True
    return False

def _chunks_to_mapping(
    chunks: Iterable[Dict[str, Any]],
    project_root: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Convert chunk dictionaries (from TextChunker) to mapping.json entries.
    - Ensures unique IDs even if chunk_index is missing by maintaining a per-doc counter.
    - Stores paths relative to project_root when possible.
    """
    mapping: List[Dict[str, Any]] = []
    per_doc_counter: Dict[str, int] = defaultdict(int)

    root_real = project_root.resolve() if project_root else None

    for ch in chunks:
        text = str(ch.get("text", ""))
        meta = ch.get("metadata", {}) or {}
        doc_id = str(meta.get("doc_id") or "")
        doc_path = str(meta.get("doc_path") or "")
        if "chunk_index" in ch:
            chunk_index = int(ch.get("chunk_index", 0))
        else:
            chunk_index = per_doc_counter[doc_id]
            per_doc_counter[doc_id] += 1

        if not doc_id:
            doc_id = Path(doc_path).as_posix()
        rec_id = f"{doc_id}::chunk{chunk_index}"
        conv_id = doc_id.split("/", 1)[0] if "/" in doc_id else doc_id

        date_iso: Optional[str]
        try:
            mt = Path(doc_path).stat().st_mtime
            date_iso = datetime.fromtimestamp(mt, tz=timezone.utc).isoformat()
        except Exception:
            date_iso = None

        rel_path = doc_path
        try:
            if root_real:
                p_real = Path(doc_path).resolve()
                rel_path = p_real.relative_to(root_real).as_posix()
        except Exception:
            rel_path = Path(doc_path).as_posix() if doc_path else ""

        mapping.append({
            "id": rec_id,
            "path": rel_path,
            "conv_id": conv_id,
            "doc_type": "document",
            "subject": Path(doc_path).name if doc_path else "",
            "date": date_iso,
            "start_date": None,
            "end_date": None,
            "from_email": "",
            "from_name": "",
            "to_emails": [],
            "cc_emails": [],
            "participants": [],
            "attachment_name": None,
            "attachment_type": None,
            "attachment_size": None,
            "snippet": text[:500],
        })
    return mapping

def _probe_embedding_dim(provider: str) -> int:
    """
    Probe the embedding width. If EMBED_DIM or EMAILOPS_EMBED_DIM is set, use it.
    Otherwise, call provider once with a non-empty string and validate.
    Fail fast on errors (no silent 768 fallback).
    """
    dim_env = os.getenv("EMBED_DIM") or os.getenv("EMAILOPS_EMBED_DIM")
    if dim_env:
        try:
            dim = int(dim_env)
            if dim <= 0:
                raise ValueError
            return dim
        except Exception:
            raise RuntimeError(f"Invalid EMBED_DIM value: {dim_env}")

    from emailops.llm_client import embed_texts  # type: ignore
    arr = np.asarray(embed_texts(["x"], provider=provider), dtype="float32")
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] <= 0:
        raise RuntimeError("Could not probe embedding dimension from provider.")
    return int(arr.shape[1])

# ----------------------------------------------------------------------------- #
# Worker entry points (top-level, safe for multiprocessing)
# ----------------------------------------------------------------------------- #

def _chunk_worker_entry(config: WorkerConfig, stats_queue, control_queue) -> None:
    """Process a batch of documents into chunks (single worker)."""
    log = _safe_logger(f"processor.chunk.worker.{config.worker_id}")

    # Initialize stats and publish "starting" before risky imports
    stats = WorkerStats(
        worker_id=config.worker_id,
        docs_processed=0,
        docs_total=len(config.jobs_assigned),
        chunks_created=0,
        bytes_processed=0,
        bytes_total=sum(j.file_size for j in config.jobs_assigned),
        start_time=time.time(),
        last_update=time.time(),
        errors=0,
        status="starting",
    )
    stats_queue.put(stats)

    # Lazy imports with error signaling
    try:
        from emailops.text_chunker import TextChunker, ChunkConfig  # type: ignore
        from emailops.utils import read_text_file                     # type: ignore
    except Exception as e:
        log.error("Import error in worker: %s", e)
        stats.errors += 1
        stats.status = "failed"
        stats.last_update = time.time()
        stats_queue.put(stats)
        return

    try:
        chunker = TextChunker(ChunkConfig(**config.chunk_config))
    except Exception as e:
        log.error("Failed to construct ChunkConfig: %s", e)
        stats.errors += 1
        stats.status = "failed"
        stats.last_update = time.time()
        stats_queue.put(stats)
        return

    chunks_dir = Path(config.chunks_dir)

    stats.status = "running"
    stats.last_update = time.time()
    stats_queue.put(stats)

    for job in config.jobs_assigned:
        # Cooperative shutdown
        try:
            if not control_queue.empty():
                cmd = control_queue.get_nowait()
                if cmd == "SHUTDOWN":
                    stats.status = "stopped"
                    stats_queue.put(stats)
                    return
        except Exception:
            pass

        stats.current_doc = job.doc_id
        try:
            # Resume support: skip if output looks up-to-date
            if config.resume and _should_skip_as_up_to_date(chunks_dir, job):
                stats.docs_processed += 1
                stats.bytes_processed += job.file_size
                stats.last_update = time.time()
                stats_queue.put(stats)
                continue

            raw = read_text_file(job.doc_path)
            chunks = chunker.chunk_text(raw, metadata={
                "doc_id": job.doc_id,
                "doc_path": str(job.doc_path)
            })
            if chunks:
                _save_chunks_to_dir(chunks_dir, job.doc_id, chunks, job.file_size)
                stats.chunks_created += len(chunks)

            stats.docs_processed += 1
            stats.bytes_processed += job.file_size
        except Exception as e:
            log.error("Error processing %s: %s", job.doc_path, e)
            stats.errors += 1

        stats.last_update = time.time()
        stats_queue.put(stats)

    stats.status = "completed"
    stats_queue.put(stats)

def _embedding_worker_entry(
    index_dir: str,
    worker_id: int,
    indices: List[int],
    all_chunks: List[Dict[str, Any]],
    stats_queue,
    batch_size: int
) -> None:
    """Embed a subset of chunks and write them to a worker-local pickle file."""
    log = _safe_logger(f"processor.embed.worker.{worker_id}")

    # Post "starting" status before risky imports
    stats = ProcessingStats(
        worker_id=worker_id,
        project_id=str(index_dir),
        chunks_processed=0,
        chunks_total=len(indices),
        start_time=time.time(),
        last_update=time.time(),
        errors=0,
        status="starting",
        account_group=0,
    )
    stats_queue.put(stats)

    try:
        from emailops.llm_client import embed_texts  # type: ignore
    except Exception as e:
        log.error("Import error in embed worker: %s", e)
        stats.errors += 1
        stats.status = "failed"
        stats.last_update = time.time()
        stats_queue.put(stats)
        return

    emb_dir = Path(index_dir) / "embeddings"
    _ensure_dir(emb_dir)

    provider = os.getenv("EMBED_PROVIDER", "vertex")
    try:
        emb_dim = _probe_embedding_dim(provider)
    except Exception as e:
        log.error("Embedding dimension probe failed: %s", e)
        stats.errors += 1
        stats.status = "failed"
        stats.last_update = time.time()
        stats_queue.put(stats)
        return

    stats.status = "running"
    stats.last_update = time.time()
    stats_queue.put(stats)

    batch_counter = 0

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_chunks = [all_chunks[i] for i in batch_indices]
        texts = [str(c.get("text", "")) for c in batch_chunks]

        try:
            arr = np.asarray(embed_texts(texts, provider=provider), dtype="float32")
            if arr.ndim != 2 or int(arr.shape[1]) != emb_dim:
                raise RuntimeError(f"Embedding provider returned shape {arr.shape}, expected (*, {emb_dim})")
        except Exception as e:
            # Best-effort: emit zeros so pipeline can continue
            log.error("Embedding batch failed: %s", e)
            arr = np.zeros((len(texts), emb_dim), dtype="float32")
            stats.errors += 1

        out_pkl = emb_dir / f"worker_{worker_id}_batch_{batch_counter}.pkl"
        try:
            with out_pkl.open("wb") as f:
                pickle.dump({"chunks": batch_chunks, "embeddings": arr}, f)
        except Exception as e:
            log.error("Failed to write batch pickle: %s", e)
            stats.errors += 1
        batch_counter += 1

        stats.chunks_processed += len(batch_indices)
        stats.last_update = time.time()
        stats_queue.put(stats)

    stats.status = "completed"
    stats_queue.put(stats)

# ----------------------------------------------------------------------------- #
# Main processor
# ----------------------------------------------------------------------------- #

class UnifiedProcessor:
    """Unified processor for text chunking and embedding operations."""

    def __init__(
        self,
        root_dir: str,
        mode: str = "chunk",  # "chunk", "embed", "repair", "fix"
        num_workers: Optional[int] = None,
        batch_size: int = 64,
        chunk_size: int = 1600,
        chunk_overlap: int = 200,
        resume: bool = True,
        test_mode: bool = False,
        log_level: str = "INFO",
        force_rechunk: bool = False,
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.mode = mode
        self.num_workers = int(num_workers or (os.cpu_count() or 1))
        self.batch_size = int(batch_size)
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.resume = bool(resume)
        self.test_mode = bool(test_mode)
        self._log_level = getattr(logging, str(log_level).upper(), logging.INFO)
        self.force_rechunk = bool(force_rechunk)

        if mode == "chunk":
            self.output_dir = self.root_dir / "_chunks"
            self.state_dir = self.output_dir / "_chunker_state"
            self.chunks_dir = self.output_dir / "chunks"
        else:
            self.index_dir = self.root_dir / "_index"
            _ensure_dir(self.index_dir)

        self.setup_logging()

        # Multiprocessing context: allow override via env for portability
        desired = os.getenv("EMAILOPS_MP_START_METHOD")
        if desired:
            if desired not in mp.get_all_start_methods():
                self.logger.warning("Requested start method %s not available; falling back.", desired)
                desired = None
        if not desired:
            desired = "spawn" if os.name == "nt" else ("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
        self.ctx = mp.get_context(desired)

    # ---- Logging ----

    def setup_logging(self) -> None:
        """Configure logging for this processor instance without clobbering global handlers."""
        if hasattr(self, "state_dir"):
            log_dir = self.state_dir
        elif hasattr(self, "index_dir"):
            log_dir = self.index_dir
        else:
            log_dir = Path.cwd()
        _ensure_dir(log_dir)
        log_file = log_dir / f"{self.mode}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"

        logger_name = f"UnifiedProcessor.{self.mode}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(self._log_level)
        self.logger.propagate = False  # do not bubble to root

        if not self.logger.handlers:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(self._log_level)
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(self._log_level)
            fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            fh.setFormatter(fmt)
            ch.setFormatter(fmt)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    # ---- Chunking ----

    def chunk_documents(self, input_dir: str, file_pattern: str = "*.txt") -> None:
        """Process documents in `input_dir` into chunk JSON files under <root>/_chunks/."""
        # Lazy import for type only
        try:
            pass  # type: ignore
        except Exception as e:
            self.logger.error("Required module emailops.text_chunker missing: %s", e)
            raise

        self.input_dir = Path(input_dir).expanduser().resolve()
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        # Force re-chunk option: drop the entire output dir
        if self.force_rechunk and (self.output_dir.exists() or self.chunks_dir.exists()):
            self.logger.warning("Force re-chunk enabled: removing existing directory %s", self.output_dir)
            shutil.rmtree(self.output_dir, ignore_errors=True)

        _ensure_dir(self.chunks_dir)
        _ensure_dir(self.state_dir)

        # Pass pure-data config to workers
        chunk_cfg_data = dict(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            respect_sentences=True,
            respect_paragraphs=True,
            progressive_scaling=True,
        )

        jobs = self._find_documents(file_pattern)
        if not jobs:
            self.logger.info("No documents found matching %r under %s", file_pattern, self.input_dir)
            return

        self.logger.info("Found %d documents to process", len(jobs))

        if self.num_workers > 1:
            self._parallel_chunk(jobs, chunk_cfg_data)
        else:
            self._sequential_chunk(jobs, chunk_cfg_data)

    def _find_documents(self, file_pattern: str) -> List[ChunkJob]:
        """Find documents under `self.input_dir` and build chunk jobs."""
        jobs: List[ChunkJob] = []
        for file_path in self.input_dir.rglob(file_pattern):
            if not file_path.is_file():
                continue
            try:
                st = file_path.stat()
                if st.st_size <= 0:
                    continue
                rel = file_path.relative_to(self.input_dir).as_posix()
                jobs.append(ChunkJob(doc_id=rel, doc_path=file_path, file_size=st.st_size))
            except Exception as e:
                self.logger.warning("Could not stat %s: %s", file_path, e)

        # Process largest files first for better throughput
        jobs.sort(key=lambda j: -j.file_size)
        if self.test_mode:
            jobs = jobs[:10]
        return jobs

    def _parallel_chunk(self, jobs: List[ChunkJob], chunk_cfg_data: Dict[str, Any]) -> None:
        """Process chunks in parallel using top-level worker function."""
        worker_configs = self._distribute_chunking_work(jobs, chunk_cfg_data)

        stats_queue = self.ctx.Queue()
        control_queue = self.ctx.Queue()
        workers: List[mp.Process] = []

        for cfg in worker_configs:
            p = self.ctx.Process(
                target=_chunk_worker_entry,
                args=(cfg, stats_queue, control_queue),
            )
            p.start()
            workers.append(p)

        self._monitor_workers(workers, stats_queue, len(worker_configs), control_queue)

        for p in workers:
            p.join(timeout=5)

    def _sequential_chunk(self, jobs: List[ChunkJob], chunk_cfg_data: Dict[str, Any]) -> None:
        """Process chunks sequentially (single process)."""
        from emailops.text_chunker import TextChunker, ChunkConfig  # type: ignore
        from emailops.utils import read_text_file                     # type: ignore

        chunker = TextChunker(ChunkConfig(**chunk_cfg_data))
        chunks_dir = self.chunks_dir

        for i, job in enumerate(jobs):
            self.logger.info("Processing %d/%d: %s", i + 1, len(jobs), job.doc_id)
            try:
                if self.resume and _should_skip_as_up_to_date(chunks_dir, job):
                    continue

                text = read_text_file(job.doc_path)
                chunks = chunker.chunk_text(text, metadata={"doc_id": job.doc_id, "doc_path": str(job.doc_path)})
                if chunks:
                    _save_chunks_to_dir(chunks_dir, job.doc_id, chunks, job.file_size)
            except Exception as e:
                self.logger.error("Error processing %s: %s", job.doc_path, e)

    def _distribute_chunking_work(self, jobs: List[ChunkJob], chunk_cfg_data: Dict[str, Any]) -> List[WorkerConfig]:
        """Distribute jobs across workers by approximate total bytes."""
        n_workers = min(self.num_workers, max(1, len(jobs)))
        bins: List[List[ChunkJob]] = [[] for _ in range(n_workers)]
        sizes = [0] * n_workers
        for job in jobs:
            idx = sizes.index(min(sizes))
            bins[idx].append(job)
            sizes[idx] += job.file_size

        configs: List[WorkerConfig] = []
        for i, worker_jobs in enumerate(bins):
            if not worker_jobs:
                continue
            configs.append(WorkerConfig(
                worker_id=i,
                jobs_assigned=worker_jobs,
                chunks_dir=str(self.chunks_dir),
                chunk_config=chunk_cfg_data,
                resume=self.resume,
            ))
        return configs

    # ---- Embeddings ----

    def create_embeddings(self, use_chunked_files: bool = True) -> None:
        """
        Create embeddings either from chunk JSON files (<root>/_chunks)
        or directly from documents under `self.root_dir` (raw-doc fallback).
        """
        # Soft dependency: validated accounts (not strictly required to embed via llm_client)
        try:
            from emailops.env_utils import load_validated_accounts  # type: ignore
            accts = load_validated_accounts()
            self.logger.info("Validated accounts loaded: %d", len(accts) if accts else 0)
        except Exception:
            self.logger.debug("env_utils.load_validated_accounts not available; proceeding without it.")

        if use_chunked_files:
            if self.num_workers > 1:
                self._parallel_embed_from_chunks()
                self.logger.info("Parallel embed finished. Run the 'repair' command to merge batches into the final index.")
            else:
                self._embed_from_chunks()
        else:
            if self.num_workers > 1:
                self.logger.warning("Parallel embedding from raw documents not implemented; using single process.")
            self._embed_from_documents()

    def _load_all_chunk_objects(self) -> List[Dict[str, Any]]:
        """Load all chunk JSON files previously written by chunking step."""
        chunks_dir = self.root_dir / "_chunks" / "chunks"
        if not chunks_dir.exists():
            self.logger.error("No chunks directory found at %s", chunks_dir)
            return []
        chunk_files = sorted(chunks_dir.glob("*.json"))  # sorted for more stable ordering
        if not chunk_files:
            self.logger.error("No chunk files found in %s", chunks_dir)
            return []

        self.logger.info("Found %d chunk files", len(chunk_files))
        all_chunks: List[Dict[str, Any]] = []
        for cf in chunk_files:
            try:
                with cf.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                for ch in data.get("chunks", []) or []:
                    ch["source_file"] = str(cf)
                    all_chunks.append(ch)
            except Exception as e:
                self.logger.error("Failed to load %s: %s", cf, e)
        return all_chunks

    def _embed_from_chunks(self) -> None:
        """Create embeddings from chunk JSON files (single process, streaming to memmap)."""
        from emailops.llm_client import embed_texts  # type: ignore
        from emailops.index_metadata import create_index_metadata, save_index_metadata  # type: ignore

        chunks = self._load_all_chunk_objects()
        if not chunks:
            return

        texts = [str(c.get("text", "")) for c in chunks]
        self.logger.info("Embedding %d chunks (single process, batch=%d)...", len(texts), self.batch_size)

        provider = os.getenv("EMBED_PROVIDER", "vertex")
        emb_dim = _probe_embedding_dim(provider)

        # Prepare memmap for embeddings.npy
        N = len(texts)
        emb_path = self.index_dir / "embeddings.npy"
        mm = np.memmap(emb_path, dtype="float32", mode="w+", shape=(N, emb_dim))

        all_ok = True
        batch = min(self.batch_size, 250)

        # Streamed embedding into memmap
        offset = 0
        for i in range(0, N, batch):
            batch_texts = texts[i:i + batch]
            try:
                arr = np.asarray(embed_texts(batch_texts, provider=provider), dtype="float32")
                if arr.ndim != 2 or int(arr.shape[1]) != emb_dim:
                    raise RuntimeError(f"Embedding provider returned shape {arr.shape}, expected (*, {emb_dim})")
            except Exception as e:
                self.logger.error("Embedding failed for batch starting at %d: %s", i, e)
                arr = np.zeros((len(batch_texts), emb_dim), dtype="float32")
                all_ok = False
            mm[offset:offset + len(batch_texts), :] = arr
            offset += len(batch_texts)

        mm.flush()

        # Build mapping compatible with downstream modules
        mapping = _chunks_to_mapping(chunks, project_root=self.root_dir)

        if offset != len(mapping):
            self.logger.error("Embeddings/document count mismatch: %d vs %d (truncating to min)", offset, len(mapping))
            n = min(offset, len(mapping))
            # Truncate memmap by rewriting a new file with n rows
            tmp_path = emb_path.with_suffix(".tmp.npy")
            tmp = np.memmap(tmp_path, dtype="float32", mode="w+", shape=(n, emb_dim))
            tmp[:] = mm[:n]
            tmp.flush()
            # Close memmaps before replacing on Windows
            try:
                del mm
                del tmp
                gc.collect()
            except Exception:
                pass
            os.replace(tmp_path, emb_path)
            mapping = mapping[:n]
        else:
            # Ensure mm is closed before FAISS attempts to re-open file on Windows
            try:
                del mm
                gc.collect()
            except Exception:
                pass

        (self.index_dir / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("Saved embeddings.npy and mapping.json to %s", self.index_dir)

        # Optional FAISS (cosine-like IP via normalization) with zero-vector guard
        try:
            import faiss  # type: ignore
            embs = np.memmap(emb_path, dtype="float32", mode="r", shape=(len(mapping), emb_dim))
            embs_for_index = np.array(embs, dtype="float32", copy=True)
            row_norms = np.einsum("ij,ij->i", embs_for_index, embs_for_index)
            zero_ratio = float(np.count_nonzero(row_norms == 0.0)) / max(1, len(row_norms))
            if zero_ratio > 0.05:
                self.logger.warning("Skipping FAISS build: %.1f%% zero vectors present. Run 'fix' to repair.", zero_ratio * 100)
            else:
                faiss.normalize_L2(embs_for_index)
                index = faiss.IndexFlatIP(int(embs_for_index.shape[1]))
                index.add(np.ascontiguousarray(embs_for_index))
                faiss.write_index(index, str(self.index_dir / "index.faiss"))
                self.logger.info("Wrote FAISS index to %s", self.index_dir / "index.faiss")
        except Exception as e:
            self.logger.warning("FAISS indexing failed (continuing without FAISS): %s", e)

        # Write metadata for compatibility with index_validation/doctor
        try:
            pass  # type: ignore
        except Exception:
            pass
        try:
            meta = create_index_metadata(
                provider=provider,
                num_documents=len(mapping),
                num_folders=len({m.get("conv_id") for m in mapping}),
                index_dir=self.index_dir,
                custom_metadata={"actual_dimensions": int(emb_dim), "all_batches_ok": all_ok}
            )
            save_index_metadata(meta, self.index_dir)
            self.logger.info("Wrote meta.json with provider=%s", meta.get("provider"))
        except Exception as e:
            self.logger.warning("Failed to write index metadata: %s", e)

    def _embed_from_documents(self) -> None:
        """
        Fallback: create embeddings directly from .txt documents under root_dir,
        excluding _chunks/ and _index/ directories. Treat each file as a single chunk.
        """
        from emailops.llm_client import embed_texts  # type: ignore
        from emailops.index_metadata import create_index_metadata, save_index_metadata  # type: ignore

        txt_files: List[Path] = []
        for p in self.root_dir.rglob("*.txt"):
            try:
                if "_chunks" in p.parts or "_index" in p.parts:
                    continue
                if p.is_file() and p.stat().st_size > 0:
                    txt_files.append(p)
            except Exception:
                continue

        if not txt_files:
            self.logger.info("No raw .txt documents found under %s", self.root_dir)
            return

        self.logger.info("Embedding %d raw documents (single process)...", len(txt_files))

        # Build synthetic "chunks" with chunk_index=0
        chunks: List[Dict[str, Any]] = []
        for fp in txt_files:
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
            rel = fp.relative_to(self.root_dir).as_posix()
            chunks.append({
                "text": text,
                "chunk_index": 0,
                "metadata": {
                    "doc_id": rel,
                    "doc_path": str(fp),
                }
            })

        provider = os.getenv("EMBED_PROVIDER", "vertex")
        emb_dim = _probe_embedding_dim(provider)
        N = len(chunks)
        emb_path = self.index_dir / "embeddings.npy"
        mm = np.memmap(emb_path, dtype="float32", mode="w+", shape=(N, emb_dim))

        # Stream
        batch = min(self.batch_size, 250)
        offset = 0
        for i in range(0, N, batch):
            batch_texts = [str(chunks[k]["text"]) for k in range(i, min(i + batch, N))]
            try:
                arr = np.asarray(embed_texts(batch_texts, provider=provider), dtype="float32")
                if arr.ndim != 2 or int(arr.shape[1]) != emb_dim:
                    raise RuntimeError(f"Embedding provider returned shape {arr.shape}, expected (*, {emb_dim})")
            except Exception as e:
                self.logger.error("Embedding failed for raw-doc batch starting at %d: %s", i, e)
                arr = np.zeros((len(batch_texts), emb_dim), dtype="float32")
            mm[offset:offset + len(batch_texts), :] = arr
            offset += len(batch_texts)
        mm.flush()

        mapping = _chunks_to_mapping(chunks, project_root=self.root_dir)

        # Close memmap before FAISS handling on Windows
        try:
            del mm
            gc.collect()
        except Exception:
            pass

        (self.index_dir / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

        try:
            import faiss  # type: ignore
            embs = np.memmap(emb_path, dtype="float32", mode="r", shape=(len(mapping), emb_dim))
            embs_for_index = np.array(embs, dtype="float32", copy=True)
            row_norms = np.einsum("ij,ij->i", embs_for_index, embs_for_index)
            zero_ratio = float(np.count_nonzero(row_norms == 0.0)) / max(1, len(row_norms))
            if zero_ratio > 0.05:
                self.logger.warning("Skipping FAISS build: %.1f%% zero vectors present. Run 'fix' to repair.", zero_ratio * 100)
            else:
                faiss.normalize_L2(embs_for_index)
                index = faiss.IndexFlatIP(int(embs_for_index.shape[1]))
                index.add(np.ascontiguousarray(embs_for_index))
                faiss.write_index(index, str(self.index_dir / "index.faiss"))
                self.logger.info("Wrote FAISS index to %s", self.index_dir / "index.faiss")
        except Exception as e:
            self.logger.warning("FAISS indexing failed (continuing without FAISS): %s", e)

        try:
            meta = create_index_metadata(
                provider=provider,
                num_documents=len(mapping),
                num_folders=len({m.get("conv_id") for m in mapping}),
                index_dir=self.index_dir,
                custom_metadata={"actual_dimensions": int(emb_dim)}
            )
            save_index_metadata(meta, self.index_dir)
            self.logger.info("Wrote meta.json with provider=%s", meta.get("provider"))
        except Exception as e:
            self.logger.warning("Failed to write index metadata: %s", e)

    def _parallel_embed_from_chunks(self) -> None:
        """Parallel embedding using worker pickles + a merge step (`repair_index`)."""
        chunks = self._load_all_chunk_objects()
        if not chunks:
            return

        total = len(chunks)
        n_workers = min(self.num_workers, max(1, total))
        parts: List[List[int]] = [[] for _ in range(n_workers)]
        for i in range(total):
            parts[i % n_workers].append(i)

        stats_queue = self.ctx.Queue()
        workers: List[mp.Process] = []
        for wid, idxs in enumerate(parts):
            if not idxs:
                continue
            p = self.ctx.Process(
                target=_embedding_worker_entry,
                args=(str(self.index_dir), wid, idxs, chunks, stats_queue, self.batch_size),
            )
            p.start()
            workers.append(p)

        self._monitor_workers(workers, stats_queue, len(workers))
        for w in workers:
            w.join(timeout=5)

        self.logger.info("Parallel embedding complete. Use 'repair' to merge batches into final artifacts.")

    def repair_index(self, remove_batches: bool = False) -> None:
        """Merge worker batch pickles into final index (embeddings.npy + mapping.json + meta.json), streaming via memmap."""
        from emailops.index_metadata import create_index_metadata, save_index_metadata  # type: ignore

        emb_dir = self.index_dir / "embeddings"
        if not emb_dir.exists():
            self.logger.error("No batch directory found at: %s", emb_dir)
            return

        pkl_files = sorted(emb_dir.glob("worker_*_batch_*.pkl"))
        if not pkl_files:
            self.logger.error("No batch pickle files found in %s", emb_dir)
            return

        self.logger.info("Scanning %d batch files...", len(pkl_files))

        # First pass: validate shapes, compute total rows and common width
        total_rows = 0
        emb_dim: Optional[int] = None
        valid_files: List[Path] = []

        for pkl in pkl_files:
            try:
                with pkl.open("rb") as f:
                    data = pickle.load(f)
                embs = np.asarray(data.get("embeddings", []), dtype="float32")
                chunks = list(data.get("chunks", []) or [])
                if embs.ndim != 2 or embs.shape[0] != len(chunks):
                    self.logger.warning("Skipping malformed batch %s (chunks=%d, embs=%s)", pkl.name, len(chunks), embs.shape)
                    continue
                if emb_dim is None:
                    emb_dim = int(embs.shape[1])
                elif int(embs.shape[1]) != emb_dim:
                    self.logger.warning("Skipping batch %s due to width mismatch (%d vs %d)", pkl.name, int(embs.shape[1]), emb_dim)
                    continue
                total_rows += embs.shape[0]
                valid_files.append(pkl)
            except Exception as e:
                self.logger.error("Failed to load %s: %s", pkl, e)

        if emb_dim is None or total_rows == 0 or not valid_files:
            self.logger.error("No valid embeddings found to merge.")
            return

        self.logger.info("Merging %d valid batches, total rows=%d, dim=%d ...", len(valid_files), total_rows, emb_dim)

        # Second pass: write memmap, accumulate chunks for mapping
        emb_path = self.index_dir / "embeddings.npy"
        mm = np.memmap(emb_path, dtype="float32", mode="w+", shape=(total_rows, emb_dim))
        all_chunks: List[Dict[str, Any]] = []
        offset = 0

        for pkl in valid_files:
            try:
                with pkl.open("rb") as f:
                    data = pickle.load(f)
                embs = np.asarray(data.get("embeddings", []), dtype="float32")
                chunks = list(data.get("chunks", []) or [])
                n = embs.shape[0]
                mm[offset:offset + n, :] = embs
                offset += n
                all_chunks.extend(chunks)
            except Exception as e:
                self.logger.error("Failed during merge of %s: %s", pkl, e)

        mm.flush()

        # Build mapping compatible with the rest of the codebase
        mapping = _chunks_to_mapping(all_chunks, project_root=self.root_dir)
        if offset != len(mapping):
            self.logger.warning("Embeddings/document mismatch after merge: %d vs %d (truncating).", offset, len(mapping))
            n = min(offset, len(mapping))
            tmp_path = emb_path.with_suffix(".tmp.npy")
            tmp = np.memmap(tmp_path, dtype="float32", mode="w+", shape=(n, emb_dim))
            tmp[:] = mm[:n]
            tmp.flush()
            # Close memmaps before replacing on Windows
            try:
                del mm
                del tmp
                gc.collect()
            except Exception:
                pass
            os.replace(tmp_path, emb_path)
            mapping = mapping[:n]
        else:
            try:
                del mm
                gc.collect()
            except Exception:
                pass

        # Save mapping
        (self.index_dir / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("Saved embeddings.npy and mapping.json to %s", self.index_dir)

        # Best-effort FAISS (cosine-like IP) with zero-vector guard
        try:
            import faiss  # type: ignore
            embs = np.memmap(emb_path, dtype="float32", mode="r", shape=(len(mapping), emb_dim))
            embs_for_index = np.array(embs, dtype="float32", copy=True)
            row_norms = np.einsum("ij,ij->i", embs_for_index, embs_for_index)
            zero_ratio = float(np.count_nonzero(row_norms == 0.0)) / max(1, len(row_norms))
            if zero_ratio > 0.05:
                self.logger.warning("Skipping FAISS build: %.1f%% zero vectors present. Run 'fix' to repair.", zero_ratio * 100)
            else:
                faiss.normalize_L2(embs_for_index)
                index = faiss.IndexFlatIP(int(embs_for_index.shape[1]))
                index.add(np.ascontiguousarray(embs_for_index))
                faiss.write_index(index, str(self.index_dir / "index.faiss"))
                self.logger.info("Wrote FAISS index")
        except Exception as e:
            self.logger.warning("Could not create FAISS index: %s", e)

        # Write metadata for compatibility with doctor/search
        try:
            meta = create_index_metadata(
                provider=os.getenv("EMBED_PROVIDER", "vertex"),
                num_documents=len(mapping),
                num_folders=len({m.get("conv_id") for m in mapping}),
                index_dir=self.index_dir,
                custom_metadata={"actual_dimensions": int(emb_dim)},
            )
            save_index_metadata(meta, self.index_dir)
        except Exception as e:
            self.logger.warning("Failed to write index metadata: %s", e)

        # Optional cleanup
        if remove_batches:
            removed = 0
            for pkl in valid_files:
                try:
                    pkl.unlink()
                    removed += 1
                except Exception:
                    pass
            self.logger.info("Removed %d batch pickle files.", removed)

    def fix_failed_embeddings(self) -> None:
        """
        Re-embed zero vectors inside worker batch pickles (best-effort),
        and also repair zero rows in embeddings.npy using mapping.json
        and original chunk JSON files if no batch pickles are present.
        """
        from emailops.llm_client import embed_texts  # type: ignore

        emb_dir = self.index_dir / "embeddings"
        provider = os.getenv("EMBED_PROVIDER", "vertex")

        # Probe dim (best-effort; continue if it fails)
        emb_dim_probe: Optional[int] = None
        try:
            emb_dim_probe = _probe_embedding_dim(provider)
        except Exception as e:
            self.logger.warning("Embedding dim probe failed during fix; will try to infer from file size: %s", e)

        # First: try to fix batch pickles if present
        if emb_dir.exists():
            pkl_files = list(emb_dir.glob("*.pkl"))
            total_fixed = 0
            if pkl_files:
                for pkl in pkl_files:
                    try:
                        with pkl.open("rb") as f:
                            data = pickle.load(f)
                        embs = np.asarray(data.get("embeddings", []), dtype="float32")
                        chunks = list(data.get("chunks", []) or [])
                        if embs.ndim != 2 or embs.shape[0] != len(chunks):
                            self.logger.warning("Skipping malformed batch %s", pkl.name)
                            continue
                        zero_mask = np.all(embs == 0, axis=1)
                        num_zeros = int(zero_mask.sum())
                        if num_zeros == 0:
                            continue
                        self.logger.info("Found %d zero vectors in %s; re-embedding...", num_zeros, pkl.name)
                        texts = [str(chunks[i].get("text", "")) for i in np.where(zero_mask)[0].tolist()]
                        try:
                            new = np.asarray(embed_texts(texts, provider=provider), dtype="float32")
                            if new.shape[0] != num_zeros or new.ndim != 2 or new.shape[1] != embs.shape[1]:
                                raise RuntimeError("Provider returned wrong shape")
                            embs[zero_mask] = new
                            with pkl.open("wb") as f:
                                pickle.dump({"chunks": chunks, "embeddings": embs}, f)
                            total_fixed += num_zeros
                        except Exception as e:
                            self.logger.error("Re-embed failed for %s: %s", pkl.name, e)
                    except Exception as e:
                        self.logger.error("Error processing %s: %s", pkl.name, e)
                self.logger.info("Fixed %d zero vectors in batch pickles.", total_fixed)
                # If we fixed batch pickles, caller will still need to run `repair`
                if total_fixed > 0:
                    return

        # Second: fix embeddings.npy directly if present (single-process embed path)
        emb_path = self.index_dir / "embeddings.npy"
        map_path = self.index_dir / "mapping.json"
        chunks_dir = self.root_dir / "_chunks" / "chunks"

        if not emb_path.exists() or not map_path.exists() or not chunks_dir.exists():
            self.logger.info("No direct embeddings.npy/mapping.json/chunks to repair; nothing to do.")
            return

        try:
            mapping = json.loads(map_path.read_text(encoding="utf-8"))
        except Exception as e:
            self.logger.error("Could not read mapping.json: %s", e)
            return

        rows = len(mapping)
        emb_dim: Optional[int] = None
        # Try probe-based shape first
        if emb_dim_probe:
            try:
                embs = np.memmap(emb_path, dtype="float32", mode="r+", shape=(rows, emb_dim_probe))
                emb_dim = emb_dim_probe
            except Exception:
                embs = None  # type: ignore
        else:
            embs = None  # type: ignore

        # Fallback to inferring from file size
        if embs is None:
            file_bytes = os.path.getsize(emb_path)
            if (rows * 4) == 0 or file_bytes % (rows * 4) != 0:
                self.logger.error("Cannot infer embedding dimension from file size (rows=%d, bytes=%d).", rows, file_bytes)
                return
            emb_dim = file_bytes // (rows * 4)
            try:
                embs = np.memmap(emb_path, dtype="float32", mode="r+", shape=(rows, emb_dim))
            except Exception as e:
                self.logger.error("Failed to open embeddings memmap: %s", e)
                return

        # Repair zero rows by reconstructing text via chunk JSONs
        zero_mask = np.all(embs == 0, axis=1)
        num_zeros = int(zero_mask.sum())
        if num_zeros == 0:
            self.logger.info("No zero vectors detected in embeddings.npy.")
            try:
                del embs
                gc.collect()
            except Exception:
                pass
            return

        # Build rec_id -> text by scanning chunk JSON files (with per-doc fallback indices)
        rec_to_text: Dict[str, str] = {}
        for cf in sorted(chunks_dir.glob("*.json")):
            try:
                data = json.loads(cf.read_text(encoding="utf-8"))
                doc_id = data.get("doc_id") or ""
                counter = 0
                for ch in data.get("chunks", []) or []:
                    idx = ch.get("chunk_index")
                    if idx is None:
                        idx = counter
                        counter += 1
                    rec_id = f"{doc_id}::chunk{int(idx)}"
                    rec_to_text[rec_id] = str(ch.get("text", ""))
            except Exception:
                continue

        # Prepare texts for zero rows in mapping order
        zero_indices = np.where(zero_mask)[0].tolist()
        texts: List[str] = []
        missing = 0
        for i in zero_indices:
            rec_id = str(mapping[i].get("id", ""))
            text = rec_to_text.get(rec_id)
            if text is None:
                text = ""
                missing += 1
            texts.append(text)

        if missing:
            self.logger.warning("Missing text for %d records while repairing zeros; those will remain zeros.", missing)

        # Re-embed in batches
        batch = min(self.batch_size, 250)
        pos = 0
        for i in range(0, len(texts), batch):
            seg = texts[i:i + batch]
            try:
                new = np.asarray(embed_texts(seg, provider=provider), dtype="float32")
                if new.ndim != 2 or new.shape[0] != len(seg) or new.shape[1] != embs.shape[1]:
                    raise RuntimeError("Provider returned wrong shape")
            except Exception as e:
                self.logger.error("Re-embed failed for a segment: %s", e)
                new = np.zeros((len(seg), embs.shape[1]), dtype="float32")
            idxs = zero_indices[pos:pos + len(seg)]
            embs[idxs, :] = new
            pos += len(seg)

        embs.flush()
        try:
            del embs
            gc.collect()
        except Exception:
            pass
        self.logger.info("Fixed %d zero vectors in embeddings.npy.", num_zeros)

    # ---- Monitoring & UI ----

    def _monitor_workers(self, workers: List[mp.Process], stats_queue, expected_count: int, control_queue=None) -> None:
        """Monitor worker progress and render a simple TTY UI, robust to crashes/hangs."""
        worker_states: Dict[int, Any] = {}
        dead: set[int] = set()
        last_recv = time.time()
        # Configurable idle timeout
        try:
            IDLE_TIMEOUT = float(os.getenv("EMAILOPS_MONITOR_IDLE_TIMEOUT", "300"))
        except Exception:
            IDLE_TIMEOUT = 300.0

        while True:
            got = False
            while True:
                try:
                    s = stats_queue.get_nowait()
                    worker_states[s.worker_id] = s
                    got = True
                    last_recv = time.time()
                except queue.Empty:
                    break
            if got:
                self._display_progress(worker_states)

            # mark dead workers as failed if they never reported completion
            for wid, p in enumerate(workers):
                if not p.is_alive() and wid not in dead:
                    dead.add(wid)
                    st = worker_states.get(wid)
                    if st:
                        st.status = "failed" if getattr(st, "status", "") not in ("completed", "stopped") else st.status
                    else:
                        class S:
                            pass
                        st = S()
                        st.worker_id = wid
                        st.status = "failed"
                        st.errors = 1
                        st.progress_percent = 0.0
                        worker_states[wid] = st

            if len(worker_states) == expected_count:
                if all(getattr(s, "status", "") in ("completed", "failed", "stopped") for s in worker_states.values()):
                    break

            if time.time() - last_recv > IDLE_TIMEOUT:
                self.logger.warning("No worker updates for %.0fs; terminating monitor.", IDLE_TIMEOUT)
                # Politely ask chunk workers to stop if we have a control queue
                if control_queue is not None:
                    try:
                        control_queue.put("SHUTDOWN")
                    except Exception:
                        pass
                break

            time.sleep(0.5)

        self._print_summary(worker_states)

    def _display_progress(self, worker_states: Dict[int, Any]) -> None:
        if sys.stdout.isatty():
            os.system("cls" if os.name == "nt" else "clear")
        print("=" * 80)
        print(f"PROCESSING - {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S %Z}")
        print("=" * 80)
        for wid in sorted(worker_states.keys()):
            s = worker_states[wid]
            pct = getattr(s, "progress_percent", 0.0)
            status = getattr(s, "status", "")
            print(f"\nWorker {wid}: [{pct:5.1f}%] {status}")
            if hasattr(s, "docs_processed"):
                print(f"  Docs: {s.docs_processed}/{s.docs_total}")
                print(f"  Chunks: {s.chunks_created}")
            elif hasattr(s, "chunks_processed"):
                print(f"  Chunks: {s.chunks_processed}/{s.chunks_total}")
            errs = getattr(s, "errors", 0)
            if errs:
                print(f"  Errors: {errs}")

    def _print_summary(self, worker_states: Dict[int, Any]) -> None:
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETED")
        print("=" * 80)
        total_processed = 0
        total_errors = 0
        for s in worker_states.values():
            if hasattr(s, "docs_processed"):
                total_processed += getattr(s, "docs_processed", 0)
            elif hasattr(s, "chunks_processed"):
                total_processed += getattr(s, "chunks_processed", 0)
            total_errors += getattr(s, "errors", 0)
        print(f"Total processed: {total_processed}")
        if total_errors:
            print(f"Total errors: {total_errors}")

# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description="Unified processor for text and embedding operations")
    sub = parser.add_subparsers(dest="command", help="Commands")

    # chunk
    p_chunk = sub.add_parser("chunk", help="Process documents into chunks")
    p_chunk.add_argument("--input", required=True, help="Input directory")
    p_chunk.add_argument("--output", required=True, help="Project root directory (will contain _chunks)")
    p_chunk.add_argument("--workers", type=int, help="Number of workers")
    p_chunk.add_argument("--chunk-size", type=int, default=1600)
    p_chunk.add_argument("--chunk-overlap", type=int, default=200)
    p_chunk.add_argument("--pattern", default="*.txt", help="File pattern")
    p_chunk.add_argument("--test", action="store_true", help="Test mode (process up to 10 files)")
    p_chunk.add_argument("--force", action="store_true", help="Force re-chunk by deleting existing chunks")
    p_chunk.add_argument("--no-resume", action="store_true", help="Disable resume/up-to-date skip")

    # embed
    p_embed = sub.add_parser("embed", help="Create embeddings (defaults to using chunk files)")
    p_embed.add_argument("--root", required=True, help="Project root directory (same as --output used in chunk)")
    p_embed.add_argument("--batch-size", type=int, default=64)
    p_embed.add_argument("--workers", type=int, help="Number of embedding workers (parallel)")
    p_embed.add_argument(
        "--raw",
        action="store_true",
        help="Embed directly from raw .txt docs under --root (not from chunk files)"
    )

    # repair
    p_repair = sub.add_parser("repair", help="Merge batch pickles into final index")
    p_repair.add_argument("--root", required=True, help="Project root directory")
    p_repair.add_argument("--remove-batches", action="store_true")

    # fix
    p_fix = sub.add_parser("fix", help="Fix failed embeddings in batches or embeddings.npy")
    p_fix.add_argument("--root", required=True, help="Project root directory")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "chunk":
        proc = UnifiedProcessor(
            root_dir=args.output,
            mode="chunk",
            num_workers=args.workers,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            test_mode=args.test,
            force_rechunk=args.force,
            resume=(not args.no_resume),
        )
        proc.chunk_documents(args.input, args.pattern)
        return 0

    if args.command == "embed":
        proc = UnifiedProcessor(
            root_dir=args.root,
            mode="embed",
            num_workers=args.workers,
            batch_size=args.batch_size,
        )
        # Default: use chunked files; --raw opt-in flips to raw-doc path
        proc.create_embeddings(use_chunked_files=(not args.raw))
        return 0

    if args.command == "repair":
        proc = UnifiedProcessor(root_dir=args.root, mode="repair")
        proc.repair_index(remove_batches=args.remove_batches)
        return 0

    if args.command == "fix":
        proc = UnifiedProcessor(root_dir=args.root, mode="fix")
        proc.fix_failed_embeddings()
        return 0

    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
