#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Vertex AI Email Indexer with Multi-Account Support (Production Ready)

- Parallel + sequential execution
- Multi-account / multi-worker embedding for Vertex via llm_client
- Robust chunked-file flow (_chunks/chunks/*.json) and legacy "folder list" flow
- Deduplicated, defensive, and CLI-friendly

Artifacts produced (via finalization):
  _index/
    - embeddings.npy
    - index.faiss            (when faiss available)
    - mapping.json
    - meta.json
    - embeddings/worker_*.pkl  (intermediate, merged by finalizer)
    - embedded_chunks.json     (only in chunked-files mode)
    - last_embedding_run.txt   (only in chunked-files mode)
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
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

# -----------------------
# Imports: prefer relative, fallback to absolute for script runs
# -----------------------
try:
    from .env_utils import get_worker_configs, LLMError  # type: ignore
except Exception:  # pragma: no cover
    from emailops.env_utils import get_worker_configs, LLMError  # type: ignore

# Worker-time imports use this same pattern to support script execution.

# -----------------------
# Constants
# -----------------------
INDEX_DIRNAME = os.getenv("INDEX_DIRNAME", "_index")

# -----------------------
# Data model
# -----------------------
@dataclass
class ProcessingStats:
    """Statistics for monitoring progress"""
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
    def estimated_time_remaining(self) -> Optional[timedelta]:
        if self.chunks_processed == 0:
            return None
        elapsed = time.time() - self.start_time
        rate = self.chunks_processed / max(elapsed, 1e-6)
        remaining = self.chunks_total - self.chunks_processed
        if rate > 0:
            return timedelta(seconds=remaining / rate)
        return None


# -----------------------
# Indexer
# -----------------------
class VertexIndexer:
    """Main coordinator for Vertex AI indexing with multi-account support"""

    def __init__(
        self,
        export_dir: str,
        mode: str = "parallel",
        resume: bool = True,
        batch_size: int = 8,
        test_mode: bool = False,
        test_chunks: int = 100,
        incremental: bool = False,
        force_rebuild: bool = False,
        use_chunked_files: bool = False,
    ) -> None:
        self.export_dir = Path(export_dir).expanduser().resolve()
        self.mode = mode
        self.resume = resume
        self.batch_size = int(batch_size)
        self.test_mode = bool(test_mode)
        self.test_chunks = int(test_chunks)
        self.incremental = bool(incremental)
        self.force_rebuild = bool(force_rebuild)
        self.use_chunked_files = bool(use_chunked_files)

        # Load validated accounts and configure worker count
        self.accounts = self._load_worker_accounts()
        self.num_workers = len(self.accounts)
        if self.num_workers == 0:
            raise LLMError("No validated accounts found. Please run: python validate_accounts.py")

        # Paths
        self.index_dir = self.export_dir / INDEX_DIRNAME
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_file = self.index_dir / "chunks.pkl"
        self.chunks_dir = self.export_dir / "_chunks" / "chunks" if self.use_chunked_files else None
        self.embedded_chunks_file = self.index_dir / "embedded_chunks.json" if self.use_chunked_files else None
        self.state_file = self.index_dir / "indexer_state.json"

        # Logging
        self.logger = logging.getLogger("VertexIndexer")
        self.setup_logging()
        self._confirm_worker_configuration()

        # IPC (create always so sequential paths can reuse)
        self.manager = mp.Manager()
        self.stats_queue = self.manager.Queue()
        self.control_queue = self.manager.Queue()
        self.workers: List[mp.Process] = []

    # ----------------------- account configuration -----------------------

    def _load_worker_accounts(self) -> List[Dict[str, Any]]:
        """Load validated accounts from env_utils and confirm configuration"""
        validated_accounts = get_worker_configs()
        if not validated_accounts:
            raise LLMError(
                "No valid accounts found. Please run validation:\n"
                "  python validate_accounts.py"
            )
        # Convert to dict format for compatibility
        accounts = [acc.to_dict() for acc in validated_accounts]
        return accounts

    def _confirm_worker_configuration(self) -> None:
        """Explicitly confirm worker count matches validated accounts"""
        if self.num_workers != len(self.accounts):
            raise RuntimeError(
                f"Worker count mismatch! num_workers={self.num_workers}, accounts={len(self.accounts)}"
            )

        account_groups: Dict[int, List[str]] = {}
        for acc in self.accounts:
            group = int(acc.get("account_group", 0))
            account_groups.setdefault(group, []).append(acc["project_id"])

        self.logger.info("=" * 80)
        self.logger.info("WORKER CONFIGURATION - AUTO-DETECTED FROM VALIDATED ACCOUNTS")
        self.logger.info("✓ Loaded %d validated accounts", len(self.accounts))
        self.logger.info("✓ Embedder workers automatically set to: %d", self.num_workers)
        self.logger.info("Account groups: %s", {k: len(v) for k, v in account_groups.items()})
        self.logger.info("=" * 80)

    # ----------------------- chunked-file incremental helpers -----------------------

    def _load_embedded_chunks_index(self) -> Dict[str, float]:
        """Load index of which chunk files have been embedded and when."""
        if self.embedded_chunks_file and self.embedded_chunks_file.exists():
            try:
                with open(self.embedded_chunks_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                self.logger.warning("Could not load embedded chunks index: %s", e)
        return {}

    def _save_embedded_chunks_index(self, embedded_index: Dict[str, float]) -> None:
        """Save index of embedded chunk files."""
        if self.embedded_chunks_file:
            with open(self.embedded_chunks_file, "w", encoding="utf-8") as f:
                json.dump(embedded_index, f, indent=2)

    def find_new_chunk_files(self) -> List[Path]:
        """Find chunk files that are new or modified since last embedding run."""
        if not self.chunks_dir or not self.chunks_dir.exists():
            return []

        embedded_index = self._load_embedded_chunks_index()
        last_embedding_time = self._get_last_embedding_time()

        new_chunk_files: List[Path] = []
        for chunk_file in self.chunks_dir.glob("*.json"):
            try:
                chunk_mtime = chunk_file.stat().st_mtime
                chunk_id = chunk_file.stem
                if (
                    chunk_id not in embedded_index
                    or chunk_mtime > float(embedded_index.get(chunk_id, 0))
                    or (last_embedding_time and chunk_mtime > last_embedding_time)
                ):
                    new_chunk_files.append(chunk_file)
            except Exception as e:
                self.logger.warning("Could not check chunk file %s: %s", chunk_file, e)

        self.logger.info("Found %d new or modified chunk files", len(new_chunk_files))
        return new_chunk_files

    def _get_last_embedding_time(self) -> Optional[float]:
        """Get timestamp of last embedding run."""
        timestamp_file = self.index_dir / "last_embedding_run.txt"
        if timestamp_file.exists():
            try:
                return float(timestamp_file.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                return None
        return None

    def _save_last_embedding_time(self) -> None:
        """Save timestamp of current embedding run."""
        timestamp_file = self.index_dir / "last_embedding_run.txt"
        timestamp_file.write_text(str(time.time()), encoding="utf-8")

    # ----------------------- logging -----------------------

    def setup_logging(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.index_dir / f"indexer_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
            force=True,
        )
        self.logger = logging.getLogger("VertexIndexer")

    # ----------------------- legacy "folder list" chunk loading -----------------------

    def load_chunks(self) -> List[Dict[str, Any]]:
        """Legacy: create or load simple per-folder chunk descriptors."""
        if not self.chunks_file.exists():
            self.logger.warning("Chunks file not found: %s", self.chunks_file)
            self.logger.info("Creating chunks from email folders...")
            return self._create_chunks()

        with open(self.chunks_file, "rb") as f:
            chunks = pickle.load(f)

        if self.test_mode:
            chunks = chunks[: self.test_chunks]
        self.logger.info("Loaded %d chunks from %s", len(chunks), self.chunks_file)
        return chunks

    def _create_chunks(self) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for folder in self.export_dir.iterdir():
            if folder.is_dir() and not folder.name.startswith("_"):
                chunks.append({"folder": folder.name, "path": str(folder), "text": ""})
        with open(self.chunks_file, "wb") as f:
            pickle.dump(chunks, f)
        return chunks

    # ----------------------- state -----------------------

    def load_state(self) -> Optional[Dict[str, Any]]:
        if self.state_file.exists() and self.resume:
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning("Could not load state: %s", e)
        return None

    def save_state(self, worker_states: Dict[int, ProcessingStats], expected_workers: int) -> None:
        state = {
            "timestamp": datetime.now().isoformat(),
            "mode": self.mode,
            "num_workers": expected_workers,
            "workers": {
                str(wid): {
                    "project_id": stats.project_id,
                    "chunks_processed": stats.chunks_processed,
                    "chunks_total": stats.chunks_total,
                    "status": stats.status,
                    "errors": stats.errors,
                }
                for wid, stats in worker_states.items()
            },
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    # ----------------------- orchestration -----------------------

    def run_parallel(self) -> None:
        """Parallel indexing orchestration (legacy non-chunked flow)."""
        if self.use_chunked_files:
            self.run_parallel_chunked_files()
            return

        if self.incremental:
            self.logger.info("Incremental mode not supported in parallel legacy flow; falling back to sequential.")
            self.run_sequential()
            return

        self.logger.info(
            "Starting parallel indexing with %d workers (from %d validated accounts)",
            self.num_workers,
            len(self.accounts),
        )

        try:
            all_chunks = self.load_chunks()
            worker_configs = self.distribute_workload(len(all_chunks))
            previous_state = self.load_state()

            for config in worker_configs:
                if previous_state and previous_state.get("num_workers") == self.num_workers:
                    worker_state = previous_state.get("workers", {}).get(str(config["worker_id"]))
                    if worker_state and worker_state.get("status") != "completed":
                        config["chunk_start"] += int(worker_state.get("chunks_processed", 0))
                worker = mp.Process(
                    target=index_worker,
                    args=(
                        config,
                        all_chunks[config["chunk_start"] : config["chunk_end"]],
                        self.export_dir,
                        self.stats_queue,
                        self.control_queue,
                        self.batch_size,
                    ),
                    daemon=True,
                )
                worker.start()
                self.workers.append(worker)

            worker_states = self.monitor_workers(expected_workers=len(worker_configs))
            self.save_state(worker_states, expected_workers=len(worker_configs))

            # Finalize once after workers exit
            finalize_parallel_index(self.export_dir)

        except KeyboardInterrupt:
            self.logger.info("Shutdown signal received.")
        except Exception as e:
            self.logger.error("Coordinator error: %s", e, exc_info=True)

    def run_parallel_chunked_files(self) -> None:
        """Parallel embedding using pre-chunked JSON files under _chunks/chunks/."""
        self.logger.info(
            "Starting parallel embedding from chunked files with %d workers (from %d validated accounts)",
            self.num_workers,
            len(self.accounts),
        )
        try:
            chunk_files = self.find_new_chunk_files()
            if not chunk_files:
                self.logger.info("No new or modified chunk files to embed.")
                return

            worker_configs = self.distribute_chunk_files(chunk_files)

            # Start workers and monitor in real-time
            with mp.get_context("spawn").Pool(processes=self.num_workers) as _pool:
                # Use ProcessPoolExecutor-like pattern but with mp.Pool to avoid Windows pickling hiccups for Manager queues
                jobs = []
                for cfg in worker_configs:
                    jobs.append(
                        _pool.apply_async(
                            embed_chunks_worker,
                            (cfg, self.export_dir, self.stats_queue, self.control_queue, self.batch_size),
                        )
                    )

                # Monitor
                worker_states = self.monitor_workers(expected_workers=len(worker_configs))
                self.save_state(worker_states, expected_workers=len(worker_configs))

                # Ensure pool completes
                for j in jobs:
                    try:
                        j.get()
                    except Exception as e:
                        self.logger.error("Worker pool error: %s", e)

            # Merge artifacts and record run timestamp/index
            self.finalize_chunked_embeddings(chunk_files)

        except KeyboardInterrupt:
            self.logger.info("Shutdown signal received.")
        except Exception as e:
            self.logger.error("Coordinator error: %s", e, exc_info=True)

    def run_sequential(self) -> None:
        """
        Sequential embedding:
          - In chunked-files mode: loops chunk files and embeds in-process.
          - In legacy mode: loops conversation folders and embeds in-process.
        """
        self.logger.info("Running sequential indexing (mode=%s, chunked_files=%s)", self.mode, self.use_chunked_files)

        # Build a single "worker config" from the first account
        if not self.accounts:
            raise LLMError("No validated accounts available for sequential run.")
        acc = self.accounts[0]
        worker_cfg = {
            "project_id": acc["project_id"],
            "credentials_path": acc["credentials_path"],
            "chunk_start": 0,
            "chunk_end": 0,  # to be filled
            "worker_id": 0,
            "account_group": int(acc.get("account_group", 0)),
        }

        if self.use_chunked_files:
            chunk_files = self.find_new_chunk_files()
            if not chunk_files:
                self.logger.info("No new or modified chunk files to embed.")
                return
            worker_cfg["chunk_files"] = chunk_files
            worker_cfg["chunk_end"] = len(chunk_files)

            # Run worker inline
            embed_chunks_worker(worker_cfg, self.export_dir, self.stats_queue, self.control_queue, self.batch_size)
            # Finalize
            self.finalize_chunked_embeddings(chunk_files)
            return

        # Legacy: load folders and embed inline
        all_chunks = self.load_chunks()
        worker_cfg["chunk_end"] = len(all_chunks)
        index_worker(worker_cfg, all_chunks, self.export_dir, self.stats_queue, self.control_queue, self.batch_size)
        finalize_parallel_index(self.export_dir)

    # ----------------------- distribution / monitoring -----------------------

    def distribute_workload(self, total_chunks: int, num_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """Distribute legacy folder chunks across workers based on validated accounts."""
        n_workers = int(num_workers or self.num_workers)
        worker_accounts = self.accounts[:n_workers]

        configs: List[Dict[str, Any]] = []
        chunks_per_worker = total_chunks // n_workers if n_workers else 0
        remainder = total_chunks % n_workers if n_workers else 0
        current_start = 0

        for i, account in enumerate(worker_accounts):
            worker_chunks = chunks_per_worker + (1 if i < remainder else 0)
            cfg = {
                "project_id": account["project_id"],
                "credentials_path": account["credentials_path"],
                "chunk_start": current_start,
                "chunk_end": current_start + worker_chunks,
                "worker_id": i,
                "account_group": int(account.get("account_group", i // 3)),
            }
            configs.append(cfg)
            current_start += worker_chunks
        return configs

    def distribute_chunk_files(self, chunk_files: List[Path], num_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """Distribute pre-chunked JSON files across workers."""
        n_workers = int(num_workers or self.num_workers)
        worker_accounts = self.accounts[:n_workers]

        configs: List[Dict[str, Any]] = []
        files_per_worker = len(chunk_files) // n_workers if n_workers else 0
        remainder = len(chunk_files) % n_workers if n_workers else 0
        current_start = 0

        for i, account in enumerate(worker_accounts):
            worker_files = files_per_worker + (1 if i < remainder else 0)
            if worker_files > 0:
                cfg = {
                    "project_id": account["project_id"],
                    "credentials_path": account["credentials_path"],
                    "chunk_start": current_start,
                    "chunk_end": current_start + worker_files,
                    "worker_id": i,
                    "account_group": int(account.get("account_group", i // 3)),
                    "chunk_files": [str(p) for p in chunk_files[current_start : current_start + worker_files]],
                }
                configs.append(cfg)
                current_start += worker_files
        return configs

    def monitor_workers(self, expected_workers: int) -> Dict[int, ProcessingStats]:
        """Drain stats_queue and display progress until expected workers complete."""
        worker_states: Dict[int, ProcessingStats] = {}

        while True:
            # Drain queue
            while not self.stats_queue.empty():
                stats = self.stats_queue.get()
                worker_states[stats.worker_id] = stats

            # Display
            if worker_states:
                self.display_progress(worker_states)

            # Exit condition
            if (
                len(worker_states) == expected_workers
                and all(s.status in ("completed", "failed", "stopped") for s in worker_states.values())
            ):
                break

            time.sleep(2)

        self.print_summary(worker_states)
        return worker_states

    def display_progress(self, worker_states: Dict[int, ProcessingStats]) -> None:
        os.system("cls" if os.name == "nt" else "clear")
        print(f"{'='*80}\nVERTEX AI PARALLEL INDEXING - {datetime.now():%Y-%m-%d %H:%M:%S}\n{'='*80}")
        print(f"Workers reported: {len(worker_states)} / {self.num_workers} (from {len(self.accounts)} validated accounts)\n")

        # Group output by account_group 0, 1, ...
        groups: Dict[int, List[ProcessingStats]] = {}
        for s in worker_states.values():
            groups.setdefault(getattr(s, "account_group", 0), []).append(s)

        for group_id in sorted(groups.keys()):
            print(f"\n📁 GCP ACCOUNT GROUP {group_id + 1}:")
            for stats in sorted(groups[group_id], key=lambda s: s.worker_id):
                self._display_worker_stats(stats)

        total_processed = sum(s.chunks_processed for s in worker_states.values())
        total_chunks = sum(s.chunks_total for s in worker_states.values())
        total_errors = sum(s.errors for s in worker_states.values())

        print(f"\n{'='*80}")
        overall_progress = (total_processed / total_chunks * 100) if total_chunks > 0 else 0
        print(f"OVERALL: {total_processed}/{total_chunks} chunks ({overall_progress:.1f}%)")
        if total_errors > 0:
            print(f"Total Errors: {total_errors}")
        print("=" * 80)

    def _display_worker_stats(self, stats: ProcessingStats) -> None:
        status_symbol = {"running": "🔄", "completed": "✅", "failed": "❌", "stopped": "🛑"}.get(stats.status, "❓")
        progress_bar = self.create_progress_bar(stats.progress_percent, 30)
        print(f"  {status_symbol} {stats.project_id[:20]:20} {progress_bar} {stats.progress_percent:5.1f}%")
        if stats.errors > 0:
            print(f"     ⚠️ Errors: {stats.errors}")

    @staticmethod
    def create_progress_bar(percent: float, width: int = 30) -> str:
        filled = int(width * percent / 100)
        return "█" * filled + "░" * (width - filled)

    def print_summary(self, worker_states: Dict[int, ProcessingStats]) -> None:
        print(f"\n{'='*80}\nINDEXING COMPLETED\n{'='*80}")
        for wid, stats in sorted(worker_states.items()):
            print(
                f"Worker {wid} ({stats.project_id}): "
                f"{stats.chunks_processed}/{stats.chunks_total} chunks, "
                f"Status: {stats.status}, Errors: {stats.errors}"
            )
        total_processed = sum(s.chunks_processed for s in worker_states.values())
        total_errors = sum(s.errors for s in worker_states.values())
        print(f"\nTotal Chunks Processed: {total_processed}")
        print(f"Total Errors: {total_errors}")
        print(f"✓ Used {len(worker_states)} embedder workers (auto-set from {len(self.accounts)} validated accounts)")

    # ----------------------- finalization helpers -----------------------

    def finalize_chunked_embeddings(self, processed_files: List[Path | str]) -> None:
        """Merge artifacts, update embedded_chunks index, and mark run time (chunked-files mode)."""
        # Normalize path list
        processed_paths = [Path(p) for p in processed_files]

        # Merge worker outputs -> index
        finalize_parallel_index(self.export_dir)

        # Update embedded_chunks.json only for processed files
        embedded_index = self._load_embedded_chunks_index()
        for p in processed_paths:
            try:
                embedded_index[p.stem] = float(p.stat().st_mtime)
            except Exception:
                embedded_index[p.stem] = time.time()

        self._save_embedded_chunks_index(embedded_index)
        self._save_last_embedding_time()
        self.logger.info("✓ Finalized chunked embeddings run with %d files updated.", len(processed_paths))

    # ----------------------- CLI entry -----------------------

    def run(self) -> None:
        if self.mode == "parallel":
            self.run_parallel()
        else:
            self.run_sequential()


# -----------------------
# Worker functions
# -----------------------
def _set_worker_env(config: Dict[str, Any]) -> None:
    os.environ["GCP_PROJECT"] = str(config["project_id"])
    os.environ["GOOGLE_CLOUD_PROJECT"] = str(config["project_id"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(config["credentials_path"])
    os.environ["EMBED_PROVIDER"] = "vertex"
    os.environ["VERTEX_EMBED_MODEL"] = os.getenv("VERTEX_EMBED_MODEL", "gemini-embedding-001")
    os.environ["GCP_REGION"] = os.getenv("GCP_REGION", "global")


def index_worker(
    config: Dict[str, Any],
    chunks: List[Any],
    export_dir: Path,
    stats_queue: Any,
    control_queue: Any,
    batch_size: int,
) -> None:
    """Worker process for embedding (legacy folder-mode)."""
    _set_worker_env(config)

    log_file = export_dir / INDEX_DIRNAME / f"worker_{config['worker_id']}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - Worker{config["worker_id"]} - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger(f"Worker{config['worker_id']}")

    # Lazy imports to avoid heavy deps at module import time
    try:
        try:
            from .llm_client import embed_texts  # type: ignore
            from .utils import load_conversation  # type: ignore
            from .email_indexer import _build_doc_entries, _extract_manifest_metadata  # type: ignore
        except Exception:
            from emailops.llm_client import embed_texts  # type: ignore
            from emailops.utils import load_conversation  # type: ignore
            from emailops.email_indexer import _build_doc_entries, _extract_manifest_metadata  # type: ignore
    except Exception as e:  # pragma: no cover
        logger.error("Failed to import worker dependencies: %s", e)
        return

    stats = ProcessingStats(
        worker_id=int(config["worker_id"]),
        project_id=str(config["project_id"]),
        chunks_processed=0,
        chunks_total=len(chunks),
        start_time=time.time(),
        last_update=time.time(),
        errors=0,
        status="running",
        account_group=int(config.get("account_group", 0)),
    )
    stats_queue.put(stats)
    last_sent_ts = time.time()

    for i in range(0, len(chunks), max(1, int(batch_size))):
        if not control_queue.empty():
            command = control_queue.get()
            if command == "SHUTDOWN":
                stats.status = "stopped"
                stats_queue.put(stats)
                return

        batch = chunks[i : i + min(batch_size, len(chunks) - i)]
        batch_texts: List[str] = []
        valid_batch_chunks: List[Dict[str, Any]] = []

        for chunk in batch:
            if isinstance(chunk, dict):
                if "path" in chunk and not chunk.get("text"):
                    try:
                        folder_path = Path(chunk["path"])
                        conv = load_conversation(folder_path)  # extracts Conversation.txt + attachments
                        metadata = _extract_manifest_metadata(conv)  # subject, dates, participants
                        subject = metadata.get("subject", "")
                        end_date = metadata.get("end_date")
                        base_id = folder_path.name

                        # Process main conversation
                        convo_text = conv.get("conversation_txt", "")
                        if convo_text:
                            prepared_text = f"Subject: {subject}\n\n{convo_text}" if subject else convo_text
                            doc_entries = _build_doc_entries(
                                prepared_text,
                                doc_id=f"{base_id}::conversation",
                                doc_path=str(folder_path / "Conversation.txt"),
                                subject=subject,
                                end_date=end_date,
                                metadata=metadata,
                            )
                            for entry in doc_entries:
                                t = entry.get("text", "")
                                if isinstance(t, str) and t.strip():
                                    batch_texts.append(t)
                                    valid_batch_chunks.append(entry)

                        # Process attachments
                        for att_i, attachment in enumerate(conv.get("attachments", []), 1):
                            if isinstance(attachment, dict) and attachment.get("text"):
                                att_path = attachment.get("path", "")
                                doc_entries = _build_doc_entries(
                                    attachment["text"],
                                    doc_id=f"{base_id}::att{att_i}",
                                    doc_path=str(att_path),
                                    subject=subject,
                                    end_date=end_date,
                                    metadata=metadata,
                                )
                                for entry in doc_entries:
                                    t = entry.get("text", "")
                                    if isinstance(t, str) and t.strip():
                                        batch_texts.append(t)
                                        valid_batch_chunks.append(entry)
                    except Exception as e:
                        logger.warning("Failed to process conversation %s: %s", chunk.get("path", ""), e)
                        continue
                else:
                    text = str(chunk.get("text", ""))
                    if text.strip():
                        batch_texts.append(text)
                        valid_batch_chunks.append(chunk)
            else:
                text = str(chunk)
                if text.strip():
                    batch_texts.append(text)
                    valid_batch_chunks.append({"id": f"ad hoc::{i}", "text": text})

        if not batch_texts:
            continue

        max_retries = 3
        for attempt in range(max_retries):
            try:
                embeddings = embed_texts(batch_texts, provider="vertex")  # llm_client handles rotation/backoff
                store_embeddings(valid_batch_chunks, embeddings, export_dir, int(config["worker_id"]), i)
                stats.chunks_processed += len(valid_batch_chunks)
                stats.last_update = time.time()
                if (stats.chunks_processed % 10 == 0) or (time.time() - last_sent_ts > 5):
                    stats_queue.put(stats)
                    last_sent_ts = time.time()
                break
            except Exception as e:
                logger.warning("Attempt %d failed: %s", attempt + 1, e)
                stats.errors += 1
                if attempt < max_retries - 1:
                    time.sleep(10 * (2 ** attempt))
                else:
                    logger.error("Failed to process batch after %d attempts", max_retries)

    stats.status = "completed"
    stats_queue.put(stats)


def embed_chunks_worker(
    config: Dict[str, Any],
    export_dir: Path,
    stats_queue: Any,
    control_queue: Any,
    batch_size: int,
) -> None:
    """Worker that processes pre-chunked JSON files under _chunks/chunks/."""
    _set_worker_env(config)

    log_file = export_dir / INDEX_DIRNAME / f"embedder_{config['worker_id']}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - EmbedWorker{config["worker_id"]} - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger(f"EmbedWorker{config['worker_id']}")

    # Lazy imports
    try:
        try:
            from .llm_client import embed_texts  # type: ignore
            from .utils import load_conversation  # type: ignore
            from .email_indexer import _extract_manifest_metadata  # type: ignore
        except Exception:
            from emailops.llm_client import embed_texts  # type: ignore
            from emailops.utils import load_conversation  # type: ignore
            from emailops.email_indexer import _extract_manifest_metadata  # type: ignore
    except Exception as e:  # pragma: no cover
        logger.error("Failed to import worker dependencies: %s", e)
        return

    # Normalize chunk_files
    raw_files = config.get("chunk_files", [])
    chunk_files: List[Path] = [Path(p) for p in raw_files]

    # Compute total planned chunks for progress
    total_chunks = 0
    for cf in chunk_files:
        try:
            with open(cf, "r", encoding="utf-8") as f:
                cd = json.load(f)
            total_chunks += len(cd.get("chunks", []))
        except Exception:
            pass

    stats = ProcessingStats(
        worker_id=int(config["worker_id"]),
        project_id=str(config["project_id"]),
        chunks_processed=0,
        chunks_total=total_chunks,
        start_time=time.time(),
        last_update=time.time(),
        errors=0,
        status="running",
        account_group=int(config.get("account_group", 0)),
    )
    stats_queue.put(stats)

    batch_accum: List[Dict[str, Any]] = []
    batch_texts: List[str] = []
    batch_idx = 0
    last_sent_ts = time.time()

    for chunk_file in chunk_files:
        if not control_queue.empty() and control_queue.get() == "SHUTDOWN":
            stats.status = "stopped"
            stats_queue.put(stats)
            return

        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)

            # Determine conversation folder and metadata
            doc_path = Path(chunk_data.get("doc_path", ""))
            convo_dir = doc_path.parent if doc_path else None

            conv_md: Dict[str, Any] = {}
            if convo_dir and convo_dir.exists():
                conv = load_conversation(convo_dir)
                conv_md = _extract_manifest_metadata(conv)

            subject = conv_md.get("subject", "")
            end_date = conv_md.get("end_date")

            # Robust conv_id derivation
            conv_id = ""
            if convo_dir and convo_dir.exists():
                conv_id = convo_dir.name
            else:
                doc_id_raw = str(chunk_data.get("doc_id", ""))
                conv_id = doc_id_raw.split("::", 1)[0] if "::" in doc_id_raw else doc_id_raw

            # Extract chunks from file
            file_chunks = chunk_data.get("chunks", [])
            for ch in file_chunks:
                # Ensure safe defaults
                ch_idx = int(ch.get("chunk_index", 0))
                ch_text = str(ch.get("text", ""))

                chunk_id = f"{conv_id}::conversation::chunk{ch_idx}"
                entry = {
                    "id": chunk_id,
                    "text": ch_text,
                    "path": str(doc_path) if doc_path else "",
                    "chunk_index": ch_idx,
                    # enrich metadata for downstream mapping (aligned w/ email_indexer)
                    "conv_id": conv_id,
                    "doc_type": "conversation",
                    "subject": subject,
                    "date": end_date,
                    "start_date": conv_md.get("start_date"),
                    "from_email": conv_md.get("from_email", ""),
                    "from_name": conv_md.get("from_name", ""),
                    "to_recipients": conv_md.get("to_recipients", []),
                    "cc_recipients": conv_md.get("cc_recipients", []),
                    "to_emails": conv_md.get("to_emails", []),
                    "cc_emails": conv_md.get("cc_emails", []),
                    "modified_time": doc_path.stat().st_mtime if doc_path and doc_path.exists() else time.time(),
                }

                # Accumulate for batched embedding
                if isinstance(ch_text, str) and ch_text.strip():
                    batch_accum.append(entry)
                    batch_texts.append(ch_text)

                # Emit when reaching batch size
                if len(batch_texts) >= int(batch_size):
                    try:
                        embs = embed_texts(batch_texts, provider="vertex")
                        store_embeddings(batch_accum, embs, export_dir, int(config["worker_id"]), batch_idx)
                        stats.chunks_processed += len(batch_texts)
                        stats.last_update = time.time()
                        if (stats.chunks_processed % 50 == 0) or (time.time() - last_sent_ts > 5):
                            stats_queue.put(stats)
                            last_sent_ts = time.time()
                    except Exception as e:
                        logger.error("Failed to embed batch in %s: %s", chunk_file, e)
                        stats.errors += 1
                    finally:
                        batch_accum.clear()
                        batch_texts.clear()
                        batch_idx += 1

        except Exception as e:
            logger.error("Failed to process chunk file %s: %s", chunk_file, e)
            stats.errors += 1

    # Flush tail batch
    if batch_texts:
        try:
            embs = embed_texts(batch_texts, provider="vertex")
            store_embeddings(batch_accum, embs, export_dir, int(config["worker_id"]), batch_idx)
            stats.chunks_processed += len(batch_texts)
        except Exception as e:
            logger.error("Failed to embed final batch: %s", e)
            stats.errors += 1

    stats.status = "completed"
    stats_queue.put(stats)


# -----------------------
# Storage / Finalization
# -----------------------
def store_embeddings(
    chunks: List[Any],
    embeddings: Any,
    export_dir: Path,
    worker_id: int,
    batch_idx: int,
) -> None:
    """Store embeddings as a pickle file (one-to-one with provided chunks)."""
    output_dir = export_dir / INDEX_DIRNAME / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"worker_{worker_id}_batch_{batch_idx}_{timestamp}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)


def _infer_num_folders(mapping: List[Dict[str, Any]]) -> int:
    """Prefer conv_id when available; fall back to parent folder names from paths."""
    conv_ids = {m.get("conv_id") for m in mapping if m.get("conv_id")}
    if conv_ids:
        return len(conv_ids)
    folders = set()
    for m in mapping:
        p = m.get("path") or ""
        if p:
            try:
                folders.add(Path(p).parent.name)
            except Exception:
                pass
    return len(folders)


def finalize_parallel_index(export_dir: Path) -> None:
    """Merge all worker outputs into final index (mapping.json, embeddings.npy, index.faiss, meta.json)."""
    import numpy as np

    # Metadata helpers
    try:
        try:
            from .index_metadata import create_index_metadata, save_index_metadata  # type: ignore
        except Exception:
            from emailops.index_metadata import create_index_metadata, save_index_metadata  # type: ignore
    except Exception as e:  # pragma: no cover
        logging.getLogger("VertexIndexer").error("Failed to import index_metadata: %s", e)
        return

    logger = logging.getLogger("VertexIndexer")
    logger.info("Finalizing parallel index...")

    emb_dir = export_dir / INDEX_DIRNAME / "embeddings"
    merged_chunks: List[Dict[str, Any]] = []
    merged_embs: List[np.ndarray] = []

    pickle_files = sorted(emb_dir.glob("worker_*_batch_*.pkl"))
    if not pickle_files:
        logger.warning("No embedding pickle files found to merge.")
        return

    for pkl in pickle_files:
        try:
            with open(pkl, "rb") as f:
                obj = pickle.load(f)
            merged_chunks.extend(obj["chunks"])
            merged_embs.append(np.asarray(obj["embeddings"], dtype="float32"))
        except Exception as e:
            logger.error("Failed to load pickle file %s: %s", pkl, e)
            continue

    if not merged_embs:
        logger.error("No embeddings were successfully loaded. Aborting finalization.")
        return

    embs = np.vstack(merged_embs)

    # Build rich mapping aligned to EmailOps schema
    mapping: List[Dict[str, Any]] = []
    now_ts = time.time()
    for i, ch in enumerate(merged_chunks):
        if not isinstance(ch, dict):
            # Very defensive fallback
            mapping.append(
                {
                    "id": f"item::{i}",
                    "conv_id": "",
                    "doc_type": "conversation",
                    "path": "",
                    "subject": "",
                    "date": None,
                    "start_date": None,
                    "snippet": str(ch)[:500],
                    "modified_time": now_ts,
                    "from_email": "",
                    "from_name": "",
                    "to_recipients": [],
                    "cc_recipients": [],
                    "to_emails": [],
                    "cc_emails": [],
                }
            )
            continue

        cid = ch.get("id") or f"item::{i}"
        path_str = ch.get("path", "")
        snippet = (ch.get("text", "") or "")[:500]
        doc_type = ch.get("doc_type") or ("attachment" if "::att" in str(cid) else "conversation")
        conv_id = ch.get("conv_id") or str(cid).split("::", 1)[0]

        mapping.append(
            {
                "id": cid,
                "conv_id": conv_id,
                "doc_type": doc_type,
                "path": path_str,
                "subject": ch.get("subject", ""),
                "date": ch.get("date"),
                "start_date": ch.get("start_date"),
                "snippet": snippet,
                "modified_time": ch.get("modified_time", now_ts),
                # participants
                "from_email": ch.get("from_email", ""),
                "from_name": ch.get("from_name", ""),
                "to_recipients": ch.get("to_recipients", []),
                "cc_recipients": ch.get("cc_recipients", []),
                "to_emails": ch.get("to_emails", []),
                "cc_emails": ch.get("cc_emails", []),
                # attachment hints (optional)
                "attachment_name": ch.get("attachment_name", ""),
                "attachment_type": ch.get("attachment_type", ""),
                "attachment_size": ch.get("attachment_size", 0),
                "attachment_index": ch.get("attachment_index", 0),
            }
        )

    out_dir = export_dir / INDEX_DIRNAME
    np.save(out_dir / "embeddings.npy", embs.astype("float32"))

    # Try to build FAISS index
    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(embs.shape[1])
        vectors_to_add = np.ascontiguousarray(embs).astype("float32")
        index.add(vectors_to_add)
        faiss.write_index(index, str(out_dir / "index.faiss"))
    except Exception as e:
        logger.warning("Could not create FAISS index: %s", e)

    (out_dir / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    # Count distinct conversations by conv_id
    num_folders = _infer_num_folders(mapping)
    meta = create_index_metadata(
        "vertex",
        num_documents=len(mapping),
        num_folders=num_folders,
        index_dir=out_dir,
        custom_metadata={"actual_dimensions": int(embs.shape[1])},
    )
    save_index_metadata(meta, out_dir)
    logger.info("✓ Index finalization complete.")


# -----------------------
# CLI
# -----------------------
def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Vertex AI Email Indexer (multi-account, production-ready)")
    ap.add_argument("--root", required=True, help="Export root containing conversation folders")
    ap.add_argument("--mode", choices=["parallel", "sequential"], default="parallel", help="Execution mode")
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("EMBED_BATCH", "8")), help="Embedding batch size per worker")
    ap.add_argument("--resume", action="store_true", default=True, help="Resume from previous state when possible")
    ap.add_argument("--no-resume", dest="resume", action="store_false", help="Do not resume previous state")
    ap.add_argument("--test-mode", action="store_true", help="Use a small subset of chunks")
    ap.add_argument("--test-chunks", type=int, default=100, help="Number of chunks when --test-mode")
    ap.add_argument("--incremental", action="store_true", help="Use sequential incremental fallback in legacy mode")
    ap.add_argument("--force-rebuild", action="store_true", help="Not used here; reserved for compatibility")
    ap.add_argument("--chunked-files", action="store_true", help="Use pre-chunked files under _chunks/chunks/")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    ap = _build_cli()
    args = ap.parse_args(argv)

    mp.freeze_support()  # Safe on Windows

    indexer = VertexIndexer(
        export_dir=args.root,
        mode=args.mode,
        resume=args.resume,
        batch_size=args.batch_size,
        test_mode=args.test_mode,
        test_chunks=args.test_chunks,
        incremental=args.incremental,
        force_rebuild=args.force_rebuild,
        use_chunked_files=args.chunked_files,
    )
    indexer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
