#!/usr/bin/env python3
"""
Unified Processing Module for EmailOps
Combines text processing, chunking, and embedding operations
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
import queue
import signal
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from contextlib import suppress

# -------------------------
# Data Classes
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
    chunk_config: Any  # ChunkConfig from emailops

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
    def estimated_time_remaining(self) -> Optional[timedelta]:
        if self.chunks_processed == 0:
            return None
        elapsed = time.time() - self.start_time
        rate = self.chunks_processed / max(elapsed, 1e-6)
        remaining = self.chunks_total - self.chunks_processed
        if rate > 0:
            return timedelta(seconds=remaining / rate)
        return None


# -------------------------
# Main Processor Class
# -------------------------

class UnifiedProcessor:
    """Unified processor for text and embedding operations"""

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
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.mode = mode
        self.num_workers = num_workers or os.cpu_count() or 1
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.resume = resume
        self.test_mode = test_mode
        self._log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Setup directories based on mode
        if mode == "chunk":
            self.output_dir = self.root_dir / "_chunks"
            self.state_dir = self.output_dir / "_chunker_state"
            self.chunks_dir = self.output_dir / "chunks"
        else:  # embedding modes
            self.index_dir = self.root_dir / "_index"
            self.index_dir.mkdir(parents=True, exist_ok=True)
            
        self.setup_logging()
        
        # Multiprocessing setup
        self.ctx = mp.get_context("spawn" if os.name == "nt" else "forkserver")
        self._shutdown_initiated = False

    def setup_logging(self):
        """Configure logging"""
        log_dir = self.state_dir if hasattr(self, 'state_dir') else self.index_dir if hasattr(self, 'index_dir') else Path.cwd()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{self.mode}_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        logging.basicConfig(
            level=self._log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
            force=True,
        )
        self.logger = logging.getLogger(f"UnifiedProcessor.{self.mode}")

    # ---- Text Chunking Operations ----
    
    def chunk_documents(self, input_dir: str, file_pattern: str = "*.txt"):
        """Process documents into chunks"""
        from emailops.text_chunker import TextChunker, ChunkConfig
        from emailops.utils import read_text_file
        
        self.input_dir = Path(input_dir)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        chunk_config = ChunkConfig(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            respect_sentences=True,
            respect_paragraphs=True,
            progressive_scaling=True,
        )
        
        # Find documents to process
        jobs = self._find_documents(file_pattern)
        if not jobs:
            self.logger.info("No new documents to process")
            return
            
        self.logger.info(f"Found {len(jobs)} documents to process")
        
        # Process documents
        if self.num_workers > 1:
            self._parallel_chunk(jobs, chunk_config)
        else:
            self._sequential_chunk(jobs, chunk_config)

    def _find_documents(self, file_pattern: str) -> List[ChunkJob]:
        """Find documents to process"""
        jobs = []
        for file_path in self.input_dir.rglob(file_pattern):
            if not file_path.is_file():
                continue
            
            rel = file_path.relative_to(self.input_dir).as_posix()
            doc_id = rel
            
            try:
                st = file_path.stat()
                if st.st_size == 0:
                    continue
                    
                jobs.append(ChunkJob(
                    doc_id=doc_id,
                    doc_path=file_path,
                    file_size=st.st_size
                ))
            except Exception as e:
                self.logger.warning(f"Could not stat file {file_path}: {e}")
                
        jobs.sort(key=lambda j: -j.file_size)  # Process larger files first
        
        if self.test_mode:
            jobs = jobs[:10]  # Limit to 10 files in test mode
            
        return jobs

    def _parallel_chunk(self, jobs: List[ChunkJob], chunk_config):
        """Process chunks in parallel"""
        # Distribute work
        worker_configs = self._distribute_chunking_work(jobs, chunk_config)
        
        # Create queues
        stats_queue = self.ctx.Queue()
        control_queue = self.ctx.Queue()
        workers = []
        
        # Start workers
        for config in worker_configs:
            worker = self.ctx.Process(
                target=self._chunk_worker,
                args=(config, stats_queue, control_queue),
            )
            worker.start()
            workers.append(worker)
            
        # Monitor progress
        self._monitor_workers(workers, stats_queue, len(worker_configs))
        
        # Cleanup
        for worker in workers:
            worker.join(timeout=5)

    def _sequential_chunk(self, jobs: List[ChunkJob], chunk_config):
        """Process chunks sequentially"""
        from emailops.text_chunker import TextChunker
        from emailops.utils import read_text_file
        
        chunker = TextChunker(chunk_config)
        
        for i, job in enumerate(jobs):
            self.logger.info(f"Processing {i+1}/{len(jobs)}: {job.doc_id}")
            
            try:
                text = read_text_file(job.doc_path)
                chunks = chunker.chunk_text(text, metadata={
                    "doc_id": job.doc_id,
                    "doc_path": str(job.doc_path)
                })
                
                if chunks:
                    self._save_chunks(job.doc_id, chunks, job.file_size)
                    
            except Exception as e:
                self.logger.error(f"Error processing {job.doc_path}: {e}")

    def _chunk_worker(self, config: WorkerConfig, stats_queue, control_queue):
        """Worker process for chunking"""
        from emailops.text_chunker import TextChunker
        from emailops.utils import read_text_file
        
        chunker = TextChunker(config.chunk_config)
        
        stats = WorkerStats(
            worker_id=config.worker_id,
            docs_processed=0,
            docs_total=len(config.jobs_assigned),
            chunks_created=0,
            bytes_processed=0,
            bytes_total=sum(job.file_size for job in config.jobs_assigned),
            start_time=time.time(),
            last_update=time.time(),
            errors=0,
            status="running",
        )
        
        stats_queue.put(stats)
        
        for job in config.jobs_assigned:
            # Check for shutdown
            if not control_queue.empty():
                try:
                    cmd = control_queue.get_nowait()
                    if cmd == "SHUTDOWN":
                        stats.status = "stopped"
                        stats_queue.put(stats)
                        return
                except:
                    pass
            
            stats.current_doc = job.doc_id
            
            try:
                text = read_text_file(job.doc_path)
                chunks = chunker.chunk_text(text, metadata={
                    "doc_id": job.doc_id,
                    "doc_path": str(job.doc_path)
                })
                
                if chunks:
                    self._save_chunks(job.doc_id, chunks, job.file_size)
                    stats.chunks_created += len(chunks)
                    
                stats.docs_processed += 1
                stats.bytes_processed += job.file_size
                
            except Exception as e:
                self.logger.error(f"Worker {config.worker_id} error: {e}")
                stats.errors += 1
            
            stats.last_update = time.time()
            stats_queue.put(stats)
        
        stats.status = "completed"
        stats_queue.put(stats)

    def _save_chunks(self, doc_id: str, chunks: List[Dict], file_size: int):
        """Save chunks to JSON file"""
        safe_id = re.sub(r"[^A-Za-z0-9._ -]+", "_", doc_id)
        h = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:8]
        output_file = self.chunks_dir / f"{safe_id}.{h}.json"
        
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
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)

    def _distribute_chunking_work(self, jobs: List[ChunkJob], chunk_config) -> List[WorkerConfig]:
        """Distribute jobs across workers"""
        n_workers = min(self.num_workers, len(jobs))
        worker_bins = [[] for _ in range(n_workers)]
        worker_sizes = [0] * n_workers
        
        for job in jobs:
            idx = worker_sizes.index(min(worker_sizes))
            worker_bins[idx].append(job)
            worker_sizes[idx] += job.file_size
        
        configs = []
        for i, worker_jobs in enumerate(worker_bins):
            if worker_jobs:
                configs.append(WorkerConfig(
                    worker_id=i,
                    jobs_assigned=worker_jobs,
                    chunk_config=chunk_config,
                ))
        
        return configs

    # ---- Embedding Operations ----
    
    def create_embeddings(self, use_chunked_files: bool = True):
        """Create embeddings from chunks or documents"""
        try:
            from emailops.env_utils import get_worker_configs, LLMError
            from emailops.llm_client import embed_texts
        except ImportError as e:
            self.logger.error(f"Missing dependencies: {e}")
            return
        
        # Load validated accounts
        accounts = get_worker_configs()
        if not accounts:
            raise LLMError("No validated accounts found")
        
        self.logger.info(f"Creating embeddings with {len(accounts)} accounts")
        
        if use_chunked_files:
            self._embed_from_chunks(accounts)
        else:
            self._embed_from_documents(accounts)

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
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                
                chunks = data.get("chunks", [])
                embeddings = np.asarray(data.get("embeddings", []), dtype="float32")
                
                if len(chunks) == embeddings.shape[0]:
                    all_chunks.extend(chunks)
                    all_embeddings.append(embeddings)
                else:
                    self.logger.warning(f"Shape mismatch in {pkl_file.name}")
                    
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
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved {len(all_chunks)} chunks with embeddings")
        
        # Try to create FAISS index
        try:
            import faiss
            index = faiss.IndexFlatIP(merged_embeddings.shape[1])
            index.add(np.ascontiguousarray(merged_embeddings).astype("float32"))
            faiss.write_index(index, str(self.index_dir / "index.faiss"))
            self.logger.info("Created FAISS index")
        except:
            self.logger.warning("Could not create FAISS index")
        
        # Clean up batches if requested
        if remove_batches:
            for pkl_file in pkl_files:
                try:
                    pkl_file.unlink()
                except:
                    pass
            self.logger.info(f"Removed {len(pkl_files)} batch files")

    def fix_failed_embeddings(self):
        """Fix chunks with zero vectors"""
        from emailops.llm_client import embed_texts
        
        emb_dir = self.index_dir / "embeddings"
        if not emb_dir.exists():
            self.logger.error("No embeddings directory found")
            return
        
        pkl_files = list(emb_dir.glob("*.pkl"))
        self.logger.info(f"Scanning {len(pkl_files)} files for zero vectors...")
        
        total_fixed = 0
        
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                
                embeddings = np.array(data["embeddings"], dtype="float32")
                chunks = data["chunks"]
                
                # Find zero vectors
                zero_mask = np.all(embeddings == 0, axis=1)
                num_zeros = int(np.sum(zero_mask))
                
                if num_zeros > 0:
                    self.logger.info(f"Found {num_zeros} zero vectors in {pkl_file.name}")
                    
                    # Re-embed
                    failed_indices = np.where(zero_mask)[0]
                    texts = [chunks[i]["text"] for i in failed_indices]
                    
                    try:
                        new_embeddings = embed_texts(texts, provider="vertex")
                        new_embs = np.array(new_embeddings, dtype="float32")
                        
                        # Replace zeros
                        for idx, failed_idx in enumerate(failed_indices):
                            embeddings[failed_idx] = new_embs[idx]
                        
                        # Save updated
                        data["embeddings"] = embeddings
                        with open(pkl_file, "wb") as f:
                            pickle.dump(data, f)
                        
                        self.logger.info(f"Fixed {num_zeros} vectors")
                        total_fixed += num_zeros
                        
                    except Exception as e:
                        self.logger.error(f"Failed to re-embed: {e}")
                        
            except Exception as e:
                self.logger.error(f"Error processing {pkl_file.name}: {e}")
        
        self.logger.info(f"Fixed {total_fixed} zero vectors total")

    def _embed_from_chunks(self, accounts):
        """Create embeddings from chunk files"""
        chunks_dir = self.root_dir / "_chunks" / "chunks"
        if not chunks_dir.exists():
            self.logger.error("No chunks directory found")
            return
        
        chunk_files = list(chunks_dir.glob("*.json"))
        if not chunk_files:
            self.logger.error("No chunk files found")
            return
        
        self.logger.info(f"Found {len(chunk_files)} chunk files to embed")
        
        # TODO: Implement parallel embedding from chunk files
        self.logger.warning("Chunk embedding not fully implemented")

    def _embed_from_documents(self, accounts):
        """Create embeddings directly from documents"""
        # TODO: Implement document embedding
        self.logger.warning("Document embedding not fully implemented")

    # ---- Monitoring ----
    
    def _monitor_workers(self, workers, stats_queue, expected_count):
        """Monitor worker progress"""
        worker_states = {}
        
        while True:
            # Drain queue
            drained = False
            while True:
                try:
                    stats = stats_queue.get_nowait()
                    worker_states[stats.worker_id] = stats
                    drained = True
                except queue.Empty:
                    break
            
            # Display progress
            if drained:
                self._display_progress(worker_states)
            
            # Check if all done
            if len(worker_states) == expected_count:
                if all(s.status in ("completed", "failed", "stopped") for s in worker_states.values()):
                    break
            
            time.sleep(0.5)
        
        self._print_summary(worker_states)

    def _display_progress(self, worker_states):
        """Display progress"""
        if sys.stdout.isatty():
            os.system("cls" if os.name == "nt" else "clear")
        
        print("=" * 80)
        print(f"PROCESSING - {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 80)
        
        for wid in sorted(worker_states.keys()):
            stats = worker_states[wid]
            print(f"\nWorker {wid}: [{stats.progress_percent:5.1f}%] {stats.status}")
            
            if hasattr(stats, 'docs_processed'):
                print(f"  Docs: {stats.docs_processed}/{stats.docs_total}")
                print(f"  Chunks: {stats.chunks_created}")
            elif hasattr(stats, 'chunks_processed'):
                print(f"  Chunks: {stats.chunks_processed}/{stats.chunks_total}")
            
            if stats.errors > 0:
                print(f"  Errors: {stats.errors}")

    def _print_summary(self, worker_states):
        """Print final summary"""
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETED")
        print("=" * 80)
        
        total_processed = 0
        total_errors = 0
        
        for stats in worker_states.values():
            if hasattr(stats, 'docs_processed'):
                total_processed += stats.docs_processed
            elif hasattr(stats, 'chunks_processed'):
                total_processed += stats.chunks_processed
            total_errors += stats.errors
        
        print(f"Total processed: {total_processed}")
        if total_errors > 0:
            print(f"Total errors: {total_errors}")


# -------------------------
# CLI Entry Point
# -------------------------

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
    chunk_parser.add_argument("--chunk-size", type=int, default=1600)
    chunk_parser.add_argument("--chunk-overlap", type=int, default=200)
    chunk_parser.add_argument("--pattern", default="*.txt", help="File pattern")
    chunk_parser.add_argument("--test", action="store_true", help="Test mode")
    
    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Create embeddings")
    embed_parser.add_argument("--root", required=True, help="Root directory")
    embed_parser.add_argument("--batch-size", type=int, default=64)
    embed_parser.add_argument("--workers", type=int, help="Number of workers")
    embed_parser.add_argument("--chunked", action="store_true", help="Use chunk files")
    
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
    
    # Execute command
    if args.command == "chunk":
        processor = UnifiedProcessor(
            root_dir=args.output,
            mode="chunk",
            num_workers=args.workers,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            test_mode=args.test,
        )
        processor.chunk_documents(args.input, args.pattern)
        
    elif args.command == "embed":
        processor = UnifiedProcessor(
            root_dir=args.root,
            mode="embed",
            num_workers=args.workers,
            batch_size=args.batch_size,
        )
        processor.create_embeddings(use_chunked_files=args.chunked)
        
    elif args.command == "repair":
        processor = UnifiedProcessor(root_dir=args.root, mode="repair")
        processor.repair_index(remove_batches=args.remove_batches)
        
    elif args.command == "fix":
        processor = UnifiedProcessor(root_dir=args.root, mode="fix")
        processor.fix_failed_embeddings()
    
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    mp.freeze_support()
