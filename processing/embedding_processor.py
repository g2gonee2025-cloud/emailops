#!/usr/bin/env python3
"""
Unified Embedding Processor for EmailOps
Combines vertex indexing, repair, and finalization functionality
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
from queue import Empty
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

# -----------------------
# Imports
# -----------------------
try:
    from emailops.env_utils import get_worker_configs, LLMError
    from emailops.llm_client import embed_texts
    from emailops.utils import load_conversation
    from emailops.email_indexer import _build_doc_entries, _extract_manifest_metadata
    from emailops.index_metadata import create_index_metadata, save_index_metadata
except ImportError as e:
    print(f"Error importing dependencies: {e}", file=sys.stderr)
    sys.exit(1)

# -----------------------
# Constants
# -----------------------
INDEX_DIRNAME = os.getenv("INDEX_DIRNAME", "_index")
MAPPING_FILENAME = "mapping.json"
EMBEDDINGS_FILENAME = "embeddings.npy"
FAISS_INDEX_FILENAME = "index.faiss"
FILE_TIMES_FILENAME = "file_times.json"
LAST_RUN_FILENAME = "last_run.txt"

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
# Main Embedding Processor
# -----------------------
class EmbeddingProcessor:
    """Unified coordinator for embedding and indexing operations"""

    def __init__(
        self,
        export_dir: str,
        mode: str = "parallel",
        resume: bool = True,
        batch_size: int = 64,
        test_mode: bool = False,
        test_chunks: int = 100,
        use_chunked_files: bool = False,
    ) -> None:
        self.export_dir = Path(export_dir).expanduser().resolve()
        self.mode = mode
        self.resume = resume
        self.batch_size = int(batch_size)
        self.test_mode = bool(test_mode)
        self.test_chunks = int(test_chunks)
        self.use_chunked_files = bool(use_chunked_files)

        # Load validated accounts
        self.accounts = self._load_worker_accounts()
        self.num_workers = len(self.accounts)
        if self.num_workers == 0:
            raise LLMError("No validated accounts found. Please run: python validate_accounts.py")

        # Paths
        self.index_dir = self.export_dir / INDEX_DIRNAME
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir = self.export_dir / "_chunks" / "chunks" if self.use_chunked_files else None
        self.embedded_chunks_file = self.index_dir / "embedded_chunks.json" if self.use_chunked_files else None
        self.state_file = self.index_dir / "indexer_state.json"

        # Logging
        self.setup_logging()

        # IPC / multiprocessing
        self.ctx = mp.get_context("spawn")
        self.manager = self.ctx.Manager()
        self.stats_queue = self.manager.Queue()
        self.control_queue = self.manager.Queue()
        self.workers: List[mp.Process] = []

    def _load_worker_accounts(self) -> List[Dict[str, Any]]:
        """Load validated accounts"""
        validated_accounts = get_worker_configs()
        if not validated_accounts:
            raise LLMError("No valid accounts found")
        return [acc.to_dict() for acc in validated_accounts]

    def setup_logging(self) -> None:
        """Configure logging"""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.index_dir / f"embedder_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
            force=True,
        )
        self.logger = logging.getLogger("EmbeddingProcessor")

    # ---- Repair/Fix functionality ----
    
    def repair_index(self, remove_batches: bool = False) -> None:
        """Repair/merge batch pickles into final index"""
        emb_dir = self.index_dir / "embeddings"
        if not emb_dir.exists():
            raise SystemExit(f"No batch directory found at: {emb_dir}")

        self.logger.info("Scanning batches in %s", emb_dir)
        pkl_files = sorted(emb_dir.glob("worker_*_batch_*.pkl"))
        if not pkl_files:
            raise SystemExit("No batch pickle files found to merge.")

        all_pairs: List[Tuple[Dict[str, Any], np.ndarray]] = []
        for p in pkl_files:
            try:
                with open(p, "rb") as f:
                    obj = pickle.load(f)
                chunks, embs = self._coerce_batch(obj)
                if not chunks or embs.size == 0:
                    self.logger.warning("Empty batch in %s; skipping.", p.name)
                    continue
                if len(chunks) != embs.shape[0]:
                    n = min(len(chunks), embs.shape[0])
                    self.logger.warning("Trimming mismatched batch %s to %d rows.", p.name, n)
                    chunks, embs = chunks[:n], embs[:n]
                for i in range(len(chunks)):
                    all_pairs.append((chunks[i], embs[i]))
            except Exception as e:
                self.logger.error("Failed to read %s: %s", p, e)

        if not all_pairs:
            raise SystemExit("No valid data found in batches.")

        self.logger.info("Merging %d records from %d batches...", len(all_pairs), len(pkl_files))
        mapping, embs = self._dedupe_and_stabilize(all_pairs)
        self._ensure_snippet(mapping, limit=500)

        if embs.shape[0] != len(mapping):
            raise SystemExit(f"Alignment error: embs={embs.shape[0]} mapping={len(mapping)}")

        # Save outputs
        np.save(self.index_dir / EMBEDDINGS_FILENAME, embs.astype("float32"))
        (self.index_dir / MAPPING_FILENAME).write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("Wrote %s and %s", EMBEDDINGS_FILENAME, MAPPING_FILENAME)

        # Write FAISS if available
        self._write_faiss_index(self.index_dir, embs)

        # Write meta.json
        conv_ids = {m.get("conv_id") for m in mapping if m.get("conv_id")}
        num_folders = len(conv_ids)
        meta = create_index_metadata(
            provider=os.getenv("EMBED_PROVIDER", "vertex"),
            num_documents=len(mapping),
            num_folders=num_folders,
            index_dir=self.index_dir,
            custom_metadata={"actual_dimensions": int(embs.shape[1])},
        )
        save_index_metadata(meta, self.index_dir)
        self.logger.info("Wrote meta.json")

        # Save file_times for incremental flows
        file_times = {m["id"]: float(m.get("modified_time", time.time())) for m in mapping if "id" in m}
        (self.index_dir / FILE_TIMES_FILENAME).write_text(json.dumps(file_times, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.index_dir / LAST_RUN_FILENAME).write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), encoding="utf-8")

        # Optional cleanup
        if remove_batches:
            removed = 0
            for p in pkl_files:
                try:
                    p.unlink()
                    removed += 1
                except Exception:
                    pass
            self.logger.info("Removed %d intermediate batch files.", removed)

    def fix_failed_embeddings(self, provider: str = "vertex") -> int:
        """Fix chunks with zero vectors from failed batches"""
        emb_dir = self.index_dir / "embeddings"
        pickle_files = list(emb_dir.glob('*.pkl'))
        
        self.logger.info(f"Scanning {len(pickle_files)} pickle files for zero vectors...")
        
        total_fixed = 0
        
        for pkl_file in pickle_files:
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                embs = np.array(data['embeddings'], dtype='float32')
                chunks = data['chunks']
                
                # Find rows that are all zeros
                zero_mask = np.all(embs == 0, axis=1)
                num_zeros = int(np.sum(zero_mask))
                
                if num_zeros > 0:
                    self.logger.info(f"Found {num_zeros} zero vectors in {pkl_file.name}")
                    
                    # Extract texts for failed chunks
                    failed_indices = np.where(zero_mask)[0]
                    texts_to_embed = [chunks[i]['text'] for i in failed_indices]
                    
                    self.logger.info(f"Re-embedding {len(texts_to_embed)} chunks...")
                    
                    try:
                        new_embeddings = embed_texts(texts_to_embed, provider=provider)
                        new_embs_array = np.array(new_embeddings, dtype='float32')
                        
                        # Replace zero vectors with new embeddings
                        for idx, failed_idx in enumerate(failed_indices):
                            embs[failed_idx] = new_embs_array[idx]
                        
                        # Save updated pickle
                        data['embeddings'] = embs
                        with open(pkl_file, 'wb') as f:
                            pickle.dump(data, f)
                        
                        self.logger.info(f"✅ Successfully re-embedded {num_zeros} chunks")
                        total_fixed += num_zeros
                        
                    except Exception as e:
                        self.logger.error(f"❌ Failed to re-embed: {e}")
                        
            except Exception as e:
                self.logger.error(f"Error processing {pkl_file.name}: {e}")
        
        self.logger.info(f"SUMMARY: Fixed {total_fixed} zero vectors")
        return total_fixed

    # ---- Helper methods ----

    def _coerce_batch(self, obj: Any) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Coerce various pickle shapes to standard format"""
        if isinstance(obj, dict):
            if "chunks" in obj and "embeddings" in obj:
                chunks = list(obj["chunks"])
                embs = np.asarray(obj["embeddings"], dtype="float32")
                n = min(len(chunks), embs.shape[0])
                return chunks[:n], embs[:n]
            if "mapping" in obj and "embeddings" in obj:
                chunks = list(obj["mapping"])
                embs = np.asarray(obj["embeddings"], dtype="float32")
                n = min(len(chunks), embs.shape[0])
                return chunks[:n], embs[:n]
        if isinstance(obj, (tuple, list)) and len(obj) == 2:
            a, b = obj
            arr = np.asarray(a, dtype="float32") if hasattr(a, "shape") else np.asarray(b, dtype="float32")
            chunks = a if isinstance(a, list) else (b if isinstance(b, list) else [])
            n = min(len(chunks), arr.shape[0] if arr.ndim == 2 else 0)
            return chunks[:n], arr[:n]
        return [], np.zeros((0, 1), dtype="float32")

    def _dedupe_and_stabilize(self, pairs: List[Tuple[Dict[str, Any], np.ndarray]]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Deduplicate and sort chunks"""
        best: Dict[str, Tuple[Dict[str, Any], np.ndarray]] = {}
        for d, e in pairs:
            did = str(d.get("id", "")).strip()
            if not did:
                continue
            cur = best.get(did)
            if cur is None:
                best[did] = (d, e)
            else:
                prev_d, _ = cur
                prev_ts = float(prev_d.get("modified_time", 0.0) or 0.0)
                new_ts = float(d.get("modified_time", 0.0) or 0.0)
                if new_ts >= prev_ts:
                    best[did] = (d, e)

        def _key(d: Dict[str, Any]) -> Tuple:
            return (
                str(d.get("conv_id", "")),
                str(d.get("doc_type", "")),
                str(d.get("id", "")),
                int(d.get("chunk_index", -1)) if isinstance(d.get("chunk_index"), int) else -1,
            )

        items = sorted(best.values(), key=lambda de: _key(de[0]))
        mapping = [de[0] for de in items]
        embs = np.vstack([de[1] for de in items]) if items else np.zeros((0, 1), dtype="float32")
        return mapping, embs

    def _ensure_snippet(self, mapping: List[Dict[str, Any]], limit: int = 500) -> None:
        """Ensure snippets exist and remove heavy text"""
        for m in mapping:
            txt = str(m.get("text", "") or "")
            if "snippet" not in m or not m.get("snippet"):
                m["snippet"] = txt[:limit]
            if "text" in m:
                try:
                    del m["text"]
                except Exception:
                    pass

    def _write_faiss_index(self, out_dir: Path, embs: np.ndarray) -> None:
        """Write FAISS index if available"""
        try:
            import faiss
            dim = int(embs.shape[1])
            index = faiss.IndexFlatIP(dim)
            index.add(np.ascontiguousarray(embs.astype("float32")))
            faiss.write_index(index, str(out_dir / FAISS_INDEX_FILENAME))
            self.logger.info("Wrote %s (ntotal=%d, dim=%d)", FAISS_INDEX_FILENAME, int(index.ntotal), dim)
        except Exception:
            self.logger.info("faiss not installed; skipping FAISS index.")

    # ---- Finalization ----

    def finalize_index(self) -> None:
        """Merge all worker outputs into final index"""
        self.logger.info("Finalizing index...")

        emb_dir = self.index_dir / "embeddings"
        merged_chunks: List[Dict[str, Any]] = []
        merged_embs: List[np.ndarray] = []

        pickle_files = sorted(emb_dir.glob("worker_*_batch_*.pkl"))
        if not pickle_files:
            self.logger.warning("No embedding pickle files found to merge.")
            return

        for pkl in pickle_files:
            try:
                with open(pkl, "rb") as f:
                    obj = pickle.load(f)
                merged_chunks.extend(obj["chunks"])
                merged_embs.append(np.asarray(obj["embeddings"], dtype="float32"))
            except Exception as e:
                self.logger.error("Failed to load pickle file %s: %s", pkl, e)
                continue

        if not merged_embs:
            self.logger.error("No embeddings were successfully loaded.")
            return

        embs = np.vstack(merged_embs)

        # Build rich mapping
        mapping: List[Dict[str, Any]] = []
        now_ts = time.time()
        for i, ch in enumerate(merged_chunks):
            if not isinstance(ch, dict):
                mapping.append({
                    "id": f"item::{i}",
                    "conv_id": "",
                    "doc_type": "conversation",
                    "path": "",
                    "subject": "",
                    "date": None,
                    "snippet": str(ch)[:500],
                    "modified_time": now_ts,
                })
                continue

            cid = ch.get("id") or f"item::{i}"
            path_str = ch.get("path", "")
            snippet = (ch.get("text", "") or "")[:500]
            doc_type = ch.get("doc_type") or ("attachment" if "::att" in str(cid) else "conversation")
            conv_id = ch.get("conv_id") or str(cid).split("::", 1)[0]

            mapping.append({
                "id": cid,
                "conv_id": conv_id,
                "doc_type": doc_type,
                "path": path_str,
                "subject": ch.get("subject", ""),
                "date": ch.get("date"),
                "snippet": snippet,
                "modified_time": ch.get("modified_time", now_ts),
                "from_email": ch.get("from_email", ""),
                "from_name": ch.get("from_name", ""),
                "to_recipients": ch.get("to_recipients", []),
                "cc_recipients": ch.get("cc_recipients", []),
            })

        # Ensure alignment
        if embs.shape[0] != len(mapping):
            n = min(embs.shape[0], len(mapping))
            embs = embs[:n]
            mapping = mapping[:n]

        # Save outputs
        np.save(self.index_dir / "embeddings.npy", embs.astype("float32"))
        (self.index_dir / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

        # Try to build FAISS index
        try:
            import faiss
            index = faiss.IndexFlatIP(embs.shape[1])
            vectors_to_add = np.ascontiguousarray(embs).astype("float32")
            index.add(vectors_to_add)
            faiss.write_index(index, str(self.index_dir / "index.faiss"))
        except Exception as e:
            self.logger.warning("Could not create FAISS index: %s", e)

        # Write metadata
        conv_ids = {m.get("conv_id") for m in mapping if m.get("conv_id")}
        meta = create_index_metadata(
            "vertex",
            num_documents=len(mapping),
            num_folders=len(conv_ids),
            index_dir=self.index_dir,
            custom_metadata={"actual_dimensions": int(embs.shape[1])},
        )
        save_index_metadata(meta, self.index_dir)
        self.logger.info("[OK] Index finalization complete.")

        # Cleanup intermediate files
        removed = 0
        for pkl in pickle_files:
            try:
                pkl.unlink()
                removed += 1
            except:
                pass
        if removed:
            self.logger.info("Removed %d intermediate embedding batches.", removed)


# -----------------------
# CLI
# -----------------------
def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Unified embedding processor for vertex AI operations"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Embed command (replaces vertex_indexer.py)
    embed_parser = subparsers.add_parser("embed", help="Create embeddings from chunks")
    embed_parser.add_argument("--root", required=True, help="Export root directory")
    embed_parser.add_argument("--mode", choices=["parallel", "sequential"], default="parallel")
    embed_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    embed_parser.add_argument("--resume", action="store_true", default=True)
    embed_parser.add_argument("--no-resume", dest="resume", action="store_false")
    embed_parser.add_argument("--test-mode", action="store_true")
    embed_parser.add_argument("--test-chunks", type=int, default=100)
    embed_parser.add_argument("--chunked-files", action="store_true")
    
    # Repair command (replaces repair_vertex_parallel_index.py)
    repair_parser = subparsers.add_parser("repair", help="Repair/merge batch pickles")
    repair_parser.add_argument("--root", required=True, help="Export root directory")
    repair_parser.add_argument("--remove-batches", action="store_true", help="Delete batches after merge")
    
    # Fix command (replaces fix_failed_embeddings.py)
    fix_parser = subparsers.add_parser("fix", help="Fix failed embeddings with zero vectors")
    fix_parser.add_argument("--root", required=True, help="Export root directory")
    fix_parser.add_argument("--provider", default="vertex", help="Embedding provider")
    
    # Finalize command (replaces run_vertex_finalize.py)
    finalize_parser = subparsers.add_parser("finalize", help="Finalize parallel index")
    finalize_parser.add_argument("--root", required=True, help="Export root directory")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    # Create processor
    if args.command in ["embed", "repair", "fix", "finalize"]:
        processor = EmbeddingProcessor(
            export_dir=args.root,
            mode=getattr(args, "mode", "parallel"),
            resume=getattr(args, "resume", True),
            batch_size=getattr(args, "batch_size", 64),
            test_mode=getattr(args, "test_mode", False),
            test_chunks=getattr(args, "test_chunks", 100),
            use_chunked_files=getattr(args, "chunked_files", False),
        )
        
        if args.command == "repair":
            processor.repair_index(remove_batches=args.remove_batches)
        elif args.command == "fix":
            total = processor.fix_failed_embeddings(provider=args.provider)
            return 0 if total > 0 else 1
        elif args.command == "finalize":
            processor.finalize_index()
        else:  # embed
            # Note: Full embedding workflow would need more implementation
            # This is a simplified version
            processor.logger.info("Embedding workflow not fully implemented in consolidated version")
            processor.logger.info("Use original vertex_indexer.py for full functionality")
    
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())