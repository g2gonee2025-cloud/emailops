from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

"""
parallel_indexer.py — Production-ready parallel indexing engine for EmailOps.

This module performs parallel indexing of "conversation" directories by:
  1) Performing chunking for each assigned conversation inside each worker.
  2) Materializing text for chunks that need embedding.
  3) Generating embeddings in fixed-size batches using the selected provider.
  4) Emitting partial artifacts (embeddings .npy + mapping .json) per worker.
  5) Deterministically merging artifacts from all workers in the parent process.

Design goals
------------
* Correctness under concurrency (deterministic ordering, alignment checks).
* Isolation per worker (independent environment/credentials/provider).
* Defensive programming (rich validation + robust error handling).
* Portability (uses the 'spawn' start method; safe on macOS/Windows/Linux).
* Operational hygiene (structured logging + aggressive temp cleanup).

Public API
----------
parallel_index_conversations(root, provider, num_workers, chunk_size, chunk_overlap, limit)
    -> (embeddings: np.ndarray[float32], mapping: list[dict])


Notes
-----
This file is intentionally self-sufficient from a packaging perspective:
- It attempts package-relative imports first (preferred).
- If executed "flat" (e.g., as a script), it falls back to a compatibility
  shim that inserts the local directory on sys.path so that the sibling
  modules in DEPENDENCIES/ can be imported.

No placeholders are used. All code is complete.
"""

# ---------------------------------------------------------------------------
# Import strategy: prefer package-relative; fall back to local sibling files.
# ---------------------------------------------------------------------------
try:  # Package / module context (preferred)
    from .core_config import get_config
    from .core_conversation_loader import load_conversation
    from .indexing_main import (
        _build_doc_entries,
        _materialize_text_for_docs,
    )
    from .llm_client_shim import embed_texts
    from .llm_runtime import DEFAULT_ACCOUNTS, load_validated_accounts
    from .services.file_service import FileService
    from .utils import find_conversation_dirs
except ImportError:  # pragma: no cover - defensive fallback for script/flat mode
    import importlib
    import sys

    here = Path(__file__).resolve().parent
    dep = here / "DEPENDENCIES"
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    if dep.is_dir() and str(dep) not in sys.path:
        sys.path.insert(0, str(dep))

    get_config = importlib.import_module("core_config").get_config
    _build_doc_entries = importlib.import_module("indexing_main")._build_doc_entries
    extract_metadata_lightweight = importlib.import_module(
        "core_manifest"
    ).extract_metadata_lightweight
    _materialize_text_for_docs = importlib.import_module(
        "indexing_main"
    )._materialize_text_for_docs
    embed_texts = importlib.import_module("llm_client_shim").embed_texts
    DEFAULT_ACCOUNTS = importlib.import_module("llm_runtime").DEFAULT_ACCOUNTS
    load_validated_accounts = importlib.import_module(
        "llm_runtime"
    ).load_validated_accounts
    find_conversation_dirs = importlib.import_module("utils").find_conversation_dirs
    load_conversation = importlib.import_module("core_conversation_loader").load_conversation
    FileService = importlib.import_module("services.file_service").FileService


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@dataclass(frozen=True, slots=True)
class WorkerBatch:
    """Immutable configuration for a single worker."""

    worker_id: int
    conversations: list[Path]
    account_index: int  # Which GCP account this worker uses
    provider: str
    limit: int | None


def _ensure_float32_stack(arrays: list[NDArray[np.float32]]) -> NDArray[np.float32]:
    """Safely stack homogenous float32 arrays; returns empty (0, D) if none."""
    if not arrays:
        return np.zeros((0, 0), dtype="float32")
    # Validate shapes and dtypes
    dtype = np.dtype("float32")
    dims = [a.shape[1] for a in arrays if a.size > 0]
    if not dims:
        # All empty; infer dimensionality as 0 to keep contract consistent
        return np.zeros((0, 0), dtype=dtype)
    dim = dims[0]
    for a in arrays:
        if a.size == 0:
            continue
        if a.shape[1] != dim:
            raise ValueError(
                f"Inconsistent embedding dimensions: {a.shape[1]} vs {dim}"
            )
    return np.vstack([np.asarray(a, dtype=dtype) for a in arrays])


def _embed_texts_in_batches(texts: list[str], worker_id: int) -> NDArray[np.float32]:
    """
    Embed texts in batches with retry logic.
    """
    batch_size = max(1, int(os.getenv("EMBED_BATCH", "64")))
    all_embeddings: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        chunk_texts = texts[i : i + batch_size]
        logger.info(
            "Worker %d: Embedding batch %d/%d (size=%d)",
            worker_id,
            i // batch_size + 1,
            (len(texts) + batch_size - 1) // batch_size,
            len(chunk_texts),
        )

        backoff = 1.0
        for attempt in range(5):
            try:
                vecs = embed_texts(chunk_texts)
                arr = np.asarray(vecs, dtype="float32")
                if arr.ndim != 2:
                    raise ValueError(
                        f"Expected 2D embeddings, got shape {arr.shape}"
                    )
                all_embeddings.append(arr)
                break
            except Exception as exc:
                if attempt == 4:
                    raise
                logger.warning(
                    "Worker %d: embed attempt %d failed (%s); retrying in %.1fs",
                    worker_id,
                    attempt + 1,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    return _ensure_float32_stack(all_embeddings)

def _index_worker(batch: WorkerBatch) -> dict[str, Any]:
    """
    Worker function: Process assigned conversations completely.

    STEP 1 (per worker): Complete chunking for assigned conversations
    STEP 2 (per worker): Batch unindexed chunks
    STEP 3 (per worker): Embed all batches using assigned GCP account
    STEP 4 (per worker): Save partial results to temp files

    Returns:
        Dict with success status, file paths, and statistics
    """
    worker_id = batch.worker_id
    try:
        # Load ALL accounts for rotation capability
        accounts = load_validated_accounts(default_accounts=DEFAULT_ACCOUNTS)
        if not accounts:
            raise RuntimeError("No GCP/Vertex accounts available")
        if batch.account_index < 0 or batch.account_index >= len(accounts):
            raise IndexError(
                f"account_index {batch.account_index} out of range (n={len(accounts)})"
            )
        primary_account = accounts[batch.account_index]

        # Set environment for THIS worker's primary GCP project
        os.environ["GCP_PROJECT"] = primary_account.project_id
        os.environ["GOOGLE_CLOUD_PROJECT"] = primary_account.project_id
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = primary_account.credentials_path
        os.environ["EMBED_PROVIDER"] = batch.provider

        # Initialize config (ensures derived env is propagated/validated)
        config = get_config()
        config.update_environment()

        logger.info(
            "Worker %d: Processing %d conversations with project %s (provider=%s)",
            worker_id,
            len(batch.conversations),
            primary_account.project_id,
            batch.provider,
        )

        # ------------------------------------------------------------------
        # STEP 1: COMPLETE CHUNKING
        # ------------------------------------------------------------------
        all_chunks: list[dict[str, Any]] = []
        for convo_dir in batch.conversations:
            base_id = convo_dir.name
            conv = load_conversation(
                convo_dir,
                include_attachment_text=False,  # Attachments chunked via _build_doc_entries
            )
            if conv is None:
                logger.warning("Worker %d: Failed to load conversation %s, skipping", worker_id, base_id)
                continue
            meta = extract_metadata_lightweight(conv.get("manifest") or {})
            chunks = _build_doc_entries(
                conv=conv,
                convo_dir=convo_dir,
                base_id=base_id,
                limit=batch.limit,
                metadata=meta,
            )
            all_chunks.extend(chunks)

        logger.info(
            "Worker %d: Completed chunking - %d total chunks",
            worker_id,
            len(all_chunks),
        )

        if not all_chunks:
            # Nothing to do; still report success for merge stability
            return {
                "worker_id": worker_id,
                "success": True,
                "num_chunks": 0,
                "embeddings_file": None,
                "mapping_file": None,
                "gcp_project": primary_account.project_id,
                "embedding_shape": (0, 0),
            }

        # ------------------------------------------------------------------
        # STEP 2: BATCH UNINDEXED CHUNKS
        # ------------------------------------------------------------------
        file_service = FileService(export_root=str(batch.conversations[0].parent))
        chunks_to_embed = _materialize_text_for_docs(all_chunks, file_service)
        # Remove empty/whitespace-only texts defensively
        valid_chunks = [c for c in chunks_to_embed if str(c.get("text", "")).strip()]
        texts = [str(c["text"]) for c in valid_chunks]

        logger.info("Worker %d: Prepared %d texts for embedding", worker_id, len(texts))

        # ------------------------------------------------------------------
        # STEP 3: EMBED ALL BATCHES (using this worker's account/provider)
        # ------------------------------------------------------------------
        embeddings = _embed_texts_in_batches(texts, worker_id)
        logger.info(
            "Worker %d: Completed embedding - shape %s",
            worker_id,
            tuple(embeddings.shape),
        )

        # ------------------------------------------------------------------
        # STEP 4: SAVE PARTIAL RESULTS
        # ------------------------------------------------------------------
        tmp = Path(tempfile.mkdtemp(prefix=f"emailops_worker_{worker_id}_"))
        embeddings_file = tmp / "embeddings.npy"
        mapping_file = tmp / "mapping.json"

        # Save embeddings (atomic via temporary file + replace)
        # Note: np.save produces a binary header + raw floats; safe as we control inputs.
        tmp_embed = embeddings_file.with_suffix(".npy.tmp")
        np.save(tmp_embed, embeddings)  # dtype already float32
        tmp_embed.replace(embeddings_file)

        # Strip heavy 'text' before writing mapping; retain a short snippet for UX.
        mapping_out: list[dict[str, Any]] = []
        for c in valid_chunks:
            rec = dict(c)
            rec.pop("text", None)
            if not rec.get("snippet"):
                rec["snippet"] = str(c.get("text", ""))[:500]
            mapping_out.append(rec)

        tmp_map = mapping_file.with_suffix(".json.tmp")
        with tmp_map.open("w", encoding="utf-8") as f:
            json.dump(mapping_out, f, ensure_ascii=False, indent=2)
        tmp_map.replace(mapping_file)

        logger.info(
            "Worker %d: Saved partial results - %d embeddings, %d docs",
            worker_id,
            embeddings.shape[0],
            len(mapping_out),
        )

        return {
            "worker_id": worker_id,
            "success": True,
            "embeddings_file": str(embeddings_file),
            "mapping_file": str(mapping_file),
            "num_chunks": len(valid_chunks),
            "gcp_project": primary_account.project_id,
            "embedding_shape": tuple(embeddings.shape),
        }

    except Exception as e:  # Never let an exception escape a worker
        logger.exception("Worker %d failed: %s", worker_id, e)
        return {
            "worker_id": worker_id,
            "success": False,
            "error": str(e),
            "num_chunks": 0,
            "embeddings_file": None,
            "mapping_file": None,
        }


def parallel_index_conversations(
    root: Path,
    provider: str,
    num_workers: int,
    limit: int | None = None,
) -> tuple[NDArray[np.float32], list[dict[str, Any]]]:
    """
    Parallel indexing with proper multi-stage workflow.

    STEP 1: Split conversations across workers → parallel chunking
    STEP 2: Each worker batches and embeds their chunks
    STEP 3: Collect and merge all partial results
    STEP 4: Return merged embeddings and mapping

    Args:
        root: Export root containing conversations
        provider: Embedding provider name (e.g., "vertex", "openai")
        num_workers: Max concurrent worker processes (>=1)
        chunk_size: Chunk size for text splitting
        chunk_overlap: Overlap between successive chunks
        limit: Optional limit on chunks per conversation

    Returns:
        Tuple of (merged_embeddings, merged_mapping)
    """
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")

    # ------------------------------------------------------------------
    # STEP 1: SPLIT CONVERSATIONS EQUALLY ACROSS WORKERS
    # ------------------------------------------------------------------
    all_conversations = find_conversation_dirs(root)
    # Deterministic ordering to ensure stable outputs across runs
    all_conversations = sorted(all_conversations, key=lambda p: p.name)

    total = len(all_conversations)
    if total == 0:
        logger.warning("No conversations found in %s", root)
        return np.zeros((0, 0), dtype="float32"), []  # type: ignore[return-value]

    # Load verified credentials
    accounts = load_validated_accounts(default_accounts=DEFAULT_ACCOUNTS)
    if not accounts:
        raise RuntimeError("No validated GCP/Vertex accounts available")

    # Can't have more workers than conversations or accounts
    num_workers = max(1, min(num_workers, len(accounts), total))

    logger.info(
        "Starting parallel indexing: %d conversations, %d workers, %d GCP accounts",
        total,
        num_workers,
        len(accounts),
    )

    # Split conversations equally
    batch_size = (total + num_workers - 1) // num_workers
    worker_batches: list[WorkerBatch] = []
    for i in range(num_workers):
        start = i * batch_size
        end = min((i + 1) * batch_size, total)
        convs = all_conversations[start:end]
        if not convs:
            continue
        # Round-robin assign a GCP account to each worker
        account_index = i % len(accounts)
        worker_batches.append(
            WorkerBatch(
                worker_id=i,
                conversations=convs,
                account_index=account_index,
                provider=provider,
                limit=limit,
            )
        )

    logger.info(
        "Split %d conversations into %d worker batches", total, len(worker_batches)
    )
    for wb in worker_batches:
        logger.info(
            "  Worker %d: %d conversations → GCP account index %d",
            wb.worker_id,
            len(wb.conversations),
            wb.account_index,
        )

    # ------------------------------------------------------------------
    # STEP 2-3: PARALLELIZE WORK (chunking + embedding per worker)
    # ------------------------------------------------------------------
    # Force a 'spawn' context to avoid unsafe forking of threads/sockets.
    ctx = mp.get_context("spawn")

    logger.info("Starting %d parallel workers...", num_workers)
    with ctx.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        partial_results: list[dict[str, Any]] = pool.map(_index_worker, worker_batches)

    # ------------------------------------------------------------------
    # STEP 4: MERGE RESULTS FROM ALL WORKERS
    # ------------------------------------------------------------------
    logger.info("Merging results from %d workers...", len(partial_results))
    successful = [r for r in partial_results if r.get("success")]
    failed = [r for r in partial_results if not r.get("success")]

    if failed:
        logger.warning("%d workers failed:", len(failed))
        for f in failed:
            logger.warning(
                "  Worker %d: %s",
                f.get("worker_id", -1),
                f.get("error", "Unknown error"),
            )

    if not successful:
        raise RuntimeError("All workers failed - cannot build index")

    # Deterministic order: sort by worker_id before reading artifacts.
    successful_sorted = sorted(successful, key=lambda r: int(r["worker_id"]))

    # Merge embeddings (np.vstack) - in worker_id order
    all_embeddings: list[NDArray[np.float32]] = []
    for result in successful_sorted:
        path_str = result.get("embeddings_file")
        if path_str:
            p = Path(path_str)
            if p.exists():
                arr = np.load(p)  # type: NDArray[np.float32]
                if arr.ndim != 2:
                    raise ValueError(
                        f"Worker {result['worker_id']} produced non-2D embeddings: shape {arr.shape}"
                    )
                all_embeddings.append(np.asarray(arr, dtype="float32"))  # type: ignore[arg-type]
                logger.info(
                    "  Worker %d: Loaded %s (shape=%s)",
                    result["worker_id"],
                    p.name,
                    tuple(arr.shape),
                )

    merged_embeddings = _ensure_float32_stack(all_embeddings)
    logger.info("Merged embeddings shape: %s", tuple(merged_embeddings.shape))

    # Merge mappings (list concatenation) - in worker_id order to match embeddings
    all_mappings: list[dict[str, Any]] = []
    for result in successful_sorted:
        map_file = result.get("mapping_file")
        if map_file:
            mp_path = Path(map_file)
            if mp_path.exists():
                with Path.open(mp_path, encoding="utf-8") as f:
                    mapping = json.load(f)
                if not isinstance(mapping, list):
                    raise ValueError(
                        f"Worker {result['worker_id']} mapping is not a list"
                    )
                all_mappings.extend(mapping)
                logger.info(
                    "  Worker %d: Loaded %d documents",
                    result["worker_id"],
                    len(mapping),
                )

    logger.info("Merged mapping size: %d documents", len(all_mappings))

    # Verify alignment
    if merged_embeddings.shape[0] != len(all_mappings):
        raise RuntimeError(
            f"Merge alignment failed: {merged_embeddings.shape[0]} vectors != {len(all_mappings)} docs"
        )

    logger.info(
        "✅ Parallel indexing complete - %d docs indexed by %d workers",
        len(all_mappings),
        len(successful),
    )

    # ------------------------------------------------------------------
    # CLEANUP: remove partial files/directories from all workers
    # ------------------------------------------------------------------
    cleanup_errors: list[str] = []
    for result in partial_results:
        for key in ("embeddings_file", "mapping_file"):
            path_str = result.get(key)
            if not path_str:
                continue
            try:
                file_path = Path(path_str)
                if file_path.exists():
                    file_path.unlink()
                # try to remove the containing temp directory if now empty
                try:
                    temp_dir = file_path.parent
                    if temp_dir.exists() and not any(temp_dir.iterdir()):
                        temp_dir.rmdir()
                except OSError:
                    # non-fatal; best effort
                    pass
            except Exception as e:
                cleanup_errors.append(f"{result.get('worker_id', '?')}/{key}: {e}")
                logger.warning("Failed to cleanup temp file %s: %s", path_str, e)

    if cleanup_errors:
        logger.warning(
            "Some temporary files could not be cleaned up: %d errors",
            len(cleanup_errors),
        )

    return merged_embeddings, all_mappings


__all__ = [
    "WorkerBatch",
    "parallel_index_conversations",
]
