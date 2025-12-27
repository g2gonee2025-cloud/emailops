from __future__ import annotations

import bisect
import json
import logging
import multiprocessing as mp
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core_config import get_config
from .utils import find_conversation_dirs

logger = logging.getLogger(__name__)

__all__ = (
    "ChunkWorkerBatch",
    "force_rechunk_all",
    "incremental_rechunk",
    "parallel_chunk_conversations",
    "prepare_index_units",
    "surgical_rechunk",
]


def prepare_index_units(
    text: str,
    doc_id: str,
    doc_path: str,
    subject: str = "",
    date: str | None = None,
) -> list[dict[str, Any]]:
    """
    Prepare text for indexing by splitting it into chunks with metadata.

    Used by the indexing pipeline to create indexable units from conversation text
    and attachments. Chunk IDs follow the required convention:
      - first chunk: `doc_id`
      - subsequent: `doc_id::chunk{N}` (N starts at 1)

    Returns a list of dicts containing:
        - id, text, path, subject, start_char, end_char, optional date
        - chunk_index (0-based, convenient for debugging)
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    if not text.strip():
        return []
    if not isinstance(doc_id, str) or not doc_id:
        raise ValueError("doc_id must be a non-empty string")
    if not isinstance(doc_path, str) or not doc_path:
        raise ValueError("doc_path must be a non-empty string")

    cfg = get_config()
    eff_chunk_size = cfg.processing.chunk_size
    eff_chunk_overlap = cfg.processing.chunk_overlap
    eff_min_chunk_size = 50  # or get from config if available
    eff_respect_sentences = True
    eff_respect_paragraphs = True
    eff_progressive_scaling = True
    eff_max_chunks = None

    eff_chunk_size, eff_chunk_overlap = _apply_progressive_scaling(
        total_len=len(text),
        size=eff_chunk_size,
        overlap=eff_chunk_overlap,
        enable=eff_progressive_scaling,
    )

    ranges = _ranges_with_overlap(
        text=text,
        chunk_size=eff_chunk_size,
        chunk_overlap=eff_chunk_overlap,
        min_chunk_size=eff_min_chunk_size,
        max_chunks=eff_max_chunks,
        respect_sentences=eff_respect_sentences,
        respect_paragraphs=eff_respect_paragraphs,
    )

    chunks: list[dict[str, Any]] = []
    for idx, (start, end) in enumerate(ranges):
        chunk_text = text[start:end]
        chunk: dict[str, Any] = {
            "id": f"{doc_id}::chunk{idx}" if idx > 0 else doc_id,
            "text": chunk_text,
            "path": doc_path,
            "subject": subject,
            "chunk_index": idx,
            "start_char": start,
            "end_char": end,  # exclusive
        }
        if date:
            chunk["date"] = date
        chunks.append(chunk)

    return chunks


# -----------------------------------------------------------------------------
# Precompiled patterns for boundary-aware chunking
# -----------------------------------------------------------------------------
# Paragraph boundary: one or more blank lines (support both LF and CRLF)
PARA_RE = re.compile(r"(?:\r?\n)\s*(?:\r?\n)+")

# Sentence boundary:
# - Western: . ! ?
# - East Asian: 。 ！ ？
# - Ellipsis: ...
# - Arabic/Persian/Urdu: ؟ ۔
# Allow common closing quotes/brackets after the punctuation, then whitespace or EoD.
# MEDIUM #19: Known edge cases (Dr. Smith, 3.14159) are trade-offs for performance.
# Overly complex patterns would significantly slow chunking for minimal accuracy gain.
SENT_RE = re.compile(r"[.!?。！？...؟۔]+" r'[)"\'\u2018\u2019»›）】〔〕〉》」』〗〙〞]*' r"(?:\s+|$)")


def _apply_progressive_scaling(
    total_len: int, size: int, overlap: int, enable: bool
) -> tuple[int, int]:
    """
    Scale chunk size for very large texts to reduce chunk counts while keeping the
    overlap ratio approximately constant.

    Estimated chunk count bands:
      ≤50  → x1
      51-150 → x1.5
      151-500 → x2
      500+ → x3
    """
    if not enable:
        size = max(1, int(size))
        return size, max(0, min(int(overlap), size - 1))

    effective_chunk_size = max(1, int(size) - int(overlap))
    estimated_chunks = max(
        1, (int(total_len) + effective_chunk_size - 1) // effective_chunk_size
    )

    if estimated_chunks <= 50:
        factor = 1.0
    elif estimated_chunks <= 150:
        factor = 1.5
    elif estimated_chunks <= 500:
        factor = 2.0
    else:
        factor = 3.0

    new_size = max(1, int(size * factor))
    new_overlap = int(overlap * (new_size / max(1, int(size))))
    new_overlap = max(0, min(new_overlap, new_size - 1))
    return new_size, new_overlap


def _compute_breakpoints(
    text: str, respect_sentences: bool, respect_paragraphs: bool
) -> list[int]:
    """
    Pre-compute candidate split positions that align with paragraph or sentence boundaries.
    Positions are indices where a chunk may end (exclusive).
    """
    breaks: set[int] = set()

    if respect_paragraphs:
        for m in PARA_RE.finditer(text):
            breaks.add(m.end())

    if respect_sentences:
        for m in SENT_RE.finditer(text):
            breaks.add(m.end())

    breaks.add(len(text))  # Always allow end-of-document
    return sorted(breaks)


def _ranges_with_overlap(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    max_chunks: int | None,
    respect_sentences: bool,
    respect_paragraphs: bool,
) -> list[tuple[int, int]]:
    """
    Produce (start, end) ranges over `text` honoring overlap and optional boundary-aware cutting.

    Guarantees forward progress even if chunk_overlap >= chunk_size.
    Ensures the final chunk is not smaller than `min_chunk_size` (if possible) by merging it.
    When `max_chunks` is set, the final produced chunk captures the remainder of the document.
    """
    n = len(text)
    if n == 0:
        return []

    # Clamp parameters to safe values
    chunk_size = max(1, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))
    min_chunk_size = max(0, min(int(min_chunk_size), chunk_size))
    if max_chunks is not None and max_chunks < 1:
        max_chunks = 1

    boundary_points: list[int] | None = (
        _compute_breakpoints(text, respect_sentences, respect_paragraphs)
        if (respect_sentences or respect_paragraphs)
        else None
    )

    ranges: list[tuple[int, int]] = []
    start = 0
    lookahead_limit = max(
        1, int(0.2 * chunk_size)
    )  # allow slight overshoot to land on a boundary

    while start < n and (max_chunks is None or len(ranges) < max_chunks):
        ideal_end = min(n, start + chunk_size)

        if boundary_points is not None:
            i = bisect.bisect_right(boundary_points, ideal_end)
            candidate_before = boundary_points[i - 1] if i > 0 else None
            candidate_after = boundary_points[i] if i < len(boundary_points) else None

            end = ideal_end
            if candidate_before is not None and candidate_before >= start + max(
                1, min_chunk_size
            ):
                end = candidate_before
            elif (
                candidate_after is not None
                and candidate_after - ideal_end <= lookahead_limit
            ):
                end = candidate_after
        else:
            end = ideal_end

        if end <= start:
            end = min(n, start + chunk_size)

        if max_chunks is not None and len(ranges) + 1 >= max_chunks:
            end = n  # capture tail

        ranges.append((start, end))

        if end >= n:
            break

        actual_len = end - start
        step = max(1, actual_len - chunk_overlap)
        start += step

    # Merge a tiny tail, if present
    if len(ranges) >= 2:
        last_start, last_end = ranges[-1]
        if (last_end - last_start) < min_chunk_size:
            prev_start, prev_end = ranges[-2]
            ranges[-2] = (prev_start, last_end)
            ranges.pop()

    return ranges


# ============================================================================
# Parallel Chunking Infrastructure (NO embedding - just text chunking)
# ============================================================================


@dataclass(frozen=True, slots=True)
class ChunkWorkerBatch:
    """Immutable configuration for a single chunking worker (NO embedding)."""

    worker_id: int
    conversations: list[Path]
    output_dir: Path  # Where this worker writes chunk files


def _enrich_chunk(
    chunk: dict[str, Any],
    conv_id: str,
    doc_type: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Enrich a single chunk with conversation-level metadata.

    IMPORTANT:
      - Keep the full chunk `text` for embedding/indexing.
      - Add a short `snippet` field for UI/preview use (do not truncate `text`).
    """
    chunk_text = chunk.get("text", "")
    if not isinstance(chunk_text, str):
        chunk_text = str(chunk_text or "")

    # Preserve full text for embeddings
    chunk["text"] = chunk_text

    # Provide a short preview snippet (first 200 chars)
    chunk["snippet"] = chunk_text[:200] if len(chunk_text) > 200 else chunk_text

    chunk.update(
        {
            "conv_id": conv_id,
            "doc_type": doc_type,
            "subject": metadata.get("subject", ""),
            "from": metadata.get("from", []),
            "to": metadata.get("to", []),
            "cc": metadata.get("cc", []),
            "start_date": metadata.get("start_date"),
            "end_date": metadata.get("end_date"),
        }
    )
    return chunk


def _chunk_worker(batch: ChunkWorkerBatch) -> dict[str, Any]:
    """
    Worker function: CLEAN then CHUNK assigned conversations (NO embedding).

    Each worker:
    1. Loads conversation data (text + attachments)
    2. CLEANS text using clean_email_text() for indexing-ready output
    3. Chunks the cleaned content using prepare_index_units
    4. Writes chunk JSON file to output_dir/{conv_id}.json
    5. Returns statistics

    Returns:
        Dict with success status, statistics, and any errors
    """
    worker_id = batch.worker_id
    stats = {
        "worker_id": worker_id,
        "success": True,
        "conversations_processed": 0,
        "total_chunks": 0,
        "errors": [],
        "conversation_details": [],
    }

    try:
        # Import here to avoid issues with multiprocessing pickling
        from .core_conversation_loader import load_conversation
        from .core_email_processing import clean_email_text
        from .core_manifest import extract_metadata_lightweight

        logger.info(
            "Worker %d: Processing %d conversations",
            worker_id,
            len(batch.conversations),
        )

        for conv_dir in batch.conversations:
            conv_id = conv_dir.name
            try:
                # Load conversation with attachments
                conv_data = load_conversation(
                    conv_dir,
                    include_attachment_text=True,
                    max_total_attachment_text=None,  # Use config defaults
                )

                # STEP 1: CLEAN text first
                text_raw = conv_data.get("conversation_txt", "")
                text_to_chunk = clean_email_text(text_raw)

                if not text_to_chunk.strip():
                    logger.warning(
                        "Worker %d: Empty text after cleaning for %s, skipping",
                        worker_id,
                        conv_id,
                    )
                    continue

                # STEP 2: CHUNK the cleaned text (no metadata yet)
                chunks = prepare_index_units(
                    text=text_to_chunk,
                    doc_id=f"{conv_id}::conversation",
                    doc_path=str(conv_dir / "Conversation.txt"),
                )

                if not chunks:
                    logger.warning(
                        "Worker %d: No chunks created for %s", worker_id, conv_id
                    )
                    continue

                # STEP 3: EXTRACT metadata from manifest AFTER chunking succeeds
                manifest = conv_data.get("manifest") or {}
                metadata = extract_metadata_lightweight(manifest)

                # STEP 4: ADD complete metadata to each chunk
                enriched_chunks = [
                    _enrich_chunk(ch, conv_id, "conversation", metadata)
                    for ch in chunks
                ]

                if not enriched_chunks:
                    logger.warning(
                        "Worker %d: No chunks after enrichment for %s",
                        worker_id,
                        conv_id,
                    )
                    continue

                # Write chunk file atomically
                chunk_file = batch.output_dir / f"{conv_id}.json"
                tmp_file = chunk_file.with_suffix(".json.tmp")

                try:
                    with tmp_file.open("w", encoding="utf-8") as f:
                        json.dump(enriched_chunks, f, ensure_ascii=False, indent=2)
                    tmp_file.replace(chunk_file)
                except Exception:
                    if tmp_file.exists():
                        tmp_file.unlink()
                    raise

                stats["conversations_processed"] += 1
                stats["total_chunks"] += len(enriched_chunks)
                stats["conversation_details"].append(
                    {
                        "conv_id": conv_id,
                        "num_chunks": len(enriched_chunks),
                    }
                )

                logger.debug(
                    "Worker %d: Chunked %s -> %d chunks",
                    worker_id,
                    conv_id,
                    len(enriched_chunks),
                )

            except Exception as e:
                error_msg = f"{conv_id}: {e!s}"
                stats["errors"].append(error_msg)
                logger.error("Worker %d: Failed to chunk %s: %s", worker_id, conv_id, e)

        logger.info(
            "Worker %d: Completed - %d conversations, %d chunks, %d errors",
            worker_id,
            stats["conversations_processed"],
            stats["total_chunks"],
            len(stats["errors"]),
        )

    except Exception as e:
        # Worker-level failure
        logger.exception("Worker %d failed catastrophically: %s", worker_id, e)
        stats["success"] = False
        stats["errors"].append(f"Worker failure: {e!s}")

    return stats


def parallel_chunk_conversations(
    root: Path,
    output_dir: Path,
    conversation_filter: set[str] | None = None,
) -> dict[str, Any]:
    """
    Parallel conversation chunking (NO embedding - just text splitting).

    This function distributes conversations across workers for parallel chunking.
    Each worker independently processes its assigned conversations and writes
    chunk files to the output directory.

    Args:
        root: Export root containing conversation folders
        output_dir: Directory to write chunk JSON files
        conversation_filter: Optional set of conv_ids to process (for surgical mode)

    Returns:
        Dict with statistics: {total_chunks, total_conversations, worker_results, failed_workers}
    """
    # Dynamically load config to get the latest settings
    from .core_config import get_config

    cfg = get_config()
    num_workers = cfg.processing.num_workers

    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all conversations
    all_conversations = find_conversation_dirs(root)
    all_conversations = sorted(all_conversations, key=lambda p: p.name)

    # Apply filter if provided (for surgical mode)
    if conversation_filter:
        all_conversations = [
            c for c in all_conversations if c.name in conversation_filter
        ]

    total = len(all_conversations)
    if total == 0:
        logger.warning("No conversations found in %s", root)
        return {
            "total_chunks": 0,
            "total_conversations": 0,
            "worker_results": [],
            "failed_workers": [],
        }

    # Can't have more workers than conversations
    num_workers = max(1, min(num_workers, total))

    logger.info(
        "Starting parallel chunking: %d conversations, %d workers", total, num_workers
    )

    # Split conversations equally across workers
    batch_size = (total + num_workers - 1) // num_workers
    worker_batches: list[ChunkWorkerBatch] = []

    for i in range(num_workers):
        start = i * batch_size
        end = min((i + 1) * batch_size, total)
        convs = all_conversations[start:end]
        if not convs:
            continue

        worker_batches.append(
            ChunkWorkerBatch(
                worker_id=i,
                conversations=convs,
                output_dir=output_dir,
            )
        )

    logger.info(
        "Split %d conversations into %d worker batches", total, len(worker_batches)
    )
    for wb in worker_batches:
        logger.info(
            "  Worker %d: %d conversations", wb.worker_id, len(wb.conversations)
        )

    # Use spawn context for safety (cross-platform)
    ctx = mp.get_context("spawn")

    logger.info("Starting %d parallel workers...", len(worker_batches))
    with ctx.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        worker_results: list[dict[str, Any]] = pool.map(_chunk_worker, worker_batches)

    # Aggregate results
    successful = [r for r in worker_results if r.get("success")]
    failed = [r for r in worker_results if not r.get("success")]

    if failed:
        logger.warning("%d workers failed:", len(failed))
        for f in failed:
            logger.warning(
                "  Worker %d: %d errors",
                f.get("worker_id", -1),
                len(f.get("errors", [])),
            )

    total_chunks = sum(r.get("total_chunks", 0) for r in successful)
    total_processed = sum(r.get("conversations_processed", 0) for r in successful)

    logger.info(
        "✅ Parallel chunking complete - %d conversations, %d chunks by %d workers",
        total_processed,
        total_chunks,
        len(successful),
    )

    return {
        "total_chunks": total_chunks,
        "total_conversations": total_processed,
        "worker_results": successful,
        "failed_workers": failed,
    }


# ============================================================================
# High-Level Public APIs for GUI
# ============================================================================


def force_rechunk_all(
    root: Path,
) -> dict[str, Any]:
    """
    Force re-chunk ALL conversations (deletes existing chunks first).

    This operation:
    1. Deletes the entire _chunks directory
    2. Recreates it
    3. Chunks all conversations in parallel
    4. Writes chunk JSON files

    Args:
        root: Export root containing conversation folders

    Returns:
        Statistics dict with total_chunks, total_conversations, etc.
    """
    root = Path(root)
    chunks_dir = root / "_chunks"

    logger.info("Force rechunk: Clearing existing chunks directory")
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)

    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Chunk all conversations in parallel
    result = parallel_chunk_conversations(
        root=root,
        output_dir=chunks_dir,
        conversation_filter=None,  # Process all
    )

    logger.info(
        "Force rechunk complete: %d conversations, %d chunks",
        result["total_conversations"],
        result["total_chunks"],
    )

    return result


def incremental_rechunk(
    root: Path,
) -> dict[str, Any]:
    """
    Incrementally chunk only new/modified conversations.

    This operation:
    1. Reads last chunk timestamp from _chunks/_last_chunk.txt
    2. Identifies conversations modified since that timestamp
    3. Chunks only those conversations in parallel
    4. Updates the timestamp file

    Args:
        root: Export root containing conversation folders
    """
    root = Path(root)
    chunks_dir = root / "_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Read last chunk timestamp
    last_chunk_file = chunks_dir / "_last_chunk.txt"
    last_chunk_time = 0.0
    if last_chunk_file.exists():
        try:
            last_chunk_time = float(last_chunk_file.read_text())
        except (ValueError, OSError):
            logger.warning("Failed to read last chunk time, treating as full rechunk")
            last_chunk_time = 0.0

    # Find all conversations
    all_convs = find_conversation_dirs(root)

    # Filter to only modified conversations
    def needs_update(conv_dir: Path) -> bool:
        """Check if conversation needs chunking."""
        chunk_file = chunks_dir / f"{conv_dir.name}.json"
        if not chunk_file.exists():
            return True

        try:
            # Check Conversation.txt
            conv_txt_path = conv_dir / "Conversation.txt"
            if (
                conv_txt_path.exists()
                and conv_txt_path.stat().st_mtime > last_chunk_time
            ):
                return True

            # Check attachments
            attachments_dir = conv_dir / "Attachments"
            if attachments_dir.exists():
                for att_path in attachments_dir.rglob("*"):
                    if (
                        att_path.is_file()
                        and att_path.stat().st_mtime > last_chunk_time
                    ):
                        return True
        except OSError:
            # If we can't stat, assume it needs update
            return True

        return False

    to_chunk_ids = {d.name for d in all_convs if needs_update(d)}

    logger.info(
        "Incremental rechunk: %d of %d conversations need updating",
        len(to_chunk_ids),
        len(all_convs),
    )

    if not to_chunk_ids:
        logger.info("No conversations need chunking")
        return {
            "total_chunks": 0,
            "total_conversations": 0,
            "worker_results": [],
            "failed_workers": [],
        }

    # Chunk modified conversations in parallel
    result = parallel_chunk_conversations(
        root=root,
        output_dir=chunks_dir,
        conversation_filter=to_chunk_ids,
    )

    # Update timestamp
    try:
        last_chunk_file.write_text(str(time.time()))
    except Exception as e:
        logger.warning("Failed to update last chunk timestamp: %s", e)

    logger.info(
        "Incremental rechunk complete: %d conversations, %d chunks",
        result["total_conversations"],
        result["total_chunks"],
    )

    return result


def surgical_rechunk(
    root: Path,
    conv_ids: list[str],
) -> dict[str, Any]:
    """
    Re-chunk specific selected conversations only.

    This operation:
    1. Validates that specified conversations exist
    2. Chunks only those conversations in parallel
    3. Overwrites existing chunk files for those conversations

    Args:
        root: Export root containing conversation folders
        conv_ids: List of conversation IDs to rechunk
    """
    root = Path(root)
    chunks_dir = root / "_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    if not conv_ids:
        logger.warning("No conversation IDs provided for surgical rechunk")
        return {
            "total_chunks": 0,
            "total_conversations": 0,
            "worker_results": [],
            "failed_workers": [],
        }

    # Use the provided conv_ids as filter
    conv_ids_set = set(conv_ids)

    logger.info(
        "Surgical rechunk: Processing %d selected conversations", len(conv_ids_set)
    )

    # Chunk selected conversations in parallel
    result = parallel_chunk_conversations(
        root=root,
        output_dir=chunks_dir,
        conversation_filter=conv_ids_set,
    )

    logger.info(
        "Surgical rechunk complete: %d conversations, %d chunks",
        result["total_conversations"],
        result["total_chunks"],
    )

    return result
