from __future__ import annotations

import bisect
import logging
import re
from typing import Any

from .core_config import get_config

logger = logging.getLogger(__name__)

__all__ = (
    "prepare_index_units",
)


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
