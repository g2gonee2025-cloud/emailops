from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import re
import bisect

# -----------------------------------------------------------------------------
# Precompiled patterns for boundary-aware chunking
# -----------------------------------------------------------------------------
# Paragraph boundary: one or more blank lines
PARA_RE = re.compile(r'\n\s*\n+')
# Sentence boundary: ., !, ?, and common CJK equivalents; allow quotes/parens; include end-of-document
SENT_RE = re.compile(r'[.!?。！？]+[)"\']*(?:\s+|$)')


@dataclass
class ChunkConfig:
    """
    Configuration for text chunking behavior.

    Attributes:
        chunk_size: Target size of each chunk in characters (default: 1500).
        chunk_overlap: Characters to overlap between chunks (default: 100).
        min_chunk_size: Minimum chunk size; smaller final chunks are merged (default: 50).
        progressive_scaling: Auto-scale chunk size for large docs (default: True).
        respect_sentences: Prefer cutting at sentence boundaries (default: True).
        respect_paragraphs: Prefer cutting at paragraph boundaries (default: True).
        max_chunks: Optional limit on number of chunks (default: None = unlimited).
        encoding: Text encoding (default: "utf-8").
    """
    chunk_size: int = 1500
    chunk_overlap: int = 100
    min_chunk_size: int = 50
    progressive_scaling: bool = True
    respect_sentences: bool = True
    respect_paragraphs: bool = True
    max_chunks: Optional[int] = None
    encoding: str = "utf-8"


def _apply_progressive_scaling(total_len: int, size: int, overlap: int, enable: bool) -> tuple[int, int]:
    """
    Simple heuristic to scale chunk size for very large texts to reduce chunk counts.
    Maintains the overlap ratio approximately.
    """
    if not enable:
        return max(1, size), max(0, min(overlap, max(1, size) - 1))

    # Scale up for long documents.
    factor = 1.0
    if total_len >= 300_000:
        factor = 2.0
    elif total_len >= 150_000:
        factor = 1.5
    elif total_len >= 80_000:
        factor = 1.25

    new_size = max(1, int(size * factor))
    # Keep overlap proportional, but strictly less than new_size
    new_overlap = int(overlap * (new_size / max(1, size)))
    new_overlap = max(0, min(new_overlap, new_size - 1))
    return new_size, new_overlap


def _compute_breakpoints(text: str, respect_sentences: bool, respect_paragraphs: bool) -> List[int]:
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

    # Always allow end of document as a breakpoint.
    breaks.add(len(text))
    return sorted(breaks)


def _ranges_with_overlap(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    max_chunks: Optional[int],
    respect_sentences: bool,
    respect_paragraphs: bool,
) -> List[Tuple[int, int]]:
    """
    Produce (start, end) ranges over `text` honoring overlap and optional boundary-aware cutting.

    Guarantees forward progress even if chunk_overlap >= chunk_size.
    Ensures the final chunk is not smaller than `min_chunk_size` (if possible) by merging it into the
    previous chunk.

    When `max_chunks` is provided, the final produced chunk captures the remainder of the document
    (capture-tail semantics).
    """
    n = len(text)
    if n == 0:
        return []

    # Clamp parameters to safe values.
    chunk_size = max(1, chunk_size)
    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
    min_chunk_size = max(0, min(min_chunk_size, chunk_size))
    if max_chunks is not None and max_chunks < 1:
        max_chunks = 1

    # Fast path: skip boundary work entirely if not requested
    boundary_points: Optional[List[int]] = (
        _compute_breakpoints(text, respect_sentences, respect_paragraphs)
        if (respect_sentences or respect_paragraphs) else None
    )

    ranges: List[Tuple[int, int]] = []
    start = 0

    # How far we'll search ahead of the target size to find a nicer boundary.
    lookahead_limit = max(1, int(0.2 * chunk_size))

    while start < n and (max_chunks is None or len(ranges) < max_chunks):
        ideal_end = min(n, start + chunk_size)

        # Snap to a nearby boundary if available
        if boundary_points is not None:
            i = bisect.bisect_right(boundary_points, ideal_end)
            candidate_before = boundary_points[i - 1] if i > 0 else None
            candidate_after = boundary_points[i] if i < len(boundary_points) else None

            end = ideal_end
            # Prefer a boundary before the ideal end as long as it doesn't make the chunk too tiny.
            if candidate_before is not None and candidate_before >= start + max(1, min_chunk_size):
                end = candidate_before
            # Otherwise, if there's a boundary shortly after ideal_end, use that.
            elif candidate_after is not None and candidate_after - ideal_end <= lookahead_limit:
                end = candidate_after
        else:
            end = ideal_end

        # Fallback guard to ensure forward motion.
        if end <= start:
            end = min(n, start + chunk_size)

        # Capture-tail semantics for the last allowed chunk
        if max_chunks is not None and len(ranges) + 1 >= max_chunks:
            end = n

        ranges.append((start, end))

        # If we've reached the end of text, stop to avoid redundant trailing windows.
        if end >= n:
            break

        # Advance start by the actual chunk length minus overlap; always at least 1 char.
        actual_len = end - start
        step = max(1, actual_len - chunk_overlap)
        start += step

    # If the very last chunk is smaller than min_chunk_size and there is a previous chunk, merge it.
    if len(ranges) >= 2:
        last_start, last_end = ranges[-1]
        if (last_end - last_start) < min_chunk_size:
            prev_start, prev_end = ranges[-2]
            # Merge by extending the previous end; drop the last range.
            ranges[-2] = (prev_start, last_end)
            ranges.pop()

    return ranges


class TextChunker:
    def __init__(self, config: ChunkConfig):
        self.config = config

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk `text` according to the `ChunkConfig`.

        Notes:
            - Prevents infinite loops if overlap >= size (guaranteed forward progress).
            - Optionally prefers sentence/paragraph boundaries when cutting.
            - Ensures the final chunk meets `min_chunk_size` by merging the tail, if needed.
            - Each chunk's metadata includes `start_char` and `end_char` (exclusive).
            - When `max_chunks` is set, the last produced chunk captures the remainder of the document.
        """
        if not text:
            return []

        # Apply progressive scaling heuristics on very large texts.
        eff_chunk_size, eff_overlap = _apply_progressive_scaling(
            total_len=len(text),
            size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            enable=self.config.progressive_scaling,
        )

        ranges = _ranges_with_overlap(
            text=text,
            chunk_size=eff_chunk_size,
            chunk_overlap=eff_overlap,
            min_chunk_size=self.config.min_chunk_size,
            max_chunks=self.config.max_chunks,
            respect_sentences=self.config.respect_sentences,
            respect_paragraphs=self.config.respect_paragraphs,
        )

        chunks: List[Dict[str, Any]] = []
        for idx, (start, end) in enumerate(ranges):
            chunk_text = text[start:end]
            chunk_metadata = dict(metadata or {})
            chunk_metadata["chunk_index"] = idx
            # Include offsets
            chunk_metadata["start_char"] = start
            chunk_metadata["end_char"] = end  # exclusive
            chunks.append({"text": chunk_text, "metadata": chunk_metadata})

        return chunks


def prepare_index_units(
    text: str,
    doc_id: str,
    doc_path: str,
    subject: str = "",
    date: Optional[str] = None,
    # Optional chunking parameters. If None, values are taken from `config` (or ChunkConfig defaults).
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    min_chunk_size: Optional[int] = None,
    respect_sentences: Optional[bool] = None,
    respect_paragraphs: Optional[bool] = None,
    progressive_scaling: Optional[bool] = None,
    max_chunks: Optional[int] = None,
    config: Optional[ChunkConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Prepare text for indexing by splitting it into chunks with metadata.

    This function is used by indexing pipelines (e.g., email_indexer.py) to create
    indexable units from conversation text and attachments.

    Args:
        text: The text content to chunk.
        doc_id: Base document identifier (e.g., "conv_id::conversation" or "conv_id::att1").
        doc_path: Path to the source document.
        subject: Email subject or document title.
        date: Optional date information.

        chunk_size: Maximum size of each chunk in characters. If None, pulls from `config` or default.
        chunk_overlap: Number of characters to overlap between chunks. If None, pulls from `config` or default.
        min_chunk_size: Minimum size for the final chunk; if the tail is smaller, it is merged into the previous chunk.
                        If None, pulls from `config` or default.
        respect_sentences: Prefer cutting at sentence boundaries. If None, pulls from `config`.
        respect_paragraphs: Prefer cutting at blank-line paragraph boundaries. If None, pulls from `config`.
        progressive_scaling: Enable size scaling for very large texts. If None, pulls from `config`.
        max_chunks: Optional cap on the number of chunks. If None, pulls from `config`. When set, the final produced
                    chunk captures the remainder of the document (capture-tail semantics).
        config: Optional ChunkConfig; when provided, supplies defaults for omitted parameters.

    Returns:
        List of chunk dictionaries, each containing:
            - id: Unique identifier for the chunk (doc_id::chunk{N}; first chunk uses doc_id).
            - text: The chunk text content.
            - path: Path to source document.
            - subject: Subject/title.
            - date: Date information (if provided).
            - start_char: Start character offset (inclusive).
            - end_char: End character offset (exclusive).

    Notes:
        - Offsets (`start_char`, `end_char`) are character indices in the original `text`.
        - When `max_chunks` is set, the last produced chunk captures the rest of the document.
    """
    if not text or not text.strip():
        return []

    # Resolve configuration values with the following precedence:
    # explicit arg -> config value -> ChunkConfig() default
    cfg = config or ChunkConfig()
    eff_chunk_size = cfg.chunk_size if chunk_size is None else chunk_size
    eff_chunk_overlap = cfg.chunk_overlap if chunk_overlap is None else chunk_overlap
    eff_min_chunk_size = cfg.min_chunk_size if min_chunk_size is None else min_chunk_size
    eff_respect_sentences = cfg.respect_sentences if respect_sentences is None else respect_sentences
    eff_respect_paragraphs = cfg.respect_paragraphs if respect_paragraphs is None else respect_paragraphs
    eff_progressive_scaling = cfg.progressive_scaling if progressive_scaling is None else progressive_scaling
    eff_max_chunks = cfg.max_chunks if max_chunks is None else max_chunks

    # Apply progressive scaling heuristics on very large texts.
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

    chunks: List[Dict[str, Any]] = []
    for idx, (start, end) in enumerate(ranges):
        chunk_text = text[start:end]

        chunk: Dict[str, Any] = {
            "id": f"{doc_id}::chunk{idx}" if idx > 0 else doc_id,
            "text": chunk_text,
            "path": doc_path,
            "subject": subject,
            "start_char": start,
            "end_char": end,  # exclusive
        }
        if date:
            chunk["date"] = date

        chunks.append(chunk)

    return chunks
