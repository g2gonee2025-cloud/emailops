#!/usr/bin/env python3
"""
Text chunking utilities for EmailOps.
Implements smart text chunking with overlap for large documents.

Refactor highlights:
- Centralized defaults (env-overridable)
- Added `prepare_index_units` convenience to emit either a single record or chunks
- Kept existing, battle-tested chunking logic intact
"""
from __future__ import annotations

import os
import re
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---- Tunables (env overrides allowed) ---------------------------------------
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1600"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_CHUNK_THRESHOLD = int(os.getenv("CHUNK_THRESHOLD", "8000"))  # when to chunk vs. pass-through

@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    min_chunk_size: int = 100
    respect_sentences: bool = True
    respect_paragraphs: bool = True
    base_max_chunks: int = 25
    progressive_scaling: bool = True

    def __post_init__(self) -> None:
        # Validation & clamping to ensure forward progress and sensible defaults
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk_overlap < 0:
            logger.warning("chunk_overlap < 0; clamping to 0")
            self.chunk_overlap = 0
        if self.chunk_overlap >= self.chunk_size:
            # Keep a conservative overlap to avoid degeneracy
            new_overlap = max(0, self.chunk_size // 4)
            logger.warning("chunk_overlap >= chunk_size; reducing overlap to %d", new_overlap)
            self.chunk_overlap = new_overlap
        if self.min_chunk_size <= 0 or self.min_chunk_size > self.chunk_size:
            # Avoid pathological settings
            suggested = max(1, min(self.chunk_size // 5, 200))
            logger.warning("min_chunk_size must be in [1, chunk_size]; setting to %d", suggested)
            self.min_chunk_size = suggested
        if self.base_max_chunks <= 0:
            logger.warning("base_max_chunks must be > 0; setting to 25")
            self.base_max_chunks = 25

class TextChunker:
    """Intelligent text chunker that respects semantic boundaries when possible."""
    _SENTENCE_END_RE = re.compile(
        r'(?:(?<=\.)|(?<=\!)|(?<=\?))[\s\n]+'
        r'|:\s*\n+'
        r'|\n{2,}'
    )
    _PARA_BREAK_RE = re.compile(r'\n{2,}')

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

    # ----------------------------- Public API ----------------------------- #
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not text:
            return []
        if text.startswith('\ufeff'):
            text = text[1:]
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.config.chunk_size:
            return [self._create_chunk(text, 0, len(text), 0, metadata)]

        chunks: List[Dict[str, Any]] = []
        if self.config.respect_paragraphs:
            chunks = self._chunk_by_paragraphs(text, metadata)
        if not chunks and self.config.respect_sentences:
            chunks = self._chunk_by_sentences(text, metadata)
        if not chunks:
            chunks = self._chunk_by_window(text, metadata)

        max_allowed = self._calculate_max_chunks(len(chunks)) if self.config.progressive_scaling else self.config.base_max_chunks
        if len(chunks) > max_allowed:
            logger.warning("Document produced %d chunks; limiting to %d", len(chunks), max_allowed)
            chunks = chunks[:max_allowed]
        for i, ch in enumerate(chunks):
            ch["chunk_index"] = i
        return chunks

    # --------------------------- Helper methods --------------------------- #
    def _calculate_max_chunks(self, chunks_needed: int) -> int:
        base = self.config.base_max_chunks
        if chunks_needed <= base:
            return chunks_needed
        above = chunks_needed - base
        additional = above // 3
        return base + additional

    def _iter_paragraph_spans(self, text: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        last = 0
        for m in self._PARA_BREAK_RE.finditer(text):
            start, end = last, m.start()
            if start < end and text[start:end].strip():
                spans.append((start, end))
            last = m.end()
        if last < len(text) and text[last:].strip():
            spans.append((last, len(text)))
        return spans

    def _chunk_by_paragraphs(self, text: str, metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        spans = self._iter_paragraph_spans(text)
        if len(spans) <= 1:
            return []
        for a, b in spans:
            if (b - a) > self.config.chunk_size:
                return []
        chunks: List[Dict[str, Any]] = []
        i = 0
        last_start_pos = -1
        while i < len(spans):
            j = i
            while j < len(spans) and (spans[j][1] - spans[i][0]) <= self.config.chunk_size:
                j += 1
            start_pos = spans[i][0]
            end_pos = spans[j - 1][1]
            if start_pos == last_start_pos:
                i = max(i + 1, j)
                continue
            chunk_text = text[start_pos:end_pos].strip()
            chunks.append(self._create_chunk(chunk_text, start_pos, end_pos, len(chunks), metadata))
            last_start_pos = start_pos
            if j >= len(spans):
                break
            if self.config.chunk_overlap > 0:
                desired = max(spans[i][0], end_pos - self.config.chunk_overlap)
                k = i
                for idx in range(i, j):
                    s, e = spans[idx]
                    if s <= desired < e:
                        k = idx
                        break
                    if desired < s:
                        k = max(i, idx - 1)
                        break
                if k <= i:
                    i = j - 1
                else:
                    i = k
            else:
                i = j
        return chunks

    def _chunk_by_sentences(self, text: str, metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        boundaries: List[int] = [m.end() for m in self._SENTENCE_END_RE.finditer(text)]
        if not boundaries or boundaries[-1] < len(text):
            boundaries.append(len(text))
        chunks: List[Dict[str, Any]] = []
        chunk_start = 0
        for idx, boundary in enumerate(boundaries):
            cur_size = boundary - chunk_start
            is_last = idx == (len(boundaries) - 1)
            if cur_size >= self.config.chunk_size or is_last:
                ctext = text[chunk_start:boundary].strip()
                if ctext and (is_last or len(ctext) >= self.config.min_chunk_size):
                    chunks.append(self._create_chunk(ctext, chunk_start, boundary, len(chunks), metadata))
                if not is_last and self.config.chunk_overlap > 0:
                    desired = max(0, boundary - self.config.chunk_overlap)
                    new_start = desired
                    for j in range(idx - 1, -1, -1):
                        prev_b = boundaries[j]
                        if boundary - prev_b <= self.config.chunk_overlap:
                            new_start = prev_b
                            break
                    if new_start <= chunk_start:
                        new_start = boundary
                    chunk_start = new_start
                else:
                    chunk_start = boundary
        return chunks

    def _chunk_by_window(self, text: str, metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        text_len = len(text)
        start = 0
        while start < text_len:
            end = min(start + self.config.chunk_size, text_len)
            if end < text_len:
                lower_bound = max(start + self.config.min_chunk_size, end - 50)
                i = end - 1
                while i >= lower_bound:
                    if text[i] in ' \n\t.!?,;:':
                        end = i + 1
                        break
                    i -= 1
            ctext = text[start:end].strip()
            is_last = end >= text_len
            if ctext and (is_last or len(ctext) >= self.config.min_chunk_size):
                chunks.append(self._create_chunk(ctext, start, end, len(chunks), metadata))
            if end >= text_len:
                break
            next_start = max(0, end - self.config.chunk_overlap)
            if chunks and next_start <= chunks[-1]["start_pos"]:
                next_start = end
            start = next_start
        return chunks

    def _create_chunk(self, text: str, start_pos: int, end_pos: int, chunk_index: int, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        chunk = {"text": text, "start_pos": start_pos, "end_pos": end_pos, "chunk_index": chunk_index, "chunk_size": len(text)}
        if metadata:
            chunk["metadata"] = metadata
        return chunk


# ---------------------------- Convenience API ---------------------------- #

def should_chunk_text(text: str, threshold: Optional[int] = None) -> bool:
    """Decide if text should be chunked based on configurable size threshold."""
    th = int(threshold if threshold is not None else DEFAULT_CHUNK_THRESHOLD)
    return len(text) > th

def chunk_for_indexing(
    text: str,
    doc_id: str,
    doc_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    subject: Optional[str] = None,
    date: Optional[str] = None
) -> List[Dict[str, Any]]:
    config = ChunkConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap, respect_sentences=True, respect_paragraphs=True)
    chunker = TextChunker(config)
    metadata: Dict[str, Any] = {"doc_id": doc_id, "doc_path": doc_path}
    if subject:
        metadata["subject"] = subject
    if date:
        metadata["date"] = date
    chunks = chunker.chunk_text(text, metadata)
    out: List[Dict[str, Any]] = []
    for ch in chunks:
        rec = {
            "id": f"{doc_id}::chunk{ch['chunk_index']}",
            "path": doc_path,
            "text": ch["text"],
            "chunk_index": ch["chunk_index"],
            "chunk_size": ch["chunk_size"],
        }
        if "metadata" in ch:
            for k, v in ch["metadata"].items():
                if k not in ("doc_id", "doc_path"):
                    rec[k] = v
        out.append(rec)
    logger.debug("Created %d chunks for document %s", len(out), doc_id)
    return out

def prepare_index_units(
    text: str,
    doc_id: str,
    doc_path: str,
    subject: Optional[str] = None,
    date: Optional[str] = None,
    force_chunk: bool = False,
    threshold: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Emit either a single record or a set of chunks for indexing with consistent metadata."""
    if force_chunk or should_chunk_text(text, threshold=threshold):
        return chunk_for_indexing(
            text, doc_id=doc_id, doc_path=doc_path,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            subject=subject, date=date
        )
    # pass-through (no chunking)
    record = {
        "id": doc_id,
        "path": doc_path,
        "text": text[:200000],
        "subject": subject or "",
        "date": date,
    }
    return [record]

if __name__ == "__main__":
    # Simple smoke test / demo consistency with previous API remains the same.
    import json
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    sample = "Hello. This is a sample paragraph.\n\nAnother paragraph follows. It has multiple sentences for testing."
    res = prepare_index_units(sample, doc_id="D1", doc_path="/tmp/doc.txt", subject="Demo", date="2024-01-01")
    print(json.dumps(res[:2], indent=2))
