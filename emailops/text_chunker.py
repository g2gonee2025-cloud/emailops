#!/usr/bin/env python3
"""
Text chunking utilities for EmailOps.
Implements smart text chunking with overlap for large documents.

Changes vs. previous version:
- Fixed IndexError in window splitting (end-bound scanning).
- Accurate start/end positions for paragraph chunks via true span indices.
- Overlap clamped/validated; guarantees forward progress.
- Always keep the final (possibly small) chunk to avoid silent tail loss.
- Deterministic reindex after max-chunk limiting.
"""
from __future__ import annotations
import re
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    chunk_size: int = 1600          # Characters per chunk
    chunk_overlap: int = 200        # Character overlap between chunks
    min_chunk_size: int = 100       # Minimum chunk size (only used to suppress tiny *intermediate* chunks)
    respect_sentences: bool = True  # Try to break at sentence boundaries
    respect_paragraphs: bool = True # Try to break at paragraph boundaries
    base_max_chunks: int = 25       # Base maximum chunks from a single document
    progressive_scaling: bool = True  # Enable progressive chunk scaling

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
            logger.warning(
                "chunk_overlap >= chunk_size; reducing overlap to %d", new_overlap
            )
            self.chunk_overlap = new_overlap
        if self.min_chunk_size <= 0 or self.min_chunk_size > self.chunk_size:
            # Avoid pathological settings
            suggested = max(1, min(self.chunk_size // 5, 200))
            logger.warning(
                "min_chunk_size must be in [1, chunk_size]; setting to %d", suggested
            )
            self.min_chunk_size = suggested
        if self.base_max_chunks <= 0:
            logger.warning("base_max_chunks must be > 0; setting to 25")
            self.base_max_chunks = 25


class TextChunker:
    """
    Intelligent text chunker that respects semantic boundaries when possible.
    Falls back to robust windowed splitting for worst cases.
    """

    # Sentence endings:
    # - standard punctuation + whitespace/newlines
    # - a colon followed by a newline (common for section headers)
    # - paragraph breaks (double newlines)
    _SENTENCE_END_RE = re.compile(
        r'(?:(?<=\.)|(?<=\!)|(?<=\?))[\s\n]+'      # ., !, ? followed by whitespace/newline
        r'|:\s*\n+'                                # header-like "Title:\n"
        r'|\n{2,}'                                 # paragraph break as hard boundary
    )
    _PARA_BREAK_RE = re.compile(r'\n{2,}')

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

    # ----------------------------- Public API ----------------------------- #

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        Returns a list of dicts containing: text, start_pos, end_pos, chunk_index, chunk_size, and optional metadata.
        """
        if not text:
            return []

        # Trim BOM and surrounding whitespace
        if text.startswith('\ufeff'):
            text = text[1:]
        text = text.strip()
        if not text:
            return []

        # Small enough? Return single chunk.
        if len(text) <= self.config.chunk_size:
            return [self._create_chunk(text, 0, len(text), 0, metadata)]

        # Prefer paragraph-aware splitting; fallback to sentence; then window.
        chunks: List[Dict[str, Any]] = []
        if self.config.respect_paragraphs:
            chunks = self._chunk_by_paragraphs(text, metadata)

        if not chunks and self.config.respect_sentences:
            chunks = self._chunk_by_sentences(text, metadata)

        if not chunks:
            chunks = self._chunk_by_window(text, metadata)

        # Progressive scaling / limiting
        if self.config.progressive_scaling:
            max_allowed = self._calculate_max_chunks(len(chunks))
        else:
            max_allowed = self.config.base_max_chunks

        if len(chunks) > max_allowed:
            logger.warning(
                "Document produced %d chunks; limiting to %d", len(chunks), max_allowed
            )
            chunks = chunks[:max_allowed]

        # Ensure contiguous chunk_index after any trimming
        for i, ch in enumerate(chunks):
            ch["chunk_index"] = i

        return chunks

    # --------------------------- Helper methods --------------------------- #

    def _calculate_max_chunks(self, chunks_needed: int) -> int:
        """
        Calculate maximum allowed chunks using progressive scaling.

        base_limit + (chunks_above_base // 3)
        Examples:
          25 -> 25
         100 -> 25 + (75/3) = 50
         200 -> 25 + (175/3) = 83
         600 -> 25 + (575/3) = 216
        """
        base = self.config.base_max_chunks
        if chunks_needed <= base:
            return chunks_needed
        above = chunks_needed - base
        additional = above // 3
        allowed = base + additional
        logger.debug(
            "Progressive scaling: needed=%d allowed=%d (base=%d additional=%d)",
            chunks_needed, allowed, base, additional
        )
        return allowed

    # -------- Paragraph-aware chunking (span-accurate, bounded size) ------ #

    def _iter_paragraph_spans(self, text: str) -> List[Tuple[int, int]]:
        """
        Return list of (start, end) spans for paragraphs split by double newlines,
        skipping empty/whitespace-only blocks.
        """
        spans: List[Tuple[int, int]] = []
        last = 0
        for m in self._PARA_BREAK_RE.finditer(text):
            start, end = last, m.start()
            if start < end and text[start:end].strip():
                spans.append((start, end))
            last = m.end()
        # final segment
        if last < len(text) and text[last:].strip():
            spans.append((last, len(text)))
        return spans

    def _chunk_by_paragraphs(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        spans = self._iter_paragraph_spans(text)
        if len(spans) <= 1:
            return []  # not enough paragraph structure to help

        # If any single paragraph is too large, let sentence/window logic handle it.
        for a, b in spans:
            if (b - a) > self.config.chunk_size:
                return []

        chunks: List[Dict[str, Any]] = []
        i = 0
        last_start_pos = -1

        while i < len(spans):
            # Grow [i..j] while staying within chunk_size
            j = i
            while j < len(spans) and (spans[j][1] - spans[i][0]) <= self.config.chunk_size:
                j += 1

            # At least one paragraph should be included by construction
            start_pos = spans[i][0]
            end_pos = spans[j - 1][1]
            if start_pos == last_start_pos:
                # Safety: ensure forward progress in pathological settings
                i = max(i + 1, j)
                continue

            chunk_text = text[start_pos:end_pos].strip()
            chunks.append(self._create_chunk(
                chunk_text, start_pos, end_pos, len(chunks), metadata
            ))
            last_start_pos = start_pos

            if j >= len(spans):
                break

            # Compute next window start with character overlap, but try to align to a paragraph start
            if self.config.chunk_overlap > 0:
                desired = max(spans[i][0], end_pos - self.config.chunk_overlap)

                # Find the paragraph containing 'desired' (or the closest start not after desired)
                k = i
                for idx in range(i, j):  # within current chunk's paragraphs
                    s, e = spans[idx]
                    if s <= desired < e:
                        k = idx
                        break
                    if desired < s:
                        # desired falls before this paragraph; start at previous if possible
                        k = max(i, idx - 1)
                        break
                # Ensure progress; if k <= i, move nearer to the end of current chunk
                if k <= i:
                    i = j - 1  # overlap last paragraph of the previous chunk
                else:
                    i = k
            else:
                i = j

        return chunks

    # ----------------------------- Sentences ------------------------------ #

    def _chunk_by_sentences(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # Build boundary indices
        boundaries: List[int] = [m.end() for m in self._SENTENCE_END_RE.finditer(text)]
        if not boundaries or boundaries[-1] < len(text):
            boundaries.append(len(text))

        chunks: List[Dict[str, Any]] = []
        chunk_start = 0

        for idx, boundary in enumerate(boundaries):
            cur_size = boundary - chunk_start
            is_last = idx == (len(boundaries) - 1)

            if cur_size >= self.config.chunk_size or is_last:
                chunk_text = text[chunk_start:boundary].strip()
                # Keep the last chunk even if smaller than min_chunk_size
                if chunk_text and (is_last or len(chunk_text) >= self.config.min_chunk_size):
                    chunks.append(self._create_chunk(
                        chunk_text, chunk_start, boundary, len(chunks), metadata
                    ))

                # Compute next start with overlap, aligned to previous boundary if possible
                if not is_last and self.config.chunk_overlap > 0:
                    desired = max(0, boundary - self.config.chunk_overlap)
                    # Snap to the nearest previous boundary within overlap
                    new_start = desired
                    for j in range(idx - 1, -1, -1):
                        prev_b = boundaries[j]
                        if boundary - prev_b <= self.config.chunk_overlap:
                            new_start = prev_b
                            break
                    # Ensure progress
                    if new_start <= chunk_start:
                        new_start = boundary
                    chunk_start = new_start
                else:
                    chunk_start = boundary

        return chunks

    # ------------------------------ Window -------------------------------- #

    def _chunk_by_window(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Sliding window chunking with soft break at whitespace/punctuation near window edges.
        Always emits the final chunk if non-empty.
        """
        chunks: List[Dict[str, Any]] = []
        text_len = len(text)
        start = 0

        while start < text_len:
            end = min(start + self.config.chunk_size, text_len)

            # Try to find a nicer break a bit before the window end (up to 50 chars)
            if end < text_len:
                lower_bound = max(start + self.config.min_chunk_size, end - 50)
                i = end - 1
                while i >= lower_bound:
                    if text[i] in ' \n\t.!?,;:':
                        end = i + 1
                        break
                    i -= 1

            chunk_text = text[start:end].strip()
            # Keep last chunk even if small
            is_last = end >= text_len
            if chunk_text and (is_last or len(chunk_text) >= self.config.min_chunk_size):
                chunks.append(self._create_chunk(
                    chunk_text, start, end, len(chunks), metadata
                ))

            if end >= text_len:
                break

            # Advance with overlap
            next_start = max(0, end - self.config.chunk_overlap)

            # Guarantee progress
            if chunks and next_start <= chunks[-1]["start_pos"]:
                next_start = end
            start = next_start

        return chunks

    # --------------------------- Chunk creation --------------------------- #

    def _create_chunk(
        self,
        text: str,
        start_pos: int,
        end_pos: int,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        chunk = {
            "text": text,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "chunk_index": chunk_index,
            "chunk_size": len(text)
        }
        if metadata:
            chunk["metadata"] = metadata
        return chunk


# ---------------------------- Convenience API ---------------------------- #

def chunk_for_indexing(
    text: str,
    doc_id: str,
    doc_path: str,
    chunk_size: int = 1600,
    chunk_overlap: int = 200,
    subject: Optional[str] = None,
    date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Chunk text for indexing with appropriate metadata.
    Returns items with keys:
      id, path, text, chunk_index, chunk_size, and any additional metadata (e.g., subject/date).
    """
    config = ChunkConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        respect_sentences=True,
        respect_paragraphs=True
    )
    chunker = TextChunker(config)

    metadata: Dict[str, Any] = {"doc_id": doc_id, "doc_path": doc_path}
    if subject:
        metadata["subject"] = subject
    if date:
        metadata["date"] = date

    chunks = chunker.chunk_text(text, metadata)

    index_chunks: List[Dict[str, Any]] = []
    for ch in chunks:
        ic = {
            "id": f"{doc_id}::chunk{ch['chunk_index']}",
            "path": doc_path,
            "text": ch["text"],
            "chunk_index": ch["chunk_index"],
            "chunk_size": ch["chunk_size"]
        }
        # Carry through metadata fields except internal duplication
        if "metadata" in ch:
            for k, v in ch["metadata"].items():
                if k not in ("doc_id", "doc_path"):
                    ic[k] = v
        index_chunks.append(ic)

    logger.debug("Created %d chunks for document %s", len(index_chunks), doc_id)
    return index_chunks


def should_chunk_text(text: str, threshold: int = 2000) -> bool:
    """
    Determine if text should be chunked based on size.
    """
    return len(text) > threshold


if __name__ == "__main__":
    # Simple smoke test / demo
    import json
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    test_text = """
    This is the first paragraph of our test document. It contains several sentences that discuss the initial topic. The paragraph is designed to be of moderate length to test our chunking algorithm.

    Here we have the second paragraph, which introduces a new idea. This paragraph is also multiple sentences long. We want to see how the chunker handles paragraph boundaries when creating chunks with overlap.

    The third paragraph continues the discussion with yet another topic. It's important that our chunking algorithm respects these natural boundaries in the text. This helps maintain semantic coherence in the resulting chunks.

    A fourth paragraph adds more content to ensure we have enough text to create multiple chunks. The chunker should handle this gracefully, creating overlapping segments that preserve context.

    Finally, the fifth paragraph concludes our test document. By having multiple paragraphs, we can test both paragraph-aware and sentence-aware chunking strategies. This ensures our implementation is robust and handles various text structures effectively.
    """

    configs = [
        ChunkConfig(chunk_size=200, chunk_overlap=50),
        ChunkConfig(chunk_size=400, chunk_overlap=100),
        ChunkConfig(chunk_size=150, chunk_overlap=0, respect_paragraphs=False),
    ]

    for i, cfg in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: chunk_size={cfg.chunk_size}, overlap={cfg.chunk_overlap}, "
              f"respect_paragraphs={cfg.respect_paragraphs}")
        print(f"{'='*60}")
        chunker = TextChunker(cfg)
        chunks = chunker.chunk_text(test_text, {"test_id": i + 1})
        for j, ch in enumerate(chunks):
            head = (ch['text'][:150] + '...') if len(ch['text']) > 150 else ch['text']
            print(f"\nChunk {j+1} (size: {ch['chunk_size']}, start:{ch['start_pos']}, end:{ch['end_pos']}):")
            print("-" * 40)
            print(head)

    print(f"\n{'='*60}\nTesting chunk_for_indexing function\n{'='*60}")
    index_chunks = chunk_for_indexing(
        test_text,
        doc_id="TEST001",
        doc_path="/test/document.txt",
        chunk_size=300,
        chunk_overlap=75,
        subject="Test Document",
        date="2024-01-01"
    )
    print(f"\nGenerated {len(index_chunks)} chunks for indexing:")
    print(json.dumps(index_chunks[0], indent=2))
