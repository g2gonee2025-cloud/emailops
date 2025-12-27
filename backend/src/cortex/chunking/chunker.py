"""
Text Chunking.

Implements §7.1 of the Canonical Blueprint:
- Split text into chunks respecting max_tokens/min_tokens/overlap_tokens
- Respect quoted_spans when classifying chunk_type
- Generate stable content_hash for deduplication
- Prefer sentence/paragraph boundaries for cleaner splits
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Try to import tiktoken for accurate token counting
_HAS_TIKTOKEN = False
_tiktoken: Any = None

try:
    import tiktoken as _tiktoken

    _HAS_TIKTOKEN = True
except ImportError:
    logger.info(
        "tiktoken not available; using character-based approximation for token counts"
    )


class Span(BaseModel):
    """Character span in text."""

    start: int
    end: int

    def overlaps(self, other: Span) -> bool:
        """Check if this span overlaps with another."""
        return not (self.end <= other.start or other.end <= self.start)

    def overlap_ratio(self, other: Span) -> float:
        """Calculate what percentage of this span overlaps with another."""
        if self.end <= other.start or other.end <= self.start:
            return 0.0

        overlap_start = max(self.start, other.start)
        overlap_end = min(self.end, other.end)
        overlap_len = max(0, overlap_end - overlap_start)
        span_len = self.end - self.start

        return overlap_len / span_len if span_len > 0 else 0.0


class ChunkingInput(BaseModel):
    """
    Input for chunking.

    Blueprint §7.1:
    * text: str
    * section_path: str
    * quoted_spans: List[Span]
    * max_tokens: int = 800
    * min_tokens: int = 25
    * overlap_tokens: int = 80
    """

    text: str
    section_path: str
    quoted_spans: list[Span] = Field(default_factory=list)
    max_tokens: int = 1600
    min_tokens: int = 25
    overlap_tokens: int = 200
    # Additional options
    model: str = "cl100k_base"  # tiktoken encoding to use
    chunk_type_hint: str | None = None  # Override chunk_type classification
    preserve_sentences: bool = True  # Try to split at sentence boundaries


class ChunkModel(BaseModel):
    """
    Chunk model.

    Blueprint §7.1:
    * text: str
    * summary: Optional[str]
    * section_path: str
    * position: int
    * char_start: int
    * char_end: int
    * chunk_type: Literal[...]
    * metadata: Dict[str, Any]
    """

    text: str
    summary: str | None = None
    section_path: str
    position: int
    char_start: int
    char_end: int
    chunk_type: Literal[
        "message_body", "attachment_text", "attachment_table", "quoted_history", "other"
    ]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """Get the content hash from metadata."""
        return self.metadata.get("content_hash", "")

    @property
    def token_count(self) -> int:
        """Get the token count from metadata."""
        return self.metadata.get("token_count", 0)


@dataclass
class TokenCounter:
    """
    Token counting utility.

    Uses tiktoken if available, otherwise approximates with chars/4.
    """

    model: str = "cl100k_base"
    _encoding: Any = None

    def __post_init__(self):
        if _HAS_TIKTOKEN:
            try:
                self._encoding = _tiktoken.get_encoding(self.model)
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding '{self.model}': {e}")

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if self._encoding:
            return len(self._encoding.encode(text))
        # Fallback: approximate as chars/4
        return len(text) // 4

    def tokens_to_chars(self, tokens: int) -> int:
        """Approximate character count for a token count."""
        # Average ~4 chars per token for English text
        return tokens * 4

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if self._encoding:
            return self._encoding.encode(text)
        # Fallback: just use characters
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        if self._encoding:
            return self._encoding.decode(tokens)
        # Fallback
        return bytes(tokens).decode("utf-8", errors="replace")


# Sentence boundary patterns
# Sentence boundary:
# - Western: . ! ?
# - East Asian: 。 ！ ？
# - Ellipsis: ...
# - Arabic/Persian/Urdu: ؟ ۔
# Allow common closing quotes/brackets after the punctuation, then whitespace or EoD.
SENTENCE_END_PATTERN = re.compile(
    r"[.!?。！？…؟۔]+"  # Simplified: use … instead of ... for ellipsis
    r'[)"\'\u2018\u2019»›）】〔〕〉》」』〗〙〞]*'
    r"(?:\s+|$)"  # Non-capturing group for alternation
)
# Paragraph boundary: one or more blank lines (support both LF and CRLF)
PARAGRAPH_PATTERN = re.compile(r"\r?\n\s*\r?\n+")


def find_sentence_boundary(text: str, target_pos: int, window: int = 100) -> int:
    """
    Find the nearest sentence boundary to target_pos.

    Searches within a window around target_pos.
    Returns the position after the sentence end, or target_pos if no boundary found.
    """
    start = max(0, target_pos - window)
    end = min(len(text), target_pos + window)
    search_text = text[start:end]

    # Look for sentence endings
    best_pos = target_pos
    best_dist = window + 1

    for match in SENTENCE_END_PATTERN.finditer(search_text):
        # The regex matches the punctuation + spacing. The boundary is at the end of the match.
        abs_pos = start + match.end()
        dist = abs(abs_pos - target_pos)
        if dist < best_dist:
            best_dist = dist
            best_pos = abs_pos

    # Also check for paragraph boundaries
    for match in PARAGRAPH_PATTERN.finditer(search_text):
        # Paragraph boundary is end of the blank lines
        abs_pos = start + match.end()
        dist = abs(abs_pos - target_pos)
        if dist < best_dist:
            best_dist = dist
            best_pos = abs_pos

    # Boundary clamp: ensure we don't return something beyond text length
    return min(best_pos, len(text))


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of text content for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def classify_chunk_type(
    char_start: int,
    char_end: int,
    quoted_spans: list[Span],
    type_hint: str | None = None,
    quote_threshold: float = 0.5,
) -> Literal[
    "message_body", "attachment_text", "attachment_table", "quoted_history", "other"
]:
    """
    Classify chunk type based on overlap with quoted spans.

    Per §7.1: "classify chunks overlapping heavily with quotes as chunk_type='quoted_history'"

    Args:
        char_start: Chunk start position
        char_end: Chunk end position
        quoted_spans: List of quoted text spans
        type_hint: Optional explicit type override
        quote_threshold: Minimum overlap ratio to classify as quoted_history

    Returns:
        Chunk type classification
    """
    if type_hint:
        if type_hint in (
            "message_body",
            "attachment_text",
            "attachment_table",
            "quoted_history",
            "other",
        ):
            return type_hint  # type: ignore

    chunk_span = Span(start=char_start, end=char_end)

    # Calculate total overlap with quoted spans
    total_overlap = 0.0
    for quoted in quoted_spans:
        ratio = chunk_span.overlap_ratio(quoted)
        total_overlap = max(
            total_overlap, ratio
        )  # Use max overlap with any quoted span

    if total_overlap >= quote_threshold:
        return "quoted_history"

    return "message_body"


def _apply_progressive_scaling(
    total_tokens: int, max_tokens: int, overlap_tokens: int
) -> tuple[int, int]:
    """
    Scale chunk size for very large texts to reduce chunk counts.

    Bands:
      ≤50 chunks -> 1x
      51-150 -> 1.5x
      151-500 -> 2.0x
      500+ -> 3.0x
    """
    effective_chunk_size = max(1, int(max_tokens) - int(overlap_tokens))
    estimated_chunks = max(
        1, (total_tokens + effective_chunk_size - 1) // effective_chunk_size
    )

    factor = 1.0
    if estimated_chunks <= 50:
        factor = 1.0
    elif estimated_chunks <= 150:
        factor = 1.5
    elif estimated_chunks <= 500:
        factor = 2.0
    else:
        factor = 3.0

    if factor > 1.0:
        new_max = max(1, int(max_tokens * factor))
        # Scale overlap to keep same ratio
        new_overlap = int(overlap_tokens * (new_max / max(1, max_tokens)))
        new_overlap = max(0, min(new_overlap, new_max - 1))
        return new_max, new_overlap

    return max_tokens, overlap_tokens


class Chunker:
    """
    Class-based chunker to manage state.
    """

    def __init__(self, input_data: ChunkingInput):
        self.input_data = input_data
        self.text = input_data.text
        self.counter = TokenCounter(model=input_data.model)
        self.all_tokens: list[int] = self.counter.encode(self.text)
        self.token_map: list[int] = []
        self._build_token_to_char_map()

    def _build_token_to_char_map(self):
        """
        Builds a map from token index to character start position in the original text.
        This is fast and reliable.
        """
        self.token_map = [0] * (len(self.all_tokens) + 1)
        current_pos = 0
        for i, token in enumerate(self.all_tokens):
            self.token_map[i] = current_pos
            current_pos += len(self.counter.decode([token]))
        self.token_map[len(self.all_tokens)] = current_pos

    def chunk(self) -> list[ChunkModel]:
        if not self.text or not self.text.strip():
            return []

        # Pre-calculate token count for progressive scaling
        total_tokens_pre = len(self.all_tokens)

        # Apply progressive scaling
        eff_max_tokens, eff_overlap_tokens = _apply_progressive_scaling(
            total_tokens_pre, self.input_data.max_tokens, self.input_data.overlap_tokens
        )

        total_tokens = len(self.all_tokens)
        chunks: list[ChunkModel] = []
        token_pos = 0
        section_idx = 0

        while token_pos < total_tokens:
            token_end = min(token_pos + eff_max_tokens, total_tokens)

            if self.input_data.preserve_sentences and token_end < total_tokens:
                target_char_pos = self.token_map[token_end]
                boundary_char_pos = find_sentence_boundary(self.text, target_char_pos)

                boundary_token_idx = -1
                for i in range(token_end, token_pos, -1):
                    if self.token_map[i] <= boundary_char_pos:
                        boundary_token_idx = i
                        break

                if boundary_token_idx > token_pos:
                    token_end = boundary_token_idx

            char_start = self.token_map[token_pos]
            char_end = self.token_map[token_end]

            chunk_text_str = self.text[char_start:char_end]
            if not chunk_text_str.strip():
                token_pos += 1
                continue

            token_count = self.counter.count(chunk_text_str)

            chunk_type = classify_chunk_type(
                char_start=char_start,
                char_end=char_end,
                quoted_spans=self.input_data.quoted_spans,
                type_hint=self.input_data.chunk_type_hint,
            )

            if (
                token_count < self.input_data.min_tokens
                and chunks
                and chunks[-1].chunk_type == chunk_type
            ):
                last_chunk = chunks[-1]
                combined_text = self.text[last_chunk.char_start : char_end]
                combined_tokens = self.counter.count(combined_text)

                if combined_tokens <= eff_max_tokens:
                    last_chunk.char_end = char_end
                    last_chunk.text = combined_text
                    last_chunk.metadata.update(
                        {
                            "content_hash": compute_content_hash(last_chunk.text),
                            "token_count": combined_tokens,
                            "original_length": len(last_chunk.text),
                        }
                    )
                else:
                    self._add_chunk(
                        chunks,
                        chunk_text_str,
                        token_count,
                        char_start,
                        char_end,
                        chunk_type,
                        section_idx,
                    )
                    section_idx += 1
            else:
                self._add_chunk(
                    chunks,
                    chunk_text_str,
                    token_count,
                    char_start,
                    char_end,
                    chunk_type,
                    section_idx,
                )
                section_idx += 1

            next_token_pos = token_end - eff_overlap_tokens
            if next_token_pos <= token_pos:
                next_token_pos = token_pos + 1
            token_pos = next_token_pos

        return chunks

    def _add_chunk(
        self, chunks_list, text, token_count, char_start, char_end, chunk_type, position
    ):
        chunks_list.append(
            ChunkModel(
                text=text,
                section_path=self.input_data.section_path,
                position=position,
                char_start=char_start,
                char_end=char_end,
                chunk_type=chunk_type,
                metadata={
                    "content_hash": compute_content_hash(text),
                    "token_count": token_count,
                    "original_length": len(text),
                },
            )
        )


def chunk_text(input_data: ChunkingInput) -> list[ChunkModel]:
    """
    Legacy wrapper for backward compatibility.
    Initializes and runs the Chunker class.
    """
    chunker = Chunker(input_data)
    return chunker.chunk()


def chunk_with_sections(
    sections: list[tuple[str, str]],  # (section_path, text)
    quoted_spans_map: dict[str, list[Span]] | None = None,
    max_tokens: int = 1600,
    min_tokens: int = 25,
    overlap_tokens: int = 200,
) -> list[ChunkModel]:
    """
    Chunk multiple sections and return unified chunk list.

    Useful for processing documents with multiple logical sections.

    Args:
        sections: List of (section_path, text) tuples
        quoted_spans_map: Optional map of section_path -> quoted_spans
        max_tokens: Maximum tokens per chunk
        min_tokens: Minimum tokens per chunk
        overlap_tokens: Overlap between chunks

    Returns:
        List of all chunks across sections
    """
    all_chunks: list[ChunkModel] = []
    quoted_spans_map = quoted_spans_map or {}

    for section_path, text in sections:
        quoted_spans = quoted_spans_map.get(section_path, [])
        input_data = ChunkingInput(
            text=text,
            section_path=section_path,
            quoted_spans=quoted_spans,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            overlap_tokens=overlap_tokens,
        )
        section_chunks = chunk_text(input_data)
        all_chunks.extend(section_chunks)

    return all_chunks


def estimate_chunk_count(
    text: str, max_tokens: int = 1600, overlap_tokens: int = 200
) -> int:
    """
    Estimate number of chunks that will be produced.

    Useful for progress bars and capacity planning.

    Args:
        text: Text to be chunked
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks

    Returns:
        Estimated chunk count
    """
    if not text:
        return 0

    counter = TokenCounter()
    total_tokens = counter.count(text)

    eff_max_tokens, eff_overlap_tokens = _apply_progressive_scaling(
        total_tokens, max_tokens, overlap_tokens
    )

    effective_step = eff_max_tokens - eff_overlap_tokens
    if effective_step <= 0:
        return 1

    return max(1, (total_tokens + effective_step - 1) // effective_step)
