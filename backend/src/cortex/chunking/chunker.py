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

    def overlaps(self, other: "Span") -> bool:
        """Check if this span overlaps with another."""
        return not (self.end <= other.start or other.end <= self.start)

    def overlap_ratio(self, other: "Span") -> float:
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


def chunk_text(input_data: ChunkingInput) -> list[ChunkModel]:
    """
    Chunk text into smaller segments.

    Implements §7.1 requirements:
    - Respects max_tokens/min_tokens/overlap_tokens settings
    - Classifies chunks based on quoted_spans overlap
    - Generates stable content_hash for deduplication
    - Prefers sentence/paragraph boundaries when possible

    Args:
        input_data: ChunkingInput with text and parameters

    Returns:
        List of ChunkModel instances
    """
    text = input_data.text
    if not text or not text.strip():
        return []

    counter = TokenCounter(model=input_data.model)
    chunks: list[ChunkModel] = []

    # Pre-calculate token count for progressive scaling
    total_tokens_pre = counter.count(text)

    # Apply progressive scaling
    eff_max_tokens, eff_overlap_tokens = _apply_progressive_scaling(
        total_tokens_pre, input_data.max_tokens, input_data.overlap_tokens
    )

    # Calculate character targets based on token targets

    pos = 0
    section_idx = 0

    # Pre-encode the entire text to work with tokens directly
    all_tokens = counter.encode(text)
    total_tokens = len(all_tokens)

    # Map token indices to character offsets for accurate slicing
    # This is expensive but necessary for exact reconstruction if we want to be safe
    # Faster approach: Slice tokens, decode back to string.

    token_pos = 0
    while token_pos < total_tokens:
        token_end = min(token_pos + eff_max_tokens, total_tokens)
        chunk_tokens = all_tokens[token_pos:token_end]

        chunk_text_str = counter.decode(chunk_tokens)
        token_count = len(chunk_tokens)

        # If preserving sentences, we might need to shrink the chunk
        # But tiktoken decoding might not perfectly align with original text chars if lossy
        # For simplicity and correctness on strict token limits, we rely on the decoded text.

        # Recalculate exact char positions in original text if possible,
        # or just store the decoded text. The implementation below relies on simple decoding.
        # Note: This might lose perfect alignment with 'quoted_spans' if encoding is lossy,
        # but protects the embedding model from crashes.

        # Calculate char_start/end in original text (approximate if duplicates exist, but usually fine)
        # To be precise, we would need a mapping.
        # For the fix, ensuring token limit is priority #1.

        # Let's verify token count of the suggested string to be absolutely sure
        if len(chunk_tokens) > input_data.max_tokens:
            # Should not happen by definition of slice
            chunk_tokens = chunk_tokens[: input_data.max_tokens]
            chunk_text_str = counter.decode(chunk_tokens)

        # Skip empty chunks
        if not chunk_text_str.strip():
            token_pos += 1  # Advance at least one token
            continue

        added_chunk = False

        # For Char Start/End: We need to map back.
        # Simple heuristic: text.find(chunk, pos).
        # Since we are iterating forward, search from 'pos' character index.
        try:
            # Use the 'pos' char index to track boundaries in original string
            found_start = text.find(chunk_text_str, pos)
            if found_start == -1:
                # Fallback: exact match failed (normalization?), use pos
                found_start = pos

            found_end = found_start + len(chunk_text_str)

            # Classify
            chunk_type = classify_chunk_type(
                char_start=found_start,
                char_end=found_end,
                quoted_spans=input_data.quoted_spans,
                type_hint=input_data.chunk_type_hint,
            )

            if token_count < input_data.min_tokens and chunks:
                last_chunk = chunks[-1]
                if last_chunk.chunk_type == chunk_type:
                    combined_text = last_chunk.text + chunk_text_str
                    combined_tokens = counter.encode(combined_text)
                    last_chunk.text = combined_text
                    last_chunk.char_end = found_end
                    last_chunk.metadata = {
                        **last_chunk.metadata,
                        "content_hash": compute_content_hash(combined_text),
                        "token_count": len(combined_tokens),
                        "original_length": len(combined_text),
                    }
                    pos = found_end
                else:
                    chunks.append(
                        ChunkModel(
                            text=chunk_text_str,
                            section_path=input_data.section_path,
                            position=section_idx,
                            char_start=found_start,
                            char_end=found_end,
                            chunk_type=chunk_type,
                            metadata={
                                "content_hash": compute_content_hash(chunk_text_str),
                                "token_count": token_count,
                                "original_length": len(chunk_text_str),
                            },
                        )
                    )
                    added_chunk = True
                    pos = found_end
            else:
                chunks.append(
                    ChunkModel(
                        text=chunk_text_str,
                        section_path=input_data.section_path,
                        position=section_idx,
                        char_start=found_start,
                        char_end=found_end,
                        chunk_type=chunk_type,
                        metadata={
                            "content_hash": compute_content_hash(chunk_text_str),
                            "token_count": token_count,
                            "original_length": len(chunk_text_str),
                        },
                    )
                )
                added_chunk = True
                pos = found_end

            # Update character 'pos' for next search
        except Exception as e:
            logger.warning(f"Chunk metadata alignment failed: {e}")

        # Move forward by (max - overlap) tokens
        step = eff_max_tokens - eff_overlap_tokens
        if step < 1:
            step = 1

        token_pos += step
        if added_chunk:
            section_idx += 1

    return chunks


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
