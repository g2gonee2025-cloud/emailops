"""
Quoted text masking.

Implements §6.3 of the Canonical Blueprint.
"""

from __future__ import annotations

from cortex.chunking.chunker import Span


def detect_quoted_spans(text: str) -> list[Span]:
    """
    Identify quotes & signatures (Talon-like logic).

    Returns list of {start: int, end: int} spans.
    """
    spans: list[Span] = []

    lines = text.splitlines()
    current_span_start = None

    for i, line in enumerate(lines):
        lines[i] = line.strip()
        is_quoted = (
            line.startswith(">")
            or (line.startswith("On ") and line.endswith("wrote:"))
            or line == "--"
        )

        if is_quoted:
            if current_span_start is None:
                # Start of a new span
                # Calculate character offset (approximate, as splitlines consumes newlines)
                # A robust implementation would track character indices.
                # For now, we'll use a simplified line-based approach or just return empty if complex.
                # Let's try to be reasonably accurate by reconstructing text up to this line.
                # This is slow O(N^2) for large texts, but acceptable for emails.
                # Optimization: Keep running length.
                pass

    # Re-implementation with character tracking
    char_idx = 0
    in_quote = False

    # Common quote headers
    # On [Date], [Name] wrote:
    # -----Original Message-----
    # From: ...

    lines_with_endings = text.splitlines(keepends=True)

    for line in lines_with_endings:
        stripped = line.strip()
        is_quote_line = (
            stripped.startswith(">")
            or (stripped.startswith("On ") and stripped.endswith("wrote:"))
            or stripped == "--"
            or stripped.startswith("-----Original Message-----")
            or (stripped.startswith("From: ") and "@" in stripped)  # Stricter heuristic
        )

        line_len = len(line)

        if is_quote_line:
            if not in_quote:
                in_quote = True
                current_span_start = char_idx
        else:
            if in_quote:
                # End of quote span?
                # Often quotes are blocks. A single non-quote line might be a wrap.
                # Strict mode: end span.
                # Relaxed mode: allow some gaps.
                # Let's use strict for now.
                in_quote = False
                spans.append(Span(start=current_span_start, end=char_idx))
                current_span_start = None

        char_idx += line_len

    if in_quote and current_span_start is not None:
        spans.append(Span(start=current_span_start, end=char_idx))

    return spans
