"""
Quoted text masking.
Implements §6.3 of the Canonical Blueprint.
"""
from __future__ import annotations
import re
from typing import List
from cortex.chunking.chunker import Span

# Regex to detect quoted lines, forwarded messages, and signatures.
# This is a single, efficient regex that can identify multiple patterns at once.
# - `^>.*$`: Matches lines starting with ">" (standard email quote).
# - `^On .* wrote:$`: Matches lines like "On {date}, {person} wrote:".
# - `^--\s*$`: Matches signature separators like "--" or "-- ".
# - `^-----Original Message-----$`: Matches forwarded message headers.
# - `From: .*@.*`: Matches "From:" lines that likely contain an email address (stricter).
# - `Sent: .*$`: Matches "Sent:" lines.
# - `To: .*@.*$`: Matches "To:" lines that likely contain an email address (stricter).
# - `Subject: .*$`: Matches "Subject:" lines.
QUOTE_PATTERNS = re.compile(
    r"^(>.*|On .* wrote:|--\s*|-----Original Message-----|From: .*@.*|Sent: .*|To: .*@.*|Subject: .*)$",
    re.MULTILINE | re.IGNORECASE,
)


def detect_quoted_spans(text: str) -> List[Span]:
    """
    Identifies and merges quoted text, forwarded messages, and signatures
    in an email body using a single-pass regex.
    Returns a list of `Span` objects with start and end character offsets.
    """
    spans = [Span(start=match.start(), end=match.end()) for match in QUOTE_PATTERNS.finditer(text)]
    if not spans:
        return []

    # Merge overlapping or adjacent spans
    merged_spans: List[Span] = []
    current_span = spans[0]

    for next_span in spans[1:]:
        gap_text = text[current_span.end:next_span.start]
        if not gap_text.strip():  # If the gap is only whitespace
            current_span.end = max(current_span.end, next_span.end)
        else:
            merged_spans.append(current_span)
            current_span = next_span

    merged_spans.append(current_span)

    # After merging, greedily consume all trailing whitespace for each span block
    # to ensure the entire quoted section is included.
    for span in merged_spans:
        end = span.end
        while end < len(text) and text[end].isspace():
            end += 1
        span.end = end

    return merged_spans
