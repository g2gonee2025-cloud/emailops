import pytest
from cortex.chunking.chunker import Span
from cortex.ingestion.quoted_masks import detect_quoted_spans

# Test cases for various email formats
TEST_CASES = [
    # Simple email with a standard quote
    (
        "Hi team,\n\nPlease see below.\n\n> This is a quote.\n\nThanks,\nJules",
        [Span(start=29, end=49)],
        "standard_quote",
    ),
    # Email with a forwarded message header and trailing newlines
    (
        "FYI\n\n-----Original Message-----\nFrom: test@example.com\nSubject: Hello\n\nBody",
        [Span(start=5, end=71)],
        "forwarded_message",
    ),
    # Email with a signature separator
    (
        "Here is the update.\n\n--\nJules Verne\nArchitect",
        [Span(start=21, end=24)],
        "signature",
    ),
    # Complex email with mixed content and adjacent lines
    (
        "Hello,\n\n> Quoted line 1\n> Quoted line 2\n\nOn Fri, Jan 1, 2024 at 12:00 PM, wrote:\n\n--\nSignature",
        [Span(start=8, end=85)],
        "mixed_content",
    ),
    # Email with no quoted text
    ("This is a simple email with no quotes.", [], "no_quotes"),
    # Email with adjacent quoted lines that should be merged (no trailing newline)
    ("> Line 1\n> Line 2\n> Line 3", [Span(start=0, end=26)], "adjacent_quotes"),
]


@pytest.mark.parametrize("text, expected_spans, case_id", TEST_CASES)
def test_detect_quoted_spans(text, expected_spans, case_id):
    """
    Tests the detect_quoted_spans function with various email formats.
    """
    result = detect_quoted_spans(text)
    assert result == expected_spans, f"Failed on case: {case_id}"
