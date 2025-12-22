"""
Email Processing.

Implements email cleaning and thread splitting.
Blueprint §6.2: Email text cleaning and metadata extraction.
"""

from __future__ import annotations

import re
from typing import Any

from cortex.utils import strip_control_chars

# =============================================================================
# Pre-compiled Regex Patterns
# =============================================================================


# Email and URL patterns for redaction
# EMAIL_REGEX is used for validation (anchored). For extraction, we need a similar pattern but unanchored.
# We'll use a local pattern for extraction but aligned with the validation one conceptually.
_EMAIL_SEARCH_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)
_URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')

# Excessive punctuation normalization
_EXCESSIVE_DOTS = re.compile(r"\.{3,}")
_EXCESSIVE_EXCLAIM = re.compile(r"!{2,}")
_EXCESSIVE_QUESTION = re.compile(r"\?{2,}")
_EXCESSIVE_EQUALS = re.compile(r"={3,}")
_EXCESSIVE_DASHES = re.compile(r"-{5,}")
_EXCESSIVE_UNDERSCORES = re.compile(r"_{5,}")

# Whitespace normalization
_MULTIPLE_SPACES = re.compile(r"[ \t]{2,}")
_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
_BLANK_LINES = re.compile(r"^\s*$", re.MULTILINE)

# Quoted reply lines (starts with >)
_QUOTED_REPLY = re.compile(r"^>+\s?(.*)$", re.MULTILINE)

# Header patterns to remove
# Header patterns to remove
# Simplified to avoid backtracking warnings (s5852)
_HEADER_PATTERNS = [
    re.compile(r"^From:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^To:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Cc:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Bcc:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Subject:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Date:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Sent:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Reply-To:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Message-ID:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^In-Reply-To:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^References:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Content-Type:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^MIME-Version:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^X-[a-z-]+:.*$", re.MULTILINE | re.IGNORECASE),
]

# Signature patterns (checked in last 2000 chars)
_SIGNATURE_PATTERNS = [
    re.compile(r"^--\s*$", re.MULTILINE),  # Standard signature delimiter
    re.compile(r"^Best\s*regards?,?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Kind\s*regards?,?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Warm\s*regards?,?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Regards?,?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Sincerely,?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Thanks?,?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Thank\s*you,?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Cheers,?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Sent from my iPhone", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Sent from my iPad", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Sent from my Samsung", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Get Outlook for", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Sent from Mail for Windows", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^This email and any attachments", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^CONFIDENTIALITY NOTICE", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^DISCLAIMER:", re.MULTILINE | re.IGNORECASE),
]

# Forwarding separator patterns
_FORWARDING_PATTERNS = [
    re.compile(r"^-{3,}\s*Original Message\s*-{3,}", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^-{3,}\s*Forwarded Message\s*-{3,}", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^_{10,}", re.MULTILINE),  # Long underscore lines
    re.compile(r"^Begin forwarded message:", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^On .+ wrote:$", re.MULTILINE),  # "On [date], [person] wrote:"
]


# =============================================================================
# Core Functions
# =============================================================================


def clean_email_text(text: str) -> str:
    """
    Clean email text by removing headers, signatures, and excessive whitespace.

    Blueprint §6.2:
    * BOM handling
    * Header removal
    * Signature/footer stripping
    * Forwarding separator removal
    * Quoted reply removal (optional, configurable)
    * Redaction of emails/URLs
    * Whitespace normalization
    * Control character stripping

    Args:
        text: Raw email body text

    Returns:
        Cleaned text ready for indexing
    """
    if not text:
        return ""

    # 1. Handle BOM
    if text.startswith("\ufeff"):
        text = text[1:]

    # 2. Remove common headers
    for pattern in _HEADER_PATTERNS:
        text = pattern.sub("", text)

    # 3. Strip signatures (check last 2000 chars)
    text = _strip_signature(text)

    # 4. Remove forwarding separators
    for pattern in _FORWARDING_PATTERNS:
        text = pattern.sub("", text)

    # 5. Remove quoted reply markers (lines starting with >)
    # Note: We keep the content but remove the > markers for cleaner text
    # In some cases we might want to preserve quoted context - this is configurable
    # For indexing, we typically want the raw content without markers
    text = _QUOTED_REPLY.sub(r"\1", text)

    # 6. Redact emails and URLs
    text = _EMAIL_SEARCH_PATTERN.sub("[email]", text)
    text = _URL_PATTERN.sub("[URL]", text)

    # 7. Normalize punctuation
    text = _EXCESSIVE_DOTS.sub("...", text)
    text = _EXCESSIVE_EXCLAIM.sub("!", text)
    text = _EXCESSIVE_QUESTION.sub("?", text)
    text = _EXCESSIVE_EQUALS.sub("", text)
    text = _EXCESSIVE_DASHES.sub("---", text)
    text = _EXCESSIVE_UNDERSCORES.sub("", text)

    # 8. Normalize whitespace
    text = _MULTIPLE_SPACES.sub(" ", text)
    text = _MULTIPLE_NEWLINES.sub("\n\n", text)

    # 9. Strip control characters
    text = strip_control_chars(text)

    # 10. Final trim
    return text.strip()


def _strip_signature(text: str) -> str:
    """
    Remove email signature from the end of the text.

    Checks the last 2000 characters for signature patterns and removes
    everything after the first match.
    """
    if len(text) < 100:
        return text

    # Look at last 2000 chars for signature
    tail_start = max(0, len(text) - 2000)
    tail = text[tail_start:]

    earliest_match = len(tail)

    for pattern in _SIGNATURE_PATTERNS:
        match = pattern.search(tail)
        if match and match.start() < earliest_match:
            earliest_match = match.start()

    if earliest_match < len(tail):
        # Found a signature - truncate
        return text[: tail_start + earliest_match].rstrip()

    return text


def extract_email_metadata(text: str) -> dict[str, Any]:
    """
    Extract email metadata from header block.

    Args:
        text: Email text with headers

    Returns:
        Dict with sender, recipients, date, subject, cc, bcc
    """
    # Split at first double newline to get header block
    parts = text.split("\n\n", 1)
    header_block = parts[0] if parts else ""

    # Unfold folded headers (lines starting with whitespace)
    header_block = re.sub(r"\n[ \t]+", " ", header_block)

    result: dict[str, Any] = {
        "sender": None,
        "recipients": [],
        "date": None,
        "subject": None,
        "cc": [],
        "bcc": [],
    }

    # Extract From
    from_match = re.search(r"^From:.*$", header_block, re.MULTILINE | re.IGNORECASE)
    if from_match:
        result["sender"] = from_match.group(0)[5:].strip()

    # Extract To (may have multiple addresses)
    to_match = re.search(r"^To:.*$", header_block, re.MULTILINE | re.IGNORECASE)
    if to_match:
        # Skip "To:" prefix (3 chars)
        result["recipients"] = [
            addr.strip() for addr in to_match.group(0)[3:].split(",")
        ]

    # Extract Cc
    cc_match = re.search(r"^Cc:.*$", header_block, re.MULTILINE | re.IGNORECASE)
    if cc_match:
        result["cc"] = [addr.strip() for addr in cc_match.group(0)[3:].split(",")]

    # Extract Bcc
    bcc_match = re.search(r"^Bcc:.*$", header_block, re.MULTILINE | re.IGNORECASE)
    if bcc_match:
        result["bcc"] = [addr.strip() for addr in bcc_match.group(0)[4:].split(",")]

    # Extract Date (try both Date: and Sent:)
    date_match = re.search(
        r"^(?:Date|Sent):.*$", header_block, re.MULTILINE | re.IGNORECASE
    )
    if date_match:
        # Split by colon to get value
        parts = date_match.group(0).split(":", 1)
        if len(parts) > 1:
            result["date"] = parts[1].strip()

    # Extract Subject
    subject_match = re.search(
        r"^Subject:.*$", header_block, re.MULTILINE | re.IGNORECASE
    )
    if subject_match:
        result["subject"] = subject_match.group(0)[8:].strip()

    return result


def split_email_thread(text: str) -> list[str]:
    """
    Split an email thread into individual messages.

    Uses common reply/forward separators to split the thread,
    then attempts to sort chronologically by Date header.

    Args:
        text: Full email thread text

    Returns:
        List of individual message texts, oldest first
    """
    if not text:
        return []

    # Split on common separators
    separators = [
        r"-{3,}\s*Original Message\s*-{3,}",
        r"-{3,}\s*Forwarded Message\s*-{3,}",
        r"_{10,}",
        r"On .+ wrote:",
        r"Begin forwarded message:",
    ]

    # Build combined pattern
    combined = "|".join(f"({sep})" for sep in separators)

    # Split but keep separators
    parts = re.split(f"({combined})", text, flags=re.IGNORECASE | re.MULTILINE)

    # Clean up parts - remove empty and separator-only parts
    messages = []
    current = ""
    for part in parts:
        if part is None:
            continue
        part = part.strip()
        if not part:
            continue
        # Check if this is a separator
        is_sep = any(re.match(sep, part, re.IGNORECASE) for sep in separators)
        if is_sep:
            if current:
                messages.append(current.strip())
                current = ""
        else:
            current = part if not current else current + "\n\n" + part

    if current:
        messages.append(current.strip())

    # Try to sort chronologically
    messages_with_dates = []
    for msg in messages:
        meta = extract_email_metadata(msg)
        date_str = meta.get("date")
        messages_with_dates.append((msg, date_str))

    # Only sort if we have dates for most messages
    dated_count = sum(1 for _, d in messages_with_dates if d)
    if dated_count >= len(messages) // 2:
        # Parse dates and sort (basic heuristic - keep original order if parsing fails)
        try:
            from dateutil import parser as date_parser

            def parse_date(d):
                if not d:
                    return None
                try:
                    return date_parser.parse(d)
                except Exception:
                    return None

            messages_with_dates.sort(key=lambda x: parse_date(x[1]) or x[1] or "")
        except ImportError:
            pass  # No dateutil, keep original order

    return [msg for msg, _ in messages_with_dates]


__all__ = ["clean_email_text", "extract_email_metadata", "split_email_thread"]
