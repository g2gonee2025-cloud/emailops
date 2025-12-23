"""
Legacy Email Processing Logic.

Archived from cortex.email_processing.py.
Contains unused functions: extract_email_metadata, split_email_thread.
"""

from __future__ import annotations

import re
import warnings
from functools import wraps
from typing import Any


# Re-implement deprecated decorator here for self-containment
def deprecated(reason):
    def decorator(func):
        @wraps(func)
        def new_func(*args, **kwargs):
            warnings.warn(
                f"Call to deprecated function {func.__name__}. {reason}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return new_func

    return decorator


@deprecated(
    "Use cortex.ingestion.conversation_parser.extract_participants_from_conversation_txt instead."
)
def extract_email_metadata(text: str) -> dict[str, Any]:
    """
    Extract email metadata from header block.
    DEPRECATED: Use cortex.ingestion.conversation_parser instead.
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
        # Use local extraction for thread splitting context
        # Suppress warning here as it's internal usage
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
