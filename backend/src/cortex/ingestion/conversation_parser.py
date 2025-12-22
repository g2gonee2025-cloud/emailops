"""
Parser for extracting participants from Conversation.txt files.

Handles deduplication across multiple messages within the same conversation.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Pattern to extract message headers
# Format: DATE TIME | From: EMAIL | To: RECIPIENT(S) | Cc: CC_RECIPIENTS
# Simplify: regex reduced from complexity 30 to <20
MESSAGE_HEADER_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s*\|\s*From:\s*([^|]+)(?:\s*\|\s*To:\s*([^|]+))?(?:\s*\|\s*Cc:\s*(.+))?$",
    re.MULTILINE | re.IGNORECASE,
)

# Pattern to extract email addresses from various formats
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# Pattern for Exchange DN format (e.g., /o=exchangelabs/ou=.../cn=recipients/cn=...)
EXCHANGE_DN_PATTERN = re.compile(
    r"/o=[^/]+/ou=[^/]+(?:/cn=[^/\s]+)+",
    re.IGNORECASE,
)


def extract_email_from_field(field: str) -> str | None:
    """
    Extract an email address from a From/To field.

    Handles:
    - Plain email: user@domain.com
    - Display name with email: "John Doe" <john@example.com>
    - Exchange DN: /o=exchangelabs/ou=.../cn=recipients/cn=abc123-username

    Returns the email address or None if not found.
    """
    if not field:
        return None

    field = field.strip()

    # Try to extract standard email first
    email_match = EMAIL_PATTERN.search(field)
    if email_match:
        return email_match.group(0).lower()

    # Check for Exchange DN format - extract the last CN component
    dn_match = EXCHANGE_DN_PATTERN.search(field)
    if dn_match:
        # Extract the last cn= component as an identifier
        dn = dn_match.group(0)
        cn_parts = re.findall(r"/cn=([^/]+)", dn, re.IGNORECASE)
        if cn_parts:
            # Use the last CN as a pseudo-email identifier
            last_cn = cn_parts[-1]
            # Clean up common prefixes like "ho1", "exch", etc.
            if "-" in last_cn:
                parts = last_cn.split("-", 1)
                if len(parts) > 1 and len(parts[1]) > 3:
                    return f"{parts[1]}@exchange.local"
            return f"{last_cn}@exchange.local"

    return None


def extract_participants_from_conversation_txt(text: str) -> list[dict[str, Any]]:
    """
    Extract unique participants from Conversation.txt content.

    Parses header lines in format:
        DATE TIME | From: SENDER | To: RECIPIENT(S) | Cc: CC_RECIPIENTS

    Deduplicates by email address (case-insensitive).

    Args:
        text: Raw content of Conversation.txt

    Returns:
        List of participant dicts with keys: name, smtp, role
    """
    if not text:
        return []

    seen_emails: set[str] = set()
    participants: list[dict[str, Any]] = []

    def add_participant(field: str, role: str) -> None:
        """Helper to add a participant if not already seen."""
        email = extract_email_from_field(field)
        if email and email not in seen_emails:
            seen_emails.add(email)
            participants.append(
                {
                    "name": _extract_display_name(field, email),
                    "smtp": email,
                    "role": role,
                }
            )

    def process_field(field: str | None, role: str) -> None:
        """Process a field that may contain multiple recipients."""
        if not field:
            return
        # Split by common delimiters: semicolon, comma
        parts = re.split(r"[;,]", field)
        for part in parts:
            part = part.strip()
            if part:
                add_participant(part, role)

    # Find all message headers
    for match in MESSAGE_HEADER_PATTERN.finditer(text):
        from_field = match.group(1)
        to_field = match.group(2)
        cc_field = match.group(3)

        # Process From (single sender)
        if from_field:
            add_participant(from_field.strip(), "sender")

        # Process To (may contain multiple recipients)
        process_field(to_field, "recipient")

        # Process Cc (may contain multiple recipients)
        process_field(cc_field, "cc")

    logger.debug(
        "Extracted %d unique participants from Conversation.txt", len(participants)
    )
    return participants


def _extract_display_name(field: str, email: str) -> str:
    """
    Extract display name from a field, defaulting to email prefix if no name found.

    Handles formats like:
    - "John Doe" <john@example.com>
    - John Doe <john@example.com>
    - john@example.com
    """
    # Check for quoted display name
    quoted_match = re.match(r'"([^"]+)"', field)
    if quoted_match:
        return quoted_match.group(1).strip()

    # Check for name before angle bracket
    angle_match = re.match(r"([^<]+)<", field)
    if angle_match:
        name = angle_match.group(1).strip()
        if name and name != email:
            return name

    # Default to email prefix
    if "@" in email:
        prefix = email.split("@")[0]
        # Clean up and capitalize
        return prefix.replace(".", " ").replace("_", " ").title()

    return email
