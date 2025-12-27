"""
Unified Parser for conversation text and participant extraction.
Replaces ad-hoc regex logic in validation.py and email_processing.py.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# Constants & Patterns
# =============================================================================

# Matches standard Conversation.txt header lines:
# 2024-10-07 14:43 | From: ... | To: ...
HEADER_LINE_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}.+?\|\s*From:.+$", re.MULTILINE | re.IGNORECASE
)

# Robust field extractor that handles "Field: Value" within pipe delimiters
FIELD_PATTERN = re.compile(r"(?:^|\|)\s*(From|To|Cc|Bcc):\s*([^|\n\r]+)", re.IGNORECASE)

# Exchange Distinguished Name (Legacy Support)
EXCHANGE_DN_PATTERN = re.compile(r"/o=[^/]+/ou=[^/]+(?:/cn=[^/\s]+)+", re.IGNORECASE)

# RFC 5322 Email Pattern (Simplified for text extraction)
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", re.IGNORECASE
)


def extract_participants_from_conversation_txt(text: str) -> List[Dict[str, Any]]:
    """
    The Single Source of Truth for extracting participants from raw text.
    Used by: Ingestion, Validation, and Backfills.

    Args:
        text: Raw content of Conversation.txt

    Returns:
        List of participant dicts with keys: name, smtp, role
    """
    if not text:
        return []

    participants: Dict[str, Dict[str, Any]] = {}

    # 1. Scan for header lines
    for line in HEADER_LINE_PATTERN.findall(text):
        _parse_header_line(line, participants)

    # 2. Convert to sorted list
    return sorted(list(participants.values()), key=lambda x: x["smtp"])


def _split_recipients(value: str) -> List[str]:
    """
    Split recipient string by comma or semicolon, respecting quoted strings.
    E.g. '"Doe, John" <j@d.com>, Jane' -> ['"Doe, John" <j@d.com>', 'Jane']
    """
    # Regex explanation:
    # 1. Quoted string: "..." (non-capturing group for content)
    # 2. OR Non-separator chars: [^;,]+
    # Pattern finds all chunks that are either a quoted string or non-delimiters
    # This is complex to do purely with split, so we verify a simple state machine approach
    # or a robust regex findall.

    # Robust regex approach:
    # Match either:
    # - A quoted string: "  (?:[^"\\]|\\.)*  "
    # - A non-special char sequence: [^;,"]+
    # Then join them until we hit a delimiter? No.

    # Simpler: Use a state-machine parser for robustness
    if not value:
        return []

    parts = []
    current = []
    in_quote = False

    for char in value:
        if char == '"':
            in_quote = not in_quote
            current.append(char)
        elif char in (",", ";") and not in_quote:
            # Delimiter found outside quote
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    if current:
        parts.append("".join(current).strip())

    return [p for p in parts if p]


def _sanitize_local_part(local: str) -> str:
    """Sanitize an email local-part to ensure it is RFC-compliant enough for our use."""
    s = (local or "").strip().lower()
    # Allowed characters per RFC 5322 (unquoted) local-part
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789!#$%&'*+/=?^_`{|}~.-")
    s = "".join(c if c in allowed else "." for c in s)
    # Collapse multiple dots and trim from ends
    s = re.sub(r"\.+", ".", s).strip(".")
    return s or "unknown"


def _is_exchange_dn(value: str) -> bool:
    """Heuristically detect Exchange/X.500 Distinguished Names."""
    if not value:
        return False
    v = value.strip()
    u = v.upper()
    return (
        u.startswith(("EX:", "X500:", "/O=", "/OU=", "/CN=", "O=", "OU=", "CN="))
        or "/CN=" in u
    )


def _extract_last_cn(dn: str) -> str:
    """Extract the last CN component from a DN string."""
    if not dn:
        return ""
    matches = re.findall(r"(?i)CN=([^/]+)", dn)
    if matches:
        return str(matches[-1])
    # Fallback: take last segment and strip any key=
    seg = dn.rsplit("/", 1)[-1]
    return seg.split("=", 1)[-1] if "=" in seg else seg


def _parse_header_line(line: str, participants: Dict[str, Dict[str, Any]]) -> None:
    """Parse a single header line and update participants dict."""
    for match in FIELD_PATTERN.finditer(line):
        role_raw, value = match.groups()
        role = _normalize_role(role_raw)

        # Use robust splitter
        raw_addrs = _split_recipients(value)

        for raw_addr in raw_addrs:
            # Convert Exchange DNs into sanitized pseudo-emails to avoid malformed addresses
            if isinstance(raw_addr, str) and _is_exchange_dn(raw_addr):
                last_cn = _extract_last_cn(raw_addr)
                raw_addr = f"{_sanitize_local_part(last_cn)}@exchange.local"
            email, name = _extract_email_and_name(raw_addr)
            if email and email not in participants:
                participants[email] = {
                    "name": name,
                    "smtp": email,
                    "role": role,  # Note: Role is first-seen basis
                }


def _extract_email_and_name(raw: str) -> Tuple[str | None, str]:
    """
    Extract clean email and display name from raw string.
    Handles: "Name <email>", "email", and Exchange DNs.
    """
    # 1. Try standard email extraction
    email_match = EMAIL_PATTERN.search(raw)
    if email_match:
        email = email_match.group(0).lower()
        # Remove email from raw to detect name
        name_candidate = (
            raw.replace(email, "").replace("<", "").replace(">", "").strip('" ').strip()
        )
        name = name_candidate if name_candidate else email.split("@")[0]
        return email, _clean_name(name)

    # 2. Try Exchange DN
    dn_match = EXCHANGE_DN_PATTERN.search(raw)
    if dn_match:
        # Pseudo-email extraction from CN
        dn = dn_match.group(0)
        cn_parts = re.findall(r"/cn=([^/]+)", dn, re.IGNORECASE)
        if cn_parts:
            last_cn = cn_parts[-1]
            # Heuristic: Usernames often follow "recipients/cn="
            pseudo_email = f"{last_cn}@exchange.local".lower()
            return pseudo_email, _clean_name(last_cn)

    return None, ""


def _clean_name(name: str) -> str:
    """Normalize display name."""
    if not name:
        return ""
    # Remove surrounding quotes, flip "Last, First"???
    # For now, just title case and strip
    return name.title().strip()


def _normalize_role(role: str) -> str:
    role = role.lower()
    if role == "from":
        return "sender"
    if role == "to":
        return "recipient"
    return role  # cc, bcc


__all__ = "extract_participants_from_conversation_txt"
