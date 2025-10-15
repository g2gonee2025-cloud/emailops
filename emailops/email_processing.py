"""
Email processing utilities.
Handles email cleaning, metadata extraction, and thread splitting.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for email processing
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})")
_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+")

# MEDIUM #20: Pre-compiled patterns for performance - all patterns compiled at module load
_EXCESSIVE_EQUALS = re.compile(r"[=\-_*]{10,}")
_EXCESSIVE_DOTS = re.compile(r"\.{4,}")
_EXCESSIVE_EXCLAMATION = re.compile(r"\!{2,}")
_EXCESSIVE_QUESTION = re.compile(r"\?{2,}")
_MULTIPLE_SPACES = re.compile(r"[ \t]+")
_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
_BLANK_LINES = re.compile(r"(?m)^\s*$")
_QUOTED_REPLY = re.compile(r"(?m)^\s*>+\s?")

# Email header and signature patterns
_HEADER_PATTERNS = [
    re.compile(p)
    for p in [
        r"(?mi)^(From|Sent|To|Subject|Cc|Bcc|Date|Reply-To|Message-ID|In-Reply-To|References):.*$",
        r"(?mi)^(Importance|X-Priority|X-Mailer|Content-Type|MIME-Version):.*$",
        r"(?mi)^(Thread-Topic|Thread-Index|Accept-Language|Content-Language):.*$",
    ]
]

_SIGNATURE_PATTERNS = [
    re.compile(p, re.MULTILINE)
    for p in [
        r"(?si)^--\s*\n.*",  # traditional signature delimiter
        r"(?si)^\s*best regards.*?$",
        r"(?si)^\s*kind regards.*?$",
        r"(?si)^\s*sincerely.*?$",
        r"(?si)^\s*thanks.*?$",
        r"(?si)^sent from my.*?$",
        r"(?si)\*{3,}.*?confidential.*?\*{3,}",
        r"(?si)this email.*?intended recipient.*?$",
    ]
]

_FORWARDING_PATTERNS = [
    re.compile(p)
    for p in [
        r"(?m)^-{3,}\s*Original Message\s*-{3,}.*?$",
        r"(?m)^-{3,}\s*Forwarded Message\s*-{3,}.*?$",
        r"(?m)^_{10,}.*?$",
    ]
]

# MEDIUM #24: Import control char pattern from centralized file_utils instead of duplicating
from .file_utils import _strip_control_chars


def clean_email_text(text: str) -> str:
    """
    Clean email body text for indexing and retrieval.

    The function is intentionally conservative to avoid removing
    substantive content. It primarily:
    - removes common header lines (From, To, Subject, etc.)
    - strips simple signatures / legal footers (last ~2k chars only)
    - removes quoted reply markers (> prefixes)
    - removes forwarding separators
    - redacts email addresses → [email@domain]
    - redacts URLs → [URL]
    - normalizes excessive punctuation and whitespace
    """
    if not text:
        return ""

    # Handle BOM
    if text.startswith("\ufeff"):
        text = text[1:]

    # Remove obvious header lines (keep body content intact)
    for pattern in _HEADER_PATTERNS:
        text = pattern.sub("", text)

    # Remove simple signatures/footers ONLY from the trailing portion
    tail = text[-2000:]  # examine last ~2k chars
    for pattern in _SIGNATURE_PATTERNS:
        tail = pattern.sub("", tail)
    text = text[:-2000] + tail if len(text) > 2000 else tail

    # Remove forwarding separators
    for pattern in _FORWARDING_PATTERNS:
        text = pattern.sub("", text)

    # Remove '>' quoting markers and normalize noise using pre-compiled patterns
    text = re.sub(r"(?m)^\s*>+\s?", "", text)
    text = _EMAIL_PATTERN.sub(r"[email@\1]", text)
    text = _URL_PATTERN.sub("[URL]", text)
    text = _EXCESSIVE_EQUALS.sub("", text)
    text = _EXCESSIVE_DOTS.sub("...", text)
    text = _EXCESSIVE_EXCLAMATION.sub("!", text)
    text = _EXCESSIVE_QUESTION.sub("?", text)
    text = _MULTIPLE_SPACES.sub(" ", text)
    text = _MULTIPLE_NEWLINES.sub("\n\n", text)
    text = _BLANK_LINES.sub("", text)  # drop blank-only lines
    return _strip_control_chars(text).strip()


def extract_email_metadata(text: str) -> dict[str, Any]:
    """
    Extract structured metadata from raw RFC-822 style headers in text.

    Heuristics only; unfolds folded headers and supports Bcc.
    Returns dict with keys: sender, recipients, date, subject, cc, bcc
    """
    md: dict[str, Any] = {
        "sender": None,
        "recipients": [],
        "date": None,
        "subject": None,
        "cc": [],
        "bcc": [],
    }

    if not text:
        return md

    # Consider only the header preamble (before the first blank line)
    header_block, *_ = text.split("\n\n", 1)
    # Normalize newlines then unfold (RFC 5322): CRLF followed by WSP -> space
    header_block = header_block.replace("\r\n", "\n").replace("\r", "\n")
    header_block = re.sub(r"\n[ \t]+", " ", header_block)

    def _get(h: str) -> str | None:
        m = re.search(rf"(?mi)^{re.escape(h)}:\s*(.+?)$", header_block)
        return m.group(1).strip() if m else None

    if v := _get("From"):
        md["sender"] = v
    if v := _get("To"):
        md["recipients"] = [x.strip() for x in v.split(",") if x.strip()]
    if v := _get("Cc"):
        md["cc"] = [x.strip() for x in v.split(",") if x.strip()]
    if v := _get("Bcc"):
        md["bcc"] = [x.strip() for x in v.split(",") if x.strip()]
    if (v := _get("Date")) or (v := _get("Sent")):
        md["date"] = v
    if v := _get("Subject"):
        md["subject"] = v

    return md


def split_email_thread(text: str) -> list[str]:
    """
    Split an email thread into individual messages.

    Heuristics:
    - Use common "Original/Forwarded Message" separators and "On ... wrote:" lines
    - As a tie-breaker, if multiple message blocks contain a 'Date:' header,
      sort the blocks chronologically; otherwise preserve input order.

    Returns:
        List of message bodies in chronological order (oldest -> newest) when possible.
    """
    if not text or not text.strip():
        return []

    # Common separators
    sep = re.compile(
        r"(?mi)^(?:-{3,}\s*Original Message\s*-{3,}|"
        r"-{3,}\s*Forwarded Message\s*-{3,}|"
        r"On .+? wrote:|"
        r"_{10,})\s*$"
    )

    parts = [p.strip() for p in sep.split(text) if p and p.strip()]
    if len(parts) <= 1:
        return [text.strip()]

    # If multiple parts have recognizable Date headers, sort them
    def _parse_date(s: str) -> datetime | None:
        m = re.search(r"(?mi)^Date:\s*(.+?)$", s)
        if not m:
            return None
        try:
            return parsedate_to_datetime(m.group(1))
        except Exception:
            return None

    dated: list[tuple[datetime, str]] = []
    undated: list[str] = []
    for p in parts:
        d = _parse_date(p)
        if d:
            dated.append((d, p))
        else:
            undated.append(p)
    if len(dated) >= 2:
        dated.sort(key=lambda x: x[0])
        ordered = [p for _, p in dated] + undated
    else:
        ordered = parts

    return ordered
