"""
Email Processing.

Implements email cleaning and thread splitting.
Blueprint §6.2: Email text cleaning and metadata extraction.
"""

from __future__ import annotations

import re

from cortex.utils import strip_control_chars

# =============================================================================
# Pre-compiled Regex Patterns
# =============================================================================


# Email and URL patterns for redaction
# EMAIL_REGEX is used for validation (anchored). For extraction, we need a similar pattern but unanchored.
# We'll use a local pattern for extraction but aligned with the validation one conceptually.
_EMAIL_SEARCH_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
_URL_FULLMATCH_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
_WWW_DOMAIN_PATTERN = re.compile(r"www\.[A-Za-z0-9.-]+")

# Constants
SIGNATURE_LOOKBACK_CHARS = 2000
MIN_TEXT_LENGTH_FOR_SIGNATURE = 100
MAX_BOILERPLATE_LINE_LENGTH = 80

# Excessive punctuation normalization
# Replacing with single char or standard marker instead of stripping entirely
_EXCESSIVE_DOTS = re.compile(r"\.{3,}")
_EXCESSIVE_EXCLAIM = re.compile(r"!{2,}")
_EXCESSIVE_QUESTION = re.compile(r"\?{2,}")
_EXCESSIVE_EQUALS = re.compile(r"={3,}")
_EXCESSIVE_DASHES = re.compile(r"-{5,}")
_EXCESSIVE_UNDERSCORES = re.compile(r"_{5,}")

# Whitespace normalization
_MULTIPLE_SPACES = re.compile(r"[ \t]{2,}")
_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
# Matches lines that are empty or contain only whitespace
_BLANK_LINES = re.compile(r"^\s*$", re.MULTILINE)

# Quoted reply lines (starts with >)
_QUOTED_REPLY = re.compile(r"^>+\s?(.*)$", re.MULTILINE)

# Header patterns to remove
# Includes standard RFC 822 headers + Conversation.txt-specific formats + i18n
_HEADER_PATTERNS = [
    # Conversation.txt-specific: "2024-10-07 14:43 | From: ..."
    re.compile(
        r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+\|\s+From:.*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(r"^From:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^To:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Cc:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Bcc:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Subject:.*$", re.MULTILINE | re.IGNORECASE),
    # Chinese subject header
    re.compile(r"^主题:.*$", re.MULTILINE | re.IGNORECASE),
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

# Boilerplate keywords for line-level filtering (merged from clean_conversation.py)
# These are checked case-insensitively against each line
_BOILERPLATE_KEYWORDS = [
    # English disclaimers
    "Email from external sender",
    "This message, and its attachments",
    "This communication is confidential",
    "legal notice",
    "confidentiality notice",
    "Please consider the environment before printing",
    "Go Green, Avoid Printing",
    # French disclaimers (i18n support)
    "Ce message, et ses éventuelles pièces jointes",
    "Ce message a été classifié",
    "L'Internet ne permettant pas d'assurer l'intégrité de ce Message",
    "Classification : Interne",
    "Classification : Internal",
    "Classification : Public",
    # Organization-specific (can be extended)
    "A member of the Nasco Insurance Group",
]

# Domain noise - lines that are just marketing URLs/domains
_DOMAIN_NOISE = [
    "CHALHOUBGROUP.COM",
    "CAREERS.CHALHOUBGROUP.COM",
]

# Forwarding separator patterns
_FORWARDING_PATTERNS = [
    re.compile(r"^-{3,}\s*Original Message\s*-{3,}", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^-{3,}\s*Forwarded Message\s*-{3,}", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^_{10,}", re.MULTILINE),  # Long underscore lines
    re.compile(r"^Begin forwarded message:", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^On .+ wrote:$", re.MULTILINE),  # "On [date], [person] wrote:"
]


def _is_boilerplate_line(line: str) -> bool:
    """
    Check if a line is boilerplate/noise that should be removed.

    Consolidated from clean_conversation.py to avoid logic forks.
    """
    stripped = line.strip()
    if not stripped:
        return False

    # Single URL lines
    if _URL_FULLMATCH_PATTERN.fullmatch(stripped):
        return True

    # Domain noise
    if stripped.upper() in _DOMAIN_NOISE:
        return True

    # Bare www domains
    if _WWW_DOMAIN_PATTERN.fullmatch(stripped):
        return True

    # Check against boilerplate keywords
    lower = stripped.lower()
    for keyword in _BOILERPLATE_KEYWORDS:
        if keyword.lower() in lower:
            return True

    # Very long address-like lines (signature blocks)
    if (
        len(stripped) > MAX_BOILERPLATE_LINE_LENGTH
        and stripped.count(",") >= 3
        and "@" not in stripped
    ):
        return True

    return False


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

    # 6. Remove boilerplate lines (disclaimers, domain noise, etc.)
    # Consolidated from clean_conversation.py to avoid logic forks
    lines = text.split("\n")
    lines = [line for line in lines if not _is_boilerplate_line(line)]
    text = "\n".join(lines)

    # 7. Redact emails and URLs
    text = _EMAIL_SEARCH_PATTERN.sub("[email]", text)
    text = _URL_PATTERN.sub("[URL]", text)

    # 8. Normalize punctuation
    text = _EXCESSIVE_DOTS.sub("...", text)
    text = _EXCESSIVE_EXCLAIM.sub("!", text)
    text = _EXCESSIVE_QUESTION.sub("?", text)
    text = _EXCESSIVE_EQUALS.sub("", text)
    text = _EXCESSIVE_DASHES.sub("---", text)
    text = _EXCESSIVE_UNDERSCORES.sub("", text)

    # 9. Normalize whitespace
    text = _MULTIPLE_SPACES.sub(" ", text)
    text = _MULTIPLE_NEWLINES.sub("\n\n", text)

    # 10. Strip control characters
    text = strip_control_chars(text)

    # 11. Final trim
    return text.strip()


def _strip_signature(text: str) -> str:
    """
    Remove email signature from the end of the text.

    Checks the last 2000 characters for signature patterns and removes
    everything after the first match.
    """
    if len(text) < MIN_TEXT_LENGTH_FOR_SIGNATURE:
        return text

    # Look at last N chars for signature
    tail_start = max(0, len(text) - SIGNATURE_LOOKBACK_CHARS)
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


__all__ = ("clean_email_text",)
