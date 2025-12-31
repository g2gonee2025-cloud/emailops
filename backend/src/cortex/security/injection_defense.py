"""
Injection Defense Module.

Implements detection against prompt injection attacks by flagging
common jailbreak patterns and override attempts.
"""

import re
import unicodedata

import structlog
from cortex.common.exceptions import SecurityError

logger = structlog.get_logger(__name__)

# Enhanced patterns to detect more sophisticated injection attempts
BLOCKLIST_PATTERNS = [
    # Original patterns
    r"ignore previous instructions",
    r"ignore all instructions",
    r"forget all previous instructions",
    r"you are now (?:a|an|in) [\s\S]*",
    r"roleplay as",
    r"system override",
    r"developer mode",
    r"god mode",
    r"jailbreak",
    r"dan mode",
    r"ignore your previous instructions",
    r"new rule:",
    r"important instruction:",
    r"ignore all prior instructions",
    r"disregard previous directions",
    r"start a new conversation",
    r"act as a",
    r"simulate a",
    r"your new persona is",
    # Sophisticated patterns
    # Detects character separators and noise, e.g., "i g n o r e"
    r"i\s*g\s*n\s*o\s*r\s*e",
    # Detects role-playing variations
    r"from now on, you are",
    r"you will now act as",
    # Detects context shifting
    r"what if someone said",
    # Detects direct commands
    r"(?:print|output) the following (?:system|developer)? ?(?:prompt|instructions)",
]

_BLOCKLIST_REGEX = re.compile("|".join(BLOCKLIST_PATTERNS), re.IGNORECASE)

_HOMOGLYPH_MAP = str.maketrans(
    {
        "\u0430": "a",  # Cyrillic a
        "\u0435": "e",  # Cyrillic e
        "\u043e": "o",  # Cyrillic o
        "\u0440": "p",  # Cyrillic p
        "\u0441": "c",  # Cyrillic c
        "\u0445": "x",  # Cyrillic x
        "\u0443": "y",  # Cyrillic y
        "\u0456": "i",  # Cyrillic i
    }
)

_SUSPICIOUS_KEYWORDS = {
    "ignore",
    "instruction",
    "instructions",
    "override",
    "jailbreak",
    "system",
    "developer",
    "prompt",
    "persona",
    "roleplay",
}


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(_HOMOGLYPH_MAP)
    return normalized


def _matches_suspicious_keywords(text: str) -> bool:
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    hits = tokens & _SUSPICIOUS_KEYWORDS
    if "ignore" in hits and ("instruction" in hits or "instructions" in hits):
        return True
    if "system" in hits and ("prompt" in hits or "instructions" in hits):
        return True
    if "developer" in hits and ("prompt" in hits or "instructions" in hits):
        return True
    return len(hits) >= 3


def contains_injection(text: str) -> bool:
    """
    Check if the text contains potential injection patterns.

    Args:
        text: The input text to check.

    Returns:
        True if injection detected, False otherwise.
    """

    if text is None:
        raise TypeError("text must be a string, not None")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        return False

    normalized = _normalize_text(text)

    if _BLOCKLIST_REGEX.search(normalized):
        logger.warning("potential_injection_detected", reason="blocklist")
        return True
    if _matches_suspicious_keywords(normalized):
        logger.warning("potential_injection_detected", reason="keyword_heuristic")
        return True
    return False


def validate_for_injection(text: str) -> None:
    """
    Validates the input text for injection patterns and raises an exception if a threat is detected.

    Args:
        text: The input text to validate.

    Raises:
        SecurityError: If the text contains a potential injection pattern.
    """
    if contains_injection(text):
        raise SecurityError(
            "Potential injection attack detected.",
            threat_type="prompt_injection",
        )
