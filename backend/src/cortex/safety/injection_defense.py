"""
Prompt Injection Defense.

Implements §11.5 of the Canonical Blueprint.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Patterns to strip from retrieved context
_INJECTION_PATTERNS = [
    r"ignore your previous instructions",
    r"ignore previous instructions",
    r"you are now",
    r"system override",
    r"new rule:",
    r"important instruction:",
    r"ignore all prior instructions",
    r"disregard previous directions",
    r"start a new conversation",
    r"act as a",
    r"simulate a",
    r"you are a helpful assistant",  # Common jailbreak start
    r"your new persona is",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def strip_injection_patterns(text: str) -> str:
    """
    Proactively strip known prompt injection patterns from text.

    Blueprint §11.5:
    * Removes patterns like "ignore your previous instructions"
    * Applied to all retrieved context before LLM calls
    """
    cleaned = text
    for pattern in _COMPILED_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    if len(cleaned) < len(text):
        logger.warning(
            "Stripped injection patterns from text (original: %d chars, cleaned: %d chars)",
            len(text),
            len(cleaned),
        )

    return cleaned
