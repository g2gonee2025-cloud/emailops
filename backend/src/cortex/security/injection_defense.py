"""
Injection Defense Module.

Implements defense against prompt injection attacks by detecting and neutralizing
common jailbreak patterns and override attempts.
"""

import re

import structlog

logger = structlog.get_logger(__name__)

# Common patterns used in prompt injection / jailbreak attempts
BLOCKLIST_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all instructions",
    r"forget all previous instructions",
    r"you are now (?:a|an) .*",
    r"roleplay as",
    r"system override",
    r"developer mode",
    r"god mode",
    r"jailbreak",
    r"DAN mode",
    r"ignore your previous instructions",
    r"you are now",
    r"new rule:",
    r"important instruction:",
    r"ignore all prior instructions",
    r"disregard previous directions",
    r"start a new conversation",
    r"act as a",
    r"simulate a",
    r"you are a helpful assistant",
    r"your new persona is",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in BLOCKLIST_PATTERNS]


def contains_injection(text: str) -> bool:
    """
    Check if the text contains potential injection patterns.

    Args:
        text: The input text to check.

    Returns:
        True if injection detected, False otherwise.
    """

    if not text:
        return False

    for pattern in _COMPILED_PATTERNS:
        if pattern.search(text):
            logger.warning("potential_injection_detected", pattern=pattern.pattern)
            return True
    return False


def strip_injection_patterns(text: str) -> str:
    """
    Proactively strip known prompt injection patterns from text.

    Blueprint §11.5:
    * Removes patterns like "ignore your previous instructions"
    * Applied to all retrieved context before LLM calls
    """
    if not text:
        return ""

    cleaned = text
    for pattern in _COMPILED_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    if len(cleaned) < len(text):
        logger.warning(
            "stripped_injection_patterns",
            original_length=len(text),
            cleaned_length=len(cleaned),
        )

    return cleaned
