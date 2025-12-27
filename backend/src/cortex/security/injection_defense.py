"""
Injection Defense Module.

Implements defense against prompt injection attacks by detecting and neutralizing
common jailbreak patterns and override attempts.
"""

import re

import structlog

logger = structlog.get_logger(__name__)

# Enhanced patterns to detect more sophisticated injection attempts
BLOCKLIST_PATTERNS = [
    # Original patterns
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
    # Sophisticated patterns
    # Detects character separators and noise, e.g., "i g n o r e"
    r"i\s*g\s*n\s*o\s*r\s*e",
    # Detects role-playing variations
    r"from now on, you are",
    r"you will now act as",
    # Detects context shifting
    r"what if someone said",
    # Detects direct commands
    r"print the following",
    r"output the following",
    # Unicode homoglyphs (example with Cyrillic 'і' and 'е')
    r"іgnorе",
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


def validate_for_injection(text: str):
    """
    Validates the input text for injection patterns and raises an exception if a threat is detected.

    Args:
        text: The input text to validate.

    Raises:
        ValueError: If the text contains a potential injection pattern.
    """
    if contains_injection(text):
        raise ValueError("Potential injection attack detected.")
