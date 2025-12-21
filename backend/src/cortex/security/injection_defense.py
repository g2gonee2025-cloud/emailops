"""
Injection Defense Module.

Implements defense against prompt injection attacks by detecting and neutralizing
common jailbreak patterns and override attempts.
"""

import re

import structlog

logger = structlog.get_logger(__name__)


class InjectionDefense:
    """
    Defends against Large Language Model (LLM) prompt injection attacks.
    """

    # Common patterns used in prompt injection / jailbreak attempts
    # Combined from security and safety modules
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

    def __init__(self, sensitivity: float = 0.8):
        """
        Initialize the InjectionDefense system.

        Args:
            sensitivity: Detection sensitivity (currently unused, reserved for future ML-based detection).
        """
        self.sensitivity = sensitivity
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.BLOCKLIST_PATTERNS]

    def contains_injection(self, text: str) -> bool:
        """
        Check if the text contains potential injection patterns.

        Args:
            text: The input text to check.

        Returns:
            True if injection detected, False otherwise.
        """
        if not text:
            return False

        for pattern in self.patterns:
            if pattern.search(text):
                logger.warning("potential_injection_detected", pattern=pattern.pattern)
                return True
        return False

    def sanitize_prompt(self, text: str) -> str:
        """
        Sanitize the input text by stripping out injection attempts.
        """
        if self.contains_injection(text):
            logger.info("sanitizing_input", action="neutralize")
            return "[CORTEX SECURITY: POTENTIAL INJECTION DETECTED - INPUT REMOVED]"

        return text


# -----------------------------------------------------------------------------
# Standalone Functions (Ported from Safety Module)
# -----------------------------------------------------------------------------

_COMPILED_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in InjectionDefense.BLOCKLIST_PATTERNS
]


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
