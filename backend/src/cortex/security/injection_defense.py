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
    # These are regex patterns to detect override instructions
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
        Currently implements a simple block strategy: if injection is detected,
        fails safe or strips the offending segment (here we trigger a warning and could redact).

        For this implementation:
        If injection is detected, we log it and return a sanitized warning message
        OR we could return the original text but flagged.

        To rely on robust defense, we will return a neutralized string if injection is found.

        Args:
            text: Input prompt.

        Returns:
            Sanitized text.
        """
        if self.contains_injection(text):
            logger.info("sanitizing_input", action="neutralize")
            return "[CORTEX SECURITY: POTENTIAL INJECTION DETECTED - INPUT REMOVED]"

        return text
