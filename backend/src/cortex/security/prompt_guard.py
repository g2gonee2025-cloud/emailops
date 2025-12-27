"""
PromptGuard: Advanced Prompt Injection & Misuse Defense.

This module provides a multi-layered defense against prompt injection,
jailbreaking, and other prompt-based attacks. It combines heuristic checks
with semantic analysis to provide a more robust defense than simple blocklists.
"""

import re
from typing import List

import structlog

logger = structlog.get_logger(__name__)

# Layer 1: Heuristic-based Filters
HEURISTIC_PATTERNS = [
    re.compile(r"[\s\S]*->[\s\S]*->[\s\S]*"),
    re.compile(r"\b(ignore|disregard|forget)\s+.*\b(instructions|directions)\b", re.IGNORECASE),
    re.compile(r"\{\{.*\}\}", re.DOTALL),
    re.compile(r"^(?=.*\b(human|user)\b)(?=.*\b(ai|assistant)\b).*", re.IGNORECASE | re.DOTALL),
]

# Layer 2: Semantic Analysis Keywords
SEMANTIC_VIOLATION_KEYWORDS = [
    "roleplay", "persona", "act as", "simulate", "developer mode", "god mode",
    "jailbreak", "DAN", "override", "system prompt", "instructions"
]

class PromptGuard:
    """
    A robust defense mechanism against prompt injection attacks.
    """

    def __init__(self, semantic_keywords: List[str] = None):
        """
        Initializes PromptGuard with optional custom semantic keywords.
        """
        self.semantic_keywords = [kw.lower() for kw in (semantic_keywords or SEMANTIC_VIOLATION_KEYWORDS)]

    def contains_injection(self, text: str) -> bool:
        """
        Checks if the text contains potential injection patterns.

        This method applies a multi-layered defense:
        1. Heuristic patterns for common attack structures.
        2. Semantic keyword matching for intent subversion.

        Args:
            text: The input text to check.

        Returns:
            True if an injection attempt is detected, False otherwise.
        """
        if not text:
            return False

        lower_text = text.lower()

        # Layer 1: Heuristic Checks
        for pattern in HEURISTIC_PATTERNS:
            if pattern.search(text):
                logger.warning("potential_injection_detected", pattern=pattern.pattern)
                return True

        # Layer 2: Semantic Keyword Checks
        if any(keyword in lower_text for keyword in self.semantic_keywords):
            logger.warning("semantic_violation_detected", text=text)
            return True

        return False

    def sanitize_prompt(self, text: str) -> str:
        """
        Sanitizes a prompt if it contains an injection attempt.

        Instead of stripping patterns (which can corrupt input), this method
        replaces a malicious prompt with a safe, neutral message. This prevents
        the LLM from processing the harmful instruction while still providing a
        traceable output.

        Args:
            text: The input text to sanitize.

        Returns:
            The original text if safe, or a sanitized message if an injection
            is detected.
        """
        if self.contains_injection(text):
            logger.warning("sanitizing_malicious_prompt")
            return (
                "POTENTIAL INJECTION DETECTED. "
                "The original prompt has been blocked for security reasons."
            )
        return text
