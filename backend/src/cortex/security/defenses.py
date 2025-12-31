"""
Prompt Injection Defenses.

Canonical Blueprint §6.3.1
"""

import logging
import re

logger = logging.getLogger(__name__)

_TAG_BLOCK_RE = re.compile(
    r"<(?P<tag>system|user|assistant|instruction|prompt|context|tool)(?:\s[^>]*)?>.*?</(?P=tag)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_TAG_SELF_RE = re.compile(
    r"<(?P<tag>system|user|assistant|instruction|prompt|context|tool)(?:\s[^>]*)?/>",
    flags=re.IGNORECASE | re.DOTALL,
)


def sanitize_user_input(
    input_text: str, max_length: int = 2048, replacement: str = ""
) -> str:
    """
    Sanitize user input to mitigate prompt injection risks.

    Args:
        input_text: The raw user input.
        max_length: Maximum allowed length of the input.
        replacement: String to replace malicious patterns with.

    Returns:
        Sanitized and truncated string.
    """
    if not isinstance(input_text, str):
        logger.warning(
            "sanitize_user_input received non-string input: %s", type(input_text)
        )
        return ""

    # 1. Strip instruction-like XML tags (system, user, instruction, etc.)
    # This prevents the user from faking a system message or altering roles.
    # We use a non-greedy match to avoid stripping large parts of the text.
    sanitized = _TAG_BLOCK_RE.sub(lambda _match: replacement, input_text)
    sanitized = _TAG_SELF_RE.sub(lambda _match: replacement, sanitized)

    # 2. Remove known meta-instructions or jailbreak phrases
    # Example: "ignore previous instructions", "act as..."
    # This list should be updated based on observed attack patterns.
    jailbreak_patterns = [
        r"ignore all previous instructions",
        r"you are now in.*mode",
        r"act as if you are",
        r"your new instructions are",
    ]
    for pattern in jailbreak_patterns:
        sanitized = re.sub(
            pattern,
            lambda _match: replacement,
            sanitized,
            flags=re.IGNORECASE,
        )

    # 3. Truncate to prevent resource exhaustion and complex injections
    if max_length <= 0:
        return ""
    sanitized = sanitized[:max_length]

    return sanitized
