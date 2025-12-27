"""
Prompt Injection Defenses.

Canonical Blueprint §6.3.1
"""
import re


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
        return ""

    # 1. Truncate to prevent resource exhaustion and complex injections
    sanitized = input_text[:max_length]

    # 2. Strip instruction-like XML tags (system, user, instruction, etc.)
    # This prevents the user from faking a system message or altering roles.
    # We use a non-greedy match to avoid stripping large parts of the text.
    sanitized = re.sub(
        r"<(?i:(?:system|user|assistant|instruction|prompt|context|tool))(?:\s[^>]*)?>.*?</(?i:\\1)>",
        replacement,
        sanitized,
        flags=re.DOTALL,
    )
    sanitized = re.sub(
        r"<(?i:(?:system|user|assistant|instruction|prompt|context|tool))(?:\s[^>]*)?/>",
        replacement,
        sanitized,
        flags=re.DOTALL,
    )

    # 3. Remove known meta-instructions or jailbreak phrases
    # Example: "ignore previous instructions", "act as..."
    # This list should be updated based on observed attack patterns.
    jailbreak_patterns = [
        r"ignore all previous instructions",
        r"you are now in.*mode",
        r"act as if you are",
        r"your new instructions are",
    ]
    for pattern in jailbreak_patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

    return sanitized
