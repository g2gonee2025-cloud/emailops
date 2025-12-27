"""
Advanced security defenses.
"""
import re

def sanitize_user_input(input_str: str) -> str:
    """
    Sanitizes user input to prevent prompt injection.
    """
    if not isinstance(input_str, str):
        return ""

    # Remove instruction-like phrases
    instruction_patterns = [
        r"ignore previous instruction",
        r"disregard earlier instruction",
        r"override these rules",
        r"you are now a .* and should respond with .*",
        r"your new instructions are .*",
    ]
    for pattern in instruction_patterns:
        input_str = re.sub(pattern, "", input_str, flags=re.IGNORECASE)

    # Remove control characters except for whitespace
    input_str = "".join(ch for ch in input_str if ch.isprintable() or ch.isspace())

    return input_str
