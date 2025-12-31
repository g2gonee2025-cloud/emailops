import sys

# Standard ANSI colors
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}


def colorize(text: str | None, color: str) -> str:
    """Apply ANSI color to text if terminal supports it."""
    # Null-safety: normalize None to empty string
    if text is None:
        text = ""
    # P2 Fix: Guard against None stdout or missing isatty
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return text
    color_code = COLORS.get(color)
    if not color_code:  # P2 Fix: Don't add reset for unknown colors
        return text
    return f"{color_code}{text}{COLORS['reset']}"
