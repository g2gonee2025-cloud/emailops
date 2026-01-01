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
    """Apply ANSI color to text when stdout is a TTY; return plain text otherwise."""
    normalized = "" if text is None else str(text)
    try:
        is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    except Exception:
        is_tty = False
    if not is_tty:
        return normalized
    color_code = COLORS.get(color)
    if not color_code:
        return normalized
    reset = COLORS.get("reset", "")
    return f"{color_code}{normalized}{reset}"
