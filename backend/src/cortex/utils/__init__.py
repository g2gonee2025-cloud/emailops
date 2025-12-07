"""
Utility modules for Cortex.
"""
import re
from pathlib import Path
from typing import Any


def strip_control_chars(s: str, *, normalize_newlines: bool = False) -> str:
    """
    Remove non-printable control characters from string.

    Args:
        s: String to clean
        normalize_newlines: If True, normalize CRLF/CR to LF before stripping

    Returns:
        Cleaned string with control characters removed

    Note:
        Control chars removed: [\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]
        Newlines (\n, \r) are preserved unless normalize_newlines=True
    """
    if not s:
        return ""

    # Optionally normalize newlines first (for embedding/indexing pipeline)
    if normalize_newlines:
        s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Remove control characters (preserving \n, \r unless normalized above)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", s)


def read_text_file(
    path: Path, encoding: str = "utf-8-sig", max_chars: int | None = None
) -> str:
    """
    Read a text file with BOM handling and optional character limit.
    """
    try:
        content = path.read_text(encoding=encoding)
        if max_chars is not None:
            return content[:max_chars]
        return content
    except Exception:
        return ""


def scrub_json(data: Any) -> Any:
    """
    Recursively remove control characters from JSON-compatible structures.
    """
    if isinstance(data, str):
        return strip_control_chars(data)
    if isinstance(data, dict):
        return {k: scrub_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [scrub_json(v) for v in data]
    return data
