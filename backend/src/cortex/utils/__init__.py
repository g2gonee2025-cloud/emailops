"""
Utility modules for Cortex.
"""

import re
from pathlib import Path
from typing import Any

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def strip_control_chars(s: str | None, *, normalize_newlines: bool = False) -> str:
    """
    Remove non-printable control characters from string.

    Args:
        s: String to clean (if None, returns empty string)
        normalize_newlines: If True, normalize CRLF/CR to LF before stripping

    Returns:
        Cleaned string with control characters removed
    """
    if not s:
        return ""

    # Optionally normalize newlines first (for embedding/indexing pipeline)
    if normalize_newlines:
        s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Remove control characters (preserving \n, \r unless normalized above)
    return _CONTROL_CHARS_RE.sub("", s)


def read_text_file(
    path: Path, encoding: str = "utf-8-sig", max_chars: int | None = None
) -> str:
    """
    Read a text file with BOM handling and optional character limit.
    """
    if max_chars is not None and max_chars < 0:
        raise ValueError("max_chars must be non-negative")

    try:
        if max_chars is None:
            content = path.read_text(encoding=encoding)
        else:
            # Avoid loading the entire file when only a prefix is needed
            with path.open("r", encoding=encoding) as f:
                content = f.read(max_chars)
        return content
    except (FileNotFoundError, PermissionError, UnicodeDecodeError, IOError):
        # Allow specific errors to bubble up or handle them?
        # The original code acted as a silent fallback (returning ""), typically used for optional files.
        # But for debugging, silent failure is bad.
        # Given "Phase 1" fixes, let's keep the signature but maybe log?
        # For now, let's stick to the safer behavior but at least narrow the exception.
        return ""


def scrub_json(data: Any) -> Any:
    """
    Recursively remove control characters from JSON-compatible structures.
    Sanitizes both keys and values in dictionaries.
    """
    if isinstance(data, str):
        return strip_control_chars(data)
    if isinstance(data, dict):
        return {
            strip_control_chars(k): scrub_json(v)
            for k, v in data.items()
            if k is not None
        }
    if isinstance(data, list):
        return [scrub_json(v) for v in data]
    return data
