from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ensure_dir",
    "find_conversation_dirs",
    "logger",
    "monitor_performance",
    "read_text_file",
    "safe_str",  # Issue #2 fix - consolidated from core_manifest and feature_summarize
    "scrub_json",
    "scrub_json_string",
    "strip_control_chars",
]


def find_conversation_dirs(root: Path) -> list[Path]:
    """
    Find all valid conversation directories in the export root.
    A valid directory is not a special folder (like _chunks) and contains
    a Conversation.txt file.
    """
    if not root.is_dir():
        return []

    return [
        d
        for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(("_", ".")) and (d / "Conversation.txt").exists()
    ]


def ensure_dir(path: Path) -> None:
    """
    Ensure a directory exists.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def read_text_file(path: Path, encoding: str = "utf-8-sig", max_chars: int | None = None) -> str:
    """
    Read a text file with BOM handling and optional character limit.
    """
    try:
        content = path.read_text(encoding=encoding)
        if max_chars is not None:
            return content[:max_chars]
        return content
    except Exception as e:
        logger.warning(f"Failed to read text file {path}: {e}")
        return ""


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

    Examples:
        >>> _strip_control_chars("Hello\x00World")
        'HelloWorld'
        >>> _strip_control_chars("Hello\r\nWorld", normalize_newlines=True)
        'Hello\nWorld'
    """
    if not s:
        return ""

    # Optionally normalize newlines first (for embedding/indexing pipeline)
    if normalize_newlines:
        s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Remove control characters (preserving \n, \r unless normalized above)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", s)


def scrub_json_string(s: str) -> str:
    """Sanitize a string before JSON parsing."""
    return strip_control_chars(s)


def scrub_json(data: Any) -> Any:
    """Recursively scrub control characters from JSON-like data."""
    if isinstance(data, dict):
        return {k: scrub_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [scrub_json(item) for item in data]
    elif isinstance(data, str):
        return strip_control_chars(data)
    return data


def monitor_performance(func):
    """
    Decorator to monitor function performance with timing and error tracking.

    Logs warnings for slow operations (>1s) and errors with elapsed time.
    Consolidated from util_processing.py (Issue #4 fix).

    Args:
        func: Function to monitor

    Returns:
        Wrapped function with performance monitoring
    """
    import functools
    import time

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time

            if elapsed > 1.0:  # Log slow operations
                logger.warning("%s took %.2f seconds", func.__name__, elapsed)
            else:
                logger.debug("%s completed in %.3f seconds", func.__name__, elapsed)

            return result
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error("%s failed after %.2f seconds: %s", func.__name__, elapsed, e)
            raise

    return wrapper


def safe_str(v: Any, max_len: int, *, strip_trailing: bool = True, warn_on_negative: bool = True) -> str:
    """
    Safely convert any value to string with length limit.

    Consolidated implementation from core_manifest.py and feature_summarize.py (Issue #2 fix).

    Args:
        v: Any value (may be None)
        max_len: Maximum string length
        strip_trailing: If True, strip trailing whitespace when truncating
        warn_on_negative: If True, log warning for negative max_len

    Returns:
        String (empty if v is None, truncated if exceeds max_len)

    Examples:
        >>> safe_str(None, 10)
        ''
        >>> safe_str("Hello World", 5)
        'Hello'
        >>> safe_str("Hello   ", 10, strip_trailing=True)
        'Hello'
    """
    if max_len < 0:
        if warn_on_negative:
            logger.warning("safe_str: Negative max_len %d, using 0", max_len)
        max_len = 0

    try:
        s = "" if v is None else str(v)
        if len(s) > max_len:
            truncated = s[:max_len]
            return truncated.rstrip() if strip_trailing else truncated
        return s
    except Exception as e:
        logger.error("safe_str: Failed to convert value: %s", e)
        return ""
