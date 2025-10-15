from __future__ import annotations

"""
Simplified utilities module with backward compatibility.
Functions have been refactored into specialized modules for better maintainability.
"""

import asyncio
import logging
import re
import unicodedata
from pathlib import Path

# String-level control character scrubber for JSON
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")

def scrub_json_string(raw: str) -> str:
    """
    Remove control characters from a raw JSON string so it can be safely parsed.
    """
    return _CONTROL_CHARS_RE.sub("", raw)

from .conversation_loader import load_conversation
from .email_processing import (
    clean_email_text,
    extract_email_metadata,
    split_email_thread,
)
from .file_utils import (
    _strip_control_chars,
    ensure_dir,
    file_lock,
    find_conversation_dirs,
    read_text_file,
    temporary_directory,
)
from .processing_utils import (
    BatchProcessor,
    Person,
    ProcessingConfig,
    get_processing_config,
    get_text_preprocessor,
    monitor_performance,
    should_skip_retrieval_cleaning,
)
from .text_extraction import extract_text, extract_text_async

# Best-effort env loading (safe if python-dotenv isn't installed)
try:  # pragma: no cover
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - optional dependency
    pass

# Library-safe logging: no basicConfig at module level
logger = logging.getLogger(__name__)

# Import all functionality from specialized modules


# Recursive JSON scrubber utility

def scrub_json(obj):
    """
    Recursively sanitize all string values and keys in a JSON-like object.
    Removes control characters and normalizes Unicode.
    """
    if isinstance(obj, dict):
        return {
            scrub_json(_strip_control_chars(unicodedata.normalize("NFC", str(k))) if isinstance(k, str) else k):
            scrub_json(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [scrub_json(v) for v in obj]
    elif isinstance(obj, str):
        return _strip_control_chars(unicodedata.normalize("NFC", obj))
    else:
        return obj

__all__ = [
    "BatchProcessor",
    "Person",
    "ProcessingConfig",
    "clean_email_text",
    "ensure_dir",
    "extract_email_metadata",
    "extract_text",
    "extract_text_async",
    "file_lock",
    "find_conversation_dirs",
    "get_processing_config",
    "get_text_preprocessor",
    "load_conversation",
    "logger",
    "monitor_performance",
    "read_text_file",
    "scrub_json",
    "scrub_json_string",
    "should_skip_retrieval_cleaning",
    "split_email_thread",
    "temporary_directory",
]

# Async compatibility wrapper
async def read_text_file_async(path: Path, *, max_chars: int | None = None) -> str:
    """
    Async version of read_text_file using thread pool.

    Args:
        path: Path to the text file
        max_chars: Optional hard limit on returned text length

    Returns:
        Decoded and sanitized string
    """
    loop = asyncio.get_event_loop()
    from functools import partial

    func = partial(read_text_file, path, max_chars=max_chars)
    return await loop.run_in_executor(None, func)
