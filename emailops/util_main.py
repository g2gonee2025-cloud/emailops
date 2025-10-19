from __future__ import annotations

import asyncio
import logging
from functools import partial
from pathlib import Path

from .core_conversation_loader import load_conversation
from .core_email_processing import (
    clean_email_text,
    extract_email_metadata,
    split_email_thread,
)
from .util_files import (
    ensure_dir,
    file_lock,
    find_conversation_dirs,
    read_text_file,
    temporary_directory,
)
from .util_processing import (
    BatchProcessor,
    Person,
    ProcessingConfig,
    get_processing_config,
    get_text_preprocessor,
    monitor_performance,
    should_skip_retrieval_cleaning,
)

"""
Simplified utilities module with backward compatibility.
Functions have been refactored into specialized modules for better maintainability.
"""

# Library-safe logging: no basicConfig at module level
logger = logging.getLogger(__name__)

# Try to import dotenv - it's optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception as e:  # pragma: no cover - optional dependency
    # dotenv is optional - log at debug level
    logger.debug("python-dotenv not available (optional): %s", e)

# Text extraction imports - these may fail if the module is broken
try:
    from .core_text_extraction import extract_text, extract_text_async
except ImportError as e:
    logger.warning("Failed to import text_extraction module: %s", e)
    # Provide stub functions as fallback
    def extract_text(path: Path, *, max_chars: int | None = None, use_cache: bool = True) -> str:
        """Fallback text extraction - just read as text file."""
        # use_cache parameter kept for API compatibility but not used in fallback
        _ = use_cache
        try:
            return read_text_file(path, max_chars=max_chars)
        except Exception:
            return ""

    async def extract_text_async(path: Path, *, max_chars: int | None = None) -> str:
        """Fallback async text extraction."""
        loop = asyncio.get_event_loop()
        func = partial(extract_text, path, max_chars=max_chars)
        return await loop.run_in_executor(None, func)

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

    func = partial(read_text_file, path, max_chars=max_chars)
    return await loop.run_in_executor(None, func)
