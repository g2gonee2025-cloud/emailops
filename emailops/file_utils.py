"""
File utilities and basic operations.
Handles file encoding detection, directory operations, and basic file I/O.
"""
from __future__ import annotations

import contextlib
import logging
import os
import re
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")


def _strip_control_chars(s: str) -> str:
    """Remove non-printable control characters and normalize newlines."""
    if not s:
        return ""
    # Normalize CRLF/CR -> LF and strip control characters
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_CHARS.sub("", s)


@lru_cache(maxsize=int(os.getenv("FILE_ENCODING_CACHE_SIZE", "1024")))
def _get_file_encoding(path: Path) -> str:
    """
    Detect file encoding with caching.
    MEDIUM #21: Configurable cache size via environment variable (default 1024)
    Returns the most likely encoding for the file.
    """
    encodings = ["utf-8-sig", "utf-8", "utf-16", "latin-1"]

    for enc in encodings:
        try:
            with Path.open(path, encoding=enc) as f:
                f.read(1024)  # Try reading first 1KB
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            continue

    return "latin-1"  # Fallback that won't fail


def read_text_file(path: Path, *, max_chars: int | None = None) -> str:
    """
    Read a text file with multiple encoding fallbacks and sanitization.

    Args:
        path: Path to the text file
        max_chars: Optional hard limit on returned text length

    Returns:
        Decoded and sanitized string (may be truncated)
        Empty string on any read failure
    """
    try:
        # Use cached encoding detection
        encoding = _get_file_encoding(path)

        # Read with detected encoding
        with Path.open(path, encoding=encoding, errors="ignore") as f:
            data = f.read(max_chars) if max_chars is not None else f.read()

        return _strip_control_chars(data)
    except Exception as e:
        logger.warning("Failed to read text file %s: %s", path, e)
        return ""


def ensure_dir(p: Path) -> None:
    """Create directory and parents if needed (idempotent)."""
    p.mkdir(parents=True, exist_ok=True)


def find_conversation_dirs(root: Path) -> list[Path]:
    """
    Heuristic: a conversation directory contains a 'Conversation.txt' file.
    """
    return sorted(p.parent for p in root.rglob("Conversation.txt"))


@contextlib.contextmanager
def temporary_directory(prefix: str = "emailops_"):
    """Context manager for temporary directory creation and cleanup."""
    import shutil

    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        logger.debug("Created temporary directory: %s", temp_dir)
        yield temp_dir
    finally:
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.debug("Cleaned up temporary directory: %s", temp_dir)
            except Exception as e:
                logger.warning("Failed to clean up temp directory %s: %s", temp_dir, e)


@contextlib.contextmanager
def file_lock(path: Path, timeout: float = 10.0):
    """Context manager for file-based locking (platform-aware)."""
    lock_path = path.with_suffix(".lock")
    lock_file = None
    start_time = time.time()

    # Platform-specific locking
    if sys.platform == "win32":
        # Windows file locking using msvcrt
        import msvcrt

        while time.time() - start_time < timeout:
            try:
                lock_file = Path.open(lock_path, "wb")
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                logger.debug("Acquired lock on %s", lock_path)
                try:
                    yield lock_path
                finally:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                    lock_file.close()
                    lock_path.unlink(missing_ok=True)
                    logger.debug("Released lock on %s", lock_path)
                return
            except OSError:
                if lock_file:
                    lock_file.close()
                time.sleep(0.1)
        raise TimeoutError(f"Failed to acquire lock on {lock_path} within {timeout} seconds")
    else:
        # Unix/Linux file locking using fcntl
        import fcntl

        lock_file = Path.open(lock_path, "w", encoding="utf-8")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            logger.debug("Acquired lock on %s", lock_path)
            yield lock_path
        except OSError:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to acquire lock on {lock_path} within {timeout} seconds") from None
            time.sleep(0.1)
        finally:
            if lock_file:
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                    lock_file.close()
                    lock_path.unlink(missing_ok=True)
                    logger.debug("Released lock on %s", lock_path)
                except Exception as e:
                    logger.warning("Error releasing lock on %s: %s", lock_path, e)
