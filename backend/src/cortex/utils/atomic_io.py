"""
Atomic I/O utilities.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

# Constants
FILE_PERMISSION_OWNER_RW = 0o600


def _validate_and_resolve_path(base_dir: Path, file_path: Path) -> Path:
    """
    Ensure the file path is within the sandboxed base directory.

    Args:
        base_dir: The secure base directory.
        file_path: The untrusted file path.

    Returns:
        The resolved, validated absolute file path.

    Raises:
        PermissionError: If path traversal is detected.
    """
    resolved_base = base_dir.resolve()
    # It is critical to resolve the base directory first.
    resolved_path = (resolved_base / file_path).resolve()

    # The most reliable way to check for path traversal is to see if the
    # resolved base path is a parent of the resolved final path.
    if resolved_base not in resolved_path.parents and resolved_path != resolved_base:
        raise PermissionError(f"Path traversal attempt blocked: {resolved_path} is not within {resolved_base}")

    return resolved_path


def atomic_write_json(base_dir: Path | str, file_path: Path | str, data: Any) -> None:
    """
    Atomically and securely write JSON data to a file in a sandboxed directory.

    This function prevents path traversal attacks by ensuring that the resolved
    file path is strictly within the specified `base_dir`. It also ensures that
    the file is either fully written or not at all by using an atomic move.

    1. Validate the path is within the secure `base_dir`.
    2. Write to temp file with restricted permissions (owner read/write).
    3. fsync the file to ensure it's written to disk.
    4. Atomically rename the temp file to the target path.
    5. fsync the parent directory to ensure the directory entry is durable.

    Args:
        base_dir: The secure base directory where writes are allowed.
        file_path: The relative file path within the base directory.
        data: The JSON-serializable data to write.

    Raises:
        PermissionError: If the resolved path is outside the base directory.
        TypeError: If paths are None.
        OSError: If file operations fail.
    """
    if base_dir is None or file_path is None:
        raise TypeError("base_dir and file_path must be a str or Path, not None")

    path = _validate_and_resolve_path(Path(base_dir), Path(file_path))
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    temp_path = None
    try:
        # Create a temporary file in the same directory to ensure atomic move.
        # delete=False is required because we manually rename it.
        with tempfile.NamedTemporaryFile(
            mode="w", dir=str(parent), delete=False, encoding="utf-8"
        ) as tf:
            temp_path = Path(tf.name)

            # Restrict permissions to owner only for security.
            os.chmod(temp_path, FILE_PERMISSION_OWNER_RW)

            # allow_nan=False to produce strictly valid JSON.
            json.dump(
                data,
                tf,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            tf.flush()
            os.fsync(tf.fileno())

        # Atomically replace the destination file with the temporary file.
        os.replace(temp_path, path)

        # fsync the parent directory to ensure the directory entry update is durable.
        # This is a strict durability requirement for some filesystems.
        fd = os.open(str(parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    except Exception:
        # If any step fails, attempt to clean up the temporary file.
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


async def atomic_write_json_async(base_dir: Path | str, file_path: Path | str, data: Any) -> None:
    """
    Asynchronously and securely write JSON data to a file in a sandboxed directory.

    This function is a wrapper around `atomic_write_json` that runs the
    synchronous, blocking file I/O operations in a separate thread to avoid
    blocking the asyncio event loop.

    Args:
        base_dir: The secure base directory where writes are allowed.
        file_path: The relative file path within the base directory.
        data: The JSON-serializable data to write.
    """
    await asyncio.to_thread(atomic_write_json, base_dir, file_path, data)
