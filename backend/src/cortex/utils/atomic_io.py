"""
Atomic I/O utilities.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

# Constants
FILE_PERMISSION_OWNER_RW = 0o600


def atomic_write_json(path: Path | str, data: Any) -> None:
    """
    Write JSON data to a file atomically.

    This function ensures that the file is either fully written or not at all,
    preventing partial writes. It achieves this by writing to a temporary file
    in the same directory and then atomically renaming it to the final destination.

    1. Write to temp file with restricted permissions (owner read/write).
    2. fsync the file to ensure it's written to disk.
    3. Atomically rename the temp file to the target path.
    4. fsync the parent directory to ensure the directory entry is durable.

    Args:
        path: The destination file path.
        data: The JSON-serializable data to write.

    Raises:
        TypeError: If the path is None.
        OSError: If file operations fail.
    """
    if path is None:
        raise TypeError("path must be a str or Path, not None")

    path = Path(path)
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
