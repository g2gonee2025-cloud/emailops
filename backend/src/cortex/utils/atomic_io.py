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

    1. Write to temp file with restricted permissions
    2. fsync file
    3. Rename to target path
    4. fsync parent directory
    """
    if path is None:
        raise TypeError("path must be a str or Path, not None")
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    # create temp file in same directory to ensure atomic move
    # delete=False because we close then move
    with tempfile.NamedTemporaryFile(
        mode="w", dir=str(parent), delete=False, encoding="utf-8"
    ) as tf:
        try:
            # Restrict permissions to owner only
            os.chmod(tf.name, 0o600)  # rw------- (owner read/write only)

            # allow_nan=False to produce valid JSON
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
            temp_path = Path(tf.name)
        except Exception:
            # Clean up if write fails
            tf.close()
            if os.path.exists(tf.name):
                os.unlink(tf.name)
            raise

    try:
        os.replace(temp_path, path)

        # fsync parent directory to ensure directory entry is updated
        # strict durability requirement
        fd = os.open(str(parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Cleanup on failure
        if temp_path.exists():
            temp_path.unlink()
        raise
