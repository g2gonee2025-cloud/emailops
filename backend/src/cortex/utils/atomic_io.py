"""
Atomic I/O utilities.
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: Any) -> None:
    """
    Write JSON data to a file atomically.

    1. Write to temp file
    2. fsync
    3. Rename to target path
    """
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", dir=str(parent), delete=False, encoding="utf-8"
    ) as tf:
        json.dump(data, tf, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        tf.flush()
        os.fsync(tf.fileno())
        temp_path = Path(tf.name)

    try:
        os.replace(temp_path, path)
    except OSError:
        # Cleanup on failure
        if temp_path.exists():
            temp_path.unlink()
        raise
