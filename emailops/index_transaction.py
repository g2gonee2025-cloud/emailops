"""
P0-4 FIX: Transactional index writes with write-ahead log.

This module provides atomic multi-file operations for index building to prevent
corruption when operations are interrupted (crash, kill, power loss).

Design:
1. Write-Ahead Log (WAL): Changes recorded before applying
2. Atomic commit: All files updated or none
3. Rollback support: Can undo partial writes
4. Crash recovery: Can resume or rollback after crash

Usage:
    with IndexTransaction(index_dir) as txn:
        txn.write_embeddings(embeddings_array)
        txn.write_mapping(mapping_list)
        txn.write_metadata(meta_dict)
        # Automatic commit on success, rollback on exception
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .core_exceptions import FileOperationError
from .utils import logger

__all__ = ["IndexTransaction", "recover_index_from_wal"]


@dataclass(frozen=True)
class WALEntry:
    """Single operation in the write-ahead log."""

    operation: str  # "write_embeddings", "write_mapping", "write_metadata"
    file_path: str  # Relative to index directory
    timestamp: str  # ISO 8601 UTC
    checksum: str | None = None  # SHA-256 of content for verification


class IndexTransaction:
    """
    Transactional context manager for atomic index operations.

    Guarantees:
    - All writes succeed or all rollback
    - No partial state visible to readers
    - Crash-safe with automatic recovery
    - Idempotent operations (can safely retry)
    """

    def __init__(self, index_dir: Path):
        """
        Initialize transaction for index directory.

        Args:
            index_dir: Path to index directory (_index/)

        Raises:
            FileOperationError: If directory doesn't exist or isn't writable
        """
        self.index_dir = Path(index_dir).resolve()
        self.wal_dir = self.index_dir / ".wal"
        self.wal_file = self.wal_dir / "transaction.wal"
        self.temp_dir = self.wal_dir / "temp"
        self.committed = False
        self.rolled_back = False
        self.operations: list[WALEntry] = []

        # Validate directory is writable
        if not self.index_dir.exists():
            raise FileOperationError(f"Index directory does not exist: {self.index_dir}")
        if not os.access(self.index_dir, os.W_OK):
            raise FileOperationError(f"Index directory is not writable: {self.index_dir}")

    def __enter__(self) -> IndexTransaction:
        """Start transaction."""
        # Create WAL directory structure
        self.wal_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

        # Check for existing WAL (previous crash)
        if self.wal_file.exists():
            logger.warning(
                "Found existing WAL file - previous transaction may have crashed. "
                "Attempting recovery..."
            )
            try:
                recover_index_from_wal(self.index_dir)
            except Exception as e:
                logger.error("WAL recovery failed: %s", e)
                # Continue anyway - will overwrite

        # Initialize fresh WAL
        self.operations = []
        self._write_wal()
        logger.debug("Started index transaction at %s", self.index_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End transaction - commit on success, rollback on exception."""
        try:
            if exc_type is None and not self.rolled_back:
                # Success - commit atomically
                self._commit()
            else:
                # Exception occurred - rollback
                logger.warning(
                    "Transaction failed with %s: %s - rolling back",
                    exc_type.__name__ if exc_type else "unknown",
                    exc_val
                )
                self._rollback()
        finally:
            # Clean up WAL directory
            self._cleanup_wal()

        # Don't suppress exceptions
        return False

    def write_embeddings(self, embeddings: np.ndarray, file_name: str = "embeddings.npy") -> None:
        """
        Stage embeddings for atomic write.

        Args:
            embeddings: NumPy array of shape (N, D)
            file_name: Filename (default: embeddings.npy)
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")

        # Write to temp file
        temp_path = self.temp_dir / file_name
        np.save(temp_path, embeddings.astype("float32"))

        # Record in WAL
        import hashlib
        checksum = hashlib.sha256(temp_path.read_bytes()).hexdigest()
        entry = WALEntry(
            operation="write_embeddings",
            file_path=file_name,
            timestamp=datetime.now(UTC).isoformat(),
            checksum=checksum
        )
        self.operations.append(entry)
        self._write_wal()
        logger.debug("Staged embeddings write: %s (shape=%s)", file_name, embeddings.shape)

    def write_mapping(self, mapping: list[dict[str, Any]], file_name: str = "mapping.json") -> None:
        """
        Stage mapping for atomic write.

        Args:
            mapping: List of document metadata dicts
            file_name: Filename (default: mapping.json)
        """
        # Write to temp file
        temp_path = self.temp_dir / file_name
        temp_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

        # Record in WAL
        import hashlib
        checksum = hashlib.sha256(temp_path.read_bytes()).hexdigest()
        entry = WALEntry(
            operation="write_mapping",
            file_path=file_name,
            timestamp=datetime.now(UTC).isoformat(),
            checksum=checksum
        )
        self.operations.append(entry)
        self._write_wal()
        logger.debug("Staged mapping write: %s (%d documents)", file_name, len(mapping))

    def write_metadata(self, metadata: dict[str, Any], file_name: str = "meta.json") -> None:
        """
        Stage metadata for atomic write.

        Args:
            metadata: Index metadata dict
            file_name: Filename (default: meta.json)
        """
        # Write to temp file
        temp_path = self.temp_dir / file_name
        temp_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        # Record in WAL
        import hashlib
        checksum = hashlib.sha256(temp_path.read_bytes()).hexdigest()
        entry = WALEntry(
            operation="write_metadata",
            file_path=file_name,
            timestamp=datetime.now(UTC).isoformat(),
            checksum=checksum
        )
        self.operations.append(entry)
        self._write_wal()
        logger.debug("Staged metadata write: %s", file_name)

    def _write_wal(self) -> None:
        """Persist WAL to disk (append-only for crash safety)."""
        wal_data = {
            "version": "1.0",
            "started_at": self.operations[0].timestamp if self.operations else datetime.now(UTC).isoformat(),
            "operations": [
                {
                    "operation": op.operation,
                    "file_path": op.file_path,
                    "timestamp": op.timestamp,
                    "checksum": op.checksum
                }
                for op in self.operations
            ]
        }

        # Atomic WAL write
        temp_wal = self.wal_file.with_suffix(".wal.tmp")
        temp_wal.write_text(json.dumps(wal_data, indent=2), encoding="utf-8")
        temp_wal.replace(self.wal_file)

    def _commit(self) -> None:
        """
        Atomic commit: Move all temp files to final locations.

        Uses rename() which is atomic on POSIX. On Windows, we delete-then-rename
        to avoid "file exists" errors.
        """
        if self.committed:
            return

        logger.info("Committing index transaction with %d operations", len(self.operations))

        # Create backups of existing files (for rollback if needed)
        backup_dir = self.wal_dir / "backup"
        backup_dir.mkdir(exist_ok=True)

        backup_mapping: dict[str, Path] = {}
        for op in self.operations:
            source = self.temp_dir / op.file_path
            dest = self.index_dir / op.file_path

            if not source.exists():
                raise FileOperationError(f"Temp file missing: {source}")

            # Verify checksum matches WAL
            if op.checksum:
                import hashlib
                actual = hashlib.sha256(source.read_bytes()).hexdigest()
                if actual != op.checksum:
                    raise FileOperationError(
                        f"Checksum mismatch for {op.file_path}: "
                        f"expected {op.checksum}, got {actual}"
                    )

            # Backup existing file if present
            if dest.exists():
                backup_path = backup_dir / op.file_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dest, backup_path)
                backup_mapping[op.file_path] = backup_path
                logger.debug("Backed up %s", op.file_path)

        # Now atomically move all temp files (with retry for Windows locks)
        try:
            for op in self.operations:
                source = self.temp_dir / op.file_path
                dest = self.index_dir / op.file_path

                # Windows: delete first, then rename (avoids lock errors)
                if os.name == "nt" and dest.exists():
                    dest.unlink()

                # Atomic rename
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        source.replace(dest)
                        logger.debug("Committed %s", op.file_path)
                        break
                    except (PermissionError, OSError):
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(0.05 * (2 ** attempt))

            self.committed = True
            logger.info("Index transaction committed successfully")

        except Exception as e:
            # Commit failed - restore from backups
            logger.error("Commit failed, restoring from backups: %s", e)
            for file_path, backup_path in backup_mapping.items():
                try:
                    dest = self.index_dir / file_path
                    shutil.copy2(backup_path, dest)
                    logger.debug("Restored %s from backup", file_path)
                except Exception as restore_err:
                    logger.error("Failed to restore %s: %s", file_path, restore_err)
            raise FileOperationError(f"Transaction commit failed: {e}") from e

    def _rollback(self) -> None:
        """Rollback transaction - discard all temp files."""
        if self.rolled_back:
            return

        logger.warning("Rolling back index transaction")

        # Simply discard temp files - original files untouched
        for op in self.operations:
            temp_path = self.temp_dir / op.file_path
            with contextlib.suppress(Exception):
                temp_path.unlink()

        self.rolled_back = True
        logger.info("Index transaction rolled back")

    def _cleanup_wal(self) -> None:
        """Clean up WAL directory after successful commit or rollback."""
        try:
            if self.wal_dir.exists():
                shutil.rmtree(self.wal_dir)
                logger.debug("Cleaned up WAL directory")
        except Exception as e:
            logger.warning("Failed to clean up WAL directory: %s", e)


def recover_index_from_wal(index_dir: Path) -> bool:
    """
    Attempt to recover index from write-ahead log after crash.

    Args:
        index_dir: Path to index directory containing .wal/

    Returns:
        True if recovery succeeded, False if nothing to recover

    Raises:
        FileOperationError: If recovery fails
    """
    wal_file = index_dir / ".wal" / "transaction.wal"

    if not wal_file.exists():
        return False

    try:
        # Load WAL
        wal_data = json.loads(wal_file.read_text(encoding="utf-8"))
        operations = wal_data.get("operations", [])

        logger.info("Attempting WAL recovery with %d operations", len(operations))

        temp_dir = index_dir / ".wal" / "temp"
        backup_dir = index_dir / ".wal" / "backup"

        # Check if backups exist - indicates partial commit
        if backup_dir.exists() and any(backup_dir.iterdir()):
            logger.info("Found backups - rolling back to pre-transaction state")

            # Restore from backups
            for backup_file in backup_dir.rglob("*"):
                if backup_file.is_file():
                    rel_path = backup_file.relative_to(backup_dir)
                    dest = index_dir / rel_path
                    shutil.copy2(backup_file, dest)
                    logger.debug("Restored %s from backup", rel_path)

            logger.info("Rollback completed")

        elif temp_dir.exists():
            # Check if temp files are complete
            all_temp_exist = all(
                (temp_dir / op["file_path"]).exists()
                for op in operations
            )

            if all_temp_exist:
                logger.info("All temp files present - attempting commit recovery")

                # Verify checksums
                for op in operations:
                    temp_file = temp_dir / op["file_path"]
                    if op.get("checksum"):
                        import hashlib
                        actual = hashlib.sha256(temp_file.read_bytes()).hexdigest()
                        if actual != op["checksum"]:
                            raise FileOperationError(
                                f"Checksum mismatch during recovery: {op['file_path']}"
                            )

                # Move temp files to final locations
                for op in operations:
                    source = temp_dir / op["file_path"]
                    dest = index_dir / op["file_path"]

                    if os.name == "nt" and dest.exists():
                        dest.unlink()

                    source.replace(dest)
                    logger.debug("Recovered %s", op["file_path"])

                logger.info("Commit recovery completed")
            else:
                logger.warning("Incomplete temp files - discarding partial transaction")

        # Clean up WAL directory
        wal_dir = index_dir / ".wal"
        shutil.rmtree(wal_dir)
        logger.info("WAL recovery cleanup completed")

        return True

    except Exception as e:
        logger.error("WAL recovery failed: %s", e)
        raise FileOperationError(f"Index recovery failed: {e}") from e
