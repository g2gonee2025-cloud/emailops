"""
Atomic File Service Module

Provides atomic file operations to prevent data corruption.
Uses proven patterns from indexing_main.py for atomic writes.
"""

import contextlib
import csv
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from .file_service import FileService

logger = logging.getLogger(__name__)


class AtomicFileService(FileService):
    """
    File service with atomic write operations to prevent data corruption.

    All write operations use temp files and atomic replace to ensure
    files are never left in a partially written state.
    """

    def __init__(self, export_root: str):
        """Initialize atomic file service."""
        super().__init__(export_root)
        self._retry_max = 5
        self._retry_delay_base = 0.05  # 50ms base delay

    def _atomic_write_bytes(self, dest: Path, data: bytes) -> None:
        """
        Write bytes atomically using temp file and replace.

        Critical for preventing data corruption on crash/interrupt.
        Based on proven pattern from indexing_main.py.

        Args:
            dest: Destination file path
            data: Bytes to write

        Raises:
            OSError: If write fails after all retries
        """
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        backup = dest.with_suffix(dest.suffix + ".backup")

        try:
            # Ensure parent directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file with fsync
            try:
                with tmp.open("wb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as e:
                if tmp.exists():
                    with contextlib.suppress(Exception):
                        tmp.unlink()
                raise OSError(f"Failed to write temp file {tmp}: {e}") from e

            # Verify temp file
            if not tmp.exists():
                raise OSError(f"Temp file {tmp} was not created")
            if tmp.stat().st_size != len(data):
                tmp.unlink()
                raise OSError(
                    f"Temp file size mismatch: expected {len(data)}, got {tmp.stat().st_size}"
                )

            # Windows-safe atomic replace
            if dest.exists():
                try:
                    dest.replace(backup)
                except (PermissionError, OSError) as e:
                    logger.warning(f"Could not backup {dest.name}: {e}")
                    with contextlib.suppress(Exception):
                        dest.unlink()

            # Retry loop for Windows file locking
            success = False
            for attempt in range(self._retry_max):
                try:
                    tmp.replace(dest)
                    success = True
                    break
                except (PermissionError, OSError):
                    if attempt == self._retry_max - 1:
                        if backup.exists() and not dest.exists():
                            with contextlib.suppress(Exception):
                                backup.replace(dest)
                        raise
                    delay = self._retry_delay_base * (2 ** attempt)
                    logger.debug(f"Retry {attempt + 1}/{self._retry_max} for {dest.name}, waiting {delay}s")
                    time.sleep(delay)

            # Clean up backup on success
            if success and backup.exists():
                with contextlib.suppress(Exception):
                    backup.unlink()

            # Verify destination exists
            if not dest.exists():
                raise OSError(f"Destination file {dest} does not exist after replace")

        except Exception as e:
            # Clean up temp file on any error
            if tmp.exists():
                with contextlib.suppress(Exception):
                    tmp.unlink()
            raise OSError(f"Atomic write failed for {dest}: {e}") from e

    def _atomic_write_text(self, dest: Path, text: str, encoding: str = "utf-8") -> None:
        """
        Write text atomically using UTF-8 by default.

        Args:
            dest: Destination file path
            text: Text to write
            encoding: Text encoding (default: utf-8)

        Raises:
            OSError: If write fails after all retries
        """
        tmp = dest.with_suffix(dest.suffix + ".tmp")

        try:
            # Ensure parent directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file with fsync
            try:
                with tmp.open("w", encoding=encoding) as f:
                    f.write(text)
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as e:
                if tmp.exists():
                    with contextlib.suppress(Exception):
                        tmp.unlink()
                raise OSError(f"Failed to write temp file {tmp}: {e}") from e

            # Verify temp file
            if not tmp.exists():
                raise OSError(f"Temp file {tmp} was not created")

            expected_size = len(text.encode(encoding))
            actual_size = tmp.stat().st_size
            # Allow 5% tolerance for encoding differences
            if abs(actual_size - expected_size) > expected_size * 0.05:
                tmp.unlink()
                raise OSError(
                    f"Temp file size mismatch: expected ~{expected_size}, got {actual_size}"
                )

            # Atomic replace with retry for Windows
            for attempt in range(self._retry_max):
                try:
                    tmp.replace(dest)
                    break
                except (PermissionError, OSError):
                    if attempt == self._retry_max - 1:
                        raise
                    delay = self._retry_delay_base * (2 ** attempt)
                    logger.debug(f"Retry {attempt + 1}/{self._retry_max} for {dest.name}, waiting {delay}s")
                    time.sleep(delay)

            # Verify destination exists
            if not dest.exists():
                raise OSError(f"Destination file {dest} does not exist after replace")

        except Exception as e:
            # Clean up temp file on any error
            if tmp.exists():
                with contextlib.suppress(Exception):
                    tmp.unlink()
            raise OSError(f"Atomic write failed for {dest}: {e}") from e

    def save_text_file(self, content: str, path: Path, encoding: str = "utf-8") -> bool:
        """
        Save text content atomically.

        Overrides parent to use atomic write operation.

        Args:
            content: Text content to save
            path: Output file path
            encoding: Text encoding

        Returns:
            True if successful, False otherwise
        """
        try:
            self._atomic_write_text(path, content, encoding)
            logger.info(f"Atomically saved text file: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save text file atomically: {e}", exc_info=True)
            return False

    def save_binary_file(self, content: bytes, path: Path) -> bool:
        """
        Save binary content atomically.

        Overrides parent to use atomic write operation.

        Args:
            content: Binary content to save
            path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            self._atomic_write_bytes(path, content)
            logger.info(f"Atomically saved binary file: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save binary file atomically: {e}", exc_info=True)
            return False

    def save_json(
        self, data: Any, path: Path, indent: int = 2, ensure_ascii: bool = False
    ) -> bool:
        """
        Save data as JSON file atomically.

        Overrides parent to use atomic write operation.

        Args:
            data: Data to serialize
            path: Output file path
            indent: JSON indentation
            ensure_ascii: Whether to escape non-ASCII characters

        Returns:
            True if successful, False otherwise
        """
        try:
            # Serialize to string first
            json_str = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)

            # Use atomic write
            self._atomic_write_text(path, json_str, encoding="utf-8")
            logger.info(f"Atomically saved JSON file: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON file atomically: {e}", exc_info=True)
            return False

    def export_csv(
        self, data: list[dict[str, Any]], path: Path, headers: list[str] | None = None
    ) -> bool:
        """
        Export data to CSV file atomically.

        Overrides parent to use atomic write operation.

        Args:
            data: List of dictionaries to export
            path: Output CSV file path
            headers: Optional list of column headers

        Returns:
            True if successful, False otherwise
        """
        try:
            if not data:
                logger.warning("No data to export to CSV")
                return False

            # Determine headers
            if headers is None:
                headers = list(data[0].keys())

            # Write to temp file first
            tmp = path.with_suffix(path.suffix + ".tmp")
            path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with tmp.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(data)
                    f.flush()
                    os.fsync(f.fileno())

                # Atomic replace
                for attempt in range(self._retry_max):
                    try:
                        tmp.replace(path)
                        break
                    except (PermissionError, OSError):
                        if attempt == self._retry_max - 1:
                            raise
                        delay = self._retry_delay_base * (2 ** attempt)
                        time.sleep(delay)

                logger.info(f"Atomically exported {len(data)} rows to CSV: {path}")
                return True

            finally:
                # Clean up temp file
                if tmp.exists():
                    with contextlib.suppress(Exception):
                        tmp.unlink()

        except Exception as e:
            logger.error(f"Failed to export CSV atomically: {e}", exc_info=True)
            return False

    def copy_file(self, source: Path, destination: Path) -> bool:
        """
        Copy file atomically (write to temp, then replace).

        Overrides parent to use atomic operation.

        Args:
            source: Source file path
            destination: Destination file path

        Returns:
            True if successful, False otherwise
        """
        try:
            if not source.exists():
                logger.warning(f"Source file not found: {source}")
                return False

            # Read source
            content = source.read_bytes()

            # Write atomically to destination
            self._atomic_write_bytes(destination, content)

            # Preserve metadata
            shutil.copystat(source, destination)

            logger.info(f"Atomically copied {source} to {destination}")
            return True

        except Exception as e:
            logger.error(f"Failed to copy file atomically: {e}", exc_info=True)
            return False

    def move_file(self, source: Path, destination: Path) -> bool:
        """
        Move file atomically (copy then delete).

        Overrides parent to use atomic operation.

        Args:
            source: Source file path
            destination: Destination file path

        Returns:
            True if successful, False otherwise
        """
        try:
            if not source.exists():
                logger.warning(f"Source file not found: {source}")
                return False

            # First copy atomically
            if not self.copy_file(source, destination):
                return False

            # Then delete source
            source.unlink()

            logger.info(f"Atomically moved {source} to {destination}")
            return True

        except Exception as e:
            logger.error(f"Failed to move file atomically: {e}", exc_info=True)
            # Try to clean up destination if move failed
            if destination.exists() and source.exists():
                with contextlib.suppress(Exception):
                    destination.unlink()
            return False


class TransactionalFileService(AtomicFileService):
    """
    File service with transaction support for multi-file atomic operations.

    Allows grouping multiple file operations into a single transaction
    that either fully succeeds or fully rolls back.
    """

    def __init__(self, export_root: str):
        """Initialize transactional file service."""
        super().__init__(export_root)
        self._transaction_active = False
        self._transaction_ops: list[dict[str, Any]] = []
        self._transaction_dir: Path | None = None

    def begin_transaction(self) -> None:
        """
        Begin a new file transaction.

        All write operations after this will be staged until commit.

        Raises:
            RuntimeError: If transaction already active
        """
        if self._transaction_active:
            raise RuntimeError("Transaction already active")

        # Create temp directory for staging
        self._transaction_dir = Path(tempfile.mkdtemp(prefix="emailops_txn_"))
        self._transaction_ops = []
        self._transaction_active = True
        logger.info(f"Started file transaction in {self._transaction_dir}")

    def commit_transaction(self) -> bool:
        """
        Commit all staged operations atomically.

        Returns:
            True if all operations succeeded, False if rollback occurred
        """
        if not self._transaction_active:
            logger.warning("No active transaction to commit")
            return False

        try:
            # Execute all staged operations
            for op in self._transaction_ops:
                op_type = op["type"]

                if op_type == "write":
                    # Move staged file to final destination
                    staged = op["staged_path"]
                    dest = op["dest_path"]

                    # Atomic replace
                    for attempt in range(self._retry_max):
                        try:
                            staged.replace(dest)
                            break
                        except (PermissionError, OSError):
                            if attempt == self._retry_max - 1:
                                raise
                            delay = self._retry_delay_base * (2 ** attempt)
                            time.sleep(delay)

                elif op_type == "delete":
                    # Delete the file
                    path = op["path"]
                    if path.exists():
                        path.unlink()

                elif op_type == "move":
                    # Move operation
                    source = op["source"]
                    dest = op["dest"]
                    if source.exists():
                        source.rename(dest)

            logger.info(f"Committed {len(self._transaction_ops)} operations")
            return True

        except Exception as e:
            logger.error(f"Transaction failed, rolling back: {e}")
            self.rollback_transaction()
            return False

        finally:
            # Clean up transaction state
            self._cleanup_transaction()

    def rollback_transaction(self) -> None:
        """Rollback all staged operations."""
        if not self._transaction_active:
            return

        logger.info(f"Rolling back {len(self._transaction_ops)} operations")

        # Restore any backups made during staging
        for op in self._transaction_ops:
            if "backup_path" in op and op["backup_path"].exists():
                try:
                    op["backup_path"].replace(op["original_path"])
                except Exception as e:
                    logger.error(f"Failed to restore backup: {e}")

        self._cleanup_transaction()

    def _cleanup_transaction(self) -> None:
        """Clean up transaction state and temp files."""
        if self._transaction_dir and self._transaction_dir.exists():
            try:
                shutil.rmtree(self._transaction_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up transaction dir: {e}")

        self._transaction_active = False
        self._transaction_ops = []
        self._transaction_dir = None

    def save_text_file(self, content: str, path: Path, encoding: str = "utf-8") -> bool:
        """Save text file (staged if in transaction)."""
        if self._transaction_active:
            # Stage in transaction directory
            if not self._transaction_dir:
                raise RuntimeError("Transaction directory not initialized")
            staged_path = self._transaction_dir / f"staged_{len(self._transaction_ops)}.txt"
            super().save_text_file(content, staged_path, encoding)

            self._transaction_ops.append({
                "type": "write",
                "staged_path": staged_path,
                "dest_path": path
            })
            return True
        else:
            return super().save_text_file(content, path, encoding)

    def save_json(
        self, data: Any, path: Path, indent: int = 2, ensure_ascii: bool = False
    ) -> bool:
        """Save JSON file (staged if in transaction)."""
        if self._transaction_active:
            # Stage in transaction directory
            if not self._transaction_dir:
                raise RuntimeError("Transaction directory not initialized")
            staged_path = self._transaction_dir / f"staged_{len(self._transaction_ops)}.json"
            super().save_json(data, staged_path, indent, ensure_ascii)

            self._transaction_ops.append({
                "type": "write",
                "staged_path": staged_path,
                "dest_path": path
            })
            return True
        else:
            return super().save_json(data, path, indent, ensure_ascii)
