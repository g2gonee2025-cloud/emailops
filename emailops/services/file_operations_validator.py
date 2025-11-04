"""
File Operations Validator

Provides comprehensive validation before any file operation to prevent data corruption.
Validates paths, permissions, content integrity, and disk space.
"""

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FileOperationValidator:
    """Validates file operations before execution to prevent data corruption."""

    # Size limits
    MAX_FILE_SIZE_MB = 100
    MIN_FREE_SPACE_MB = 500  # Minimum free space required

    # Content validation
    MAX_JSON_DEPTH = 10  # Prevent deeply nested JSON
    MAX_CSV_COLUMNS = 1000  # Prevent memory exhaustion

    def __init__(self, export_root: Path):
        """
        Initialize validator with export root.

        Args:
            export_root: Root directory for operations
        """
        self.export_root = Path(export_root).resolve()

    def validate_write_operation(
        self,
        path: Path,
        content: Any,
        _operation_type: str = "write"
    ) -> tuple[bool, str | None]:
        """
        Validate a write operation before execution.

        Args:
            path: Target path for write
            content: Content to write
            operation_type: Type of operation (write, export, save)

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Path validation
        valid, error = self._validate_path(path, for_write=True)
        if not valid:
            return False, f"Path validation failed: {error}"

        # Check disk space
        valid, error = self._check_disk_space(path, content)
        if not valid:
            return False, f"Insufficient disk space: {error}"

        # Content validation based on file type
        if path.suffix.lower() == '.json':
            valid, error = self._validate_json_content(content)
            if not valid:
                return False, f"JSON validation failed: {error}"
        elif path.suffix.lower() == '.csv':
            valid, error = self._validate_csv_content(content)
            if not valid:
                return False, f"CSV validation failed: {error}"

        # Check for overwrites
        if path.exists():
            valid, error = self._validate_overwrite(path)
            if not valid:
                return False, f"Overwrite validation failed: {error}"

        return True, None

    def validate_read_operation(
        self,
        path: Path,
        _operation_type: str = "read"
    ) -> tuple[bool, str | None]:
        """
        Validate a read operation before execution.

        Args:
            path: Path to read from
            operation_type: Type of operation (read, import, load)

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Path validation
        valid, error = self._validate_path(path, for_write=False)
        if not valid:
            return False, f"Path validation failed: {error}"

        # File existence
        if not path.exists():
            return False, f"File does not exist: {path}"

        # File size check
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.MAX_FILE_SIZE_MB:
                return False, f"File too large: {size_mb:.1f}MB > {self.MAX_FILE_SIZE_MB}MB"
        except OSError as e:
            return False, f"Cannot access file: {e}"

        # Check file is readable
        if not os.access(path, os.R_OK):
            return False, f"File is not readable: {path}"

        return True, None

    def validate_delete_operation(
        self,
        path: Path
    ) -> tuple[bool, str | None]:
        """
        Validate a delete operation before execution.

        Args:
            path: Path to delete

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Path validation
        valid, error = self._validate_path(path, for_write=True)
        if not valid:
            return False, f"Path validation failed: {error}"

        # Check if it's a system file
        if self._is_system_file(path):
            return False, "Cannot delete system files"

        # Check if file is in use (Windows)
        if os.name == 'nt':
            try:
                # Try to open exclusively
                with path.open('rb+'):
                    pass
            except (OSError, PermissionError):
                return False, "File is in use and cannot be deleted"

        return True, None

    def _validate_path(self, path: Path, for_write: bool) -> tuple[bool, str | None]:
        """
        Validate path security and accessibility.

        Args:
            path: Path to validate
            for_write: Whether this is for a write operation

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            resolved = path.resolve()

            # Check path is within export root
            try:
                resolved.relative_to(self.export_root)
            except ValueError:
                return False, "Path is outside allowed directory"

            # Check for directory traversal
            if ".." in str(path):
                return False, "Path traversal detected"

            # Check path length (Windows limit)
            if len(str(resolved)) > 260:
                return False, "Path too long for Windows"

            # For write operations, check parent directory
            if for_write:
                parent = resolved.parent
                if not parent.exists():
                    # Try to create parent
                    try:
                        parent.mkdir(parents=True, exist_ok=True)
                    except OSError as e:
                        return False, f"Cannot create parent directory: {e}"

                if not os.access(parent, os.W_OK):
                    return False, "Parent directory is not writable"

            return True, None

        except Exception as e:
            return False, f"Path validation error: {e}"

    def _check_disk_space(self, path: Path, content: Any) -> tuple[bool, str | None]:
        """
        Check if there's enough disk space for the operation.

        Args:
            path: Target path
            content: Content to write

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Estimate content size
            if isinstance(content, str):
                content_size = len(content.encode('utf-8'))
            elif isinstance(content, bytes):
                content_size = len(content)
            elif isinstance(content, (list, dict)):
                # Rough estimate for structured data
                import json
                content_size = len(json.dumps(content).encode('utf-8'))
            else:
                content_size = 0

            # Get free space
            stat = shutil.disk_usage(path.parent if path.parent.exists() else self.export_root)
            free_mb = stat.free / (1024 * 1024)

            # Check minimum free space
            if free_mb < self.MIN_FREE_SPACE_MB:
                return False, f"Low disk space: {free_mb:.1f}MB"

            # Check if content fits
            content_mb = content_size / (1024 * 1024)
            if content_mb > free_mb - self.MIN_FREE_SPACE_MB:
                return False, f"Insufficient space for {content_mb:.1f}MB file"

            return True, None

        except Exception as e:
            logger.warning(f"Disk space check failed: {e}")
            return True, None  # Don't block on check failure

    def _validate_json_content(self, content: Any) -> tuple[bool, str | None]:
        """
        Validate JSON content for safety.

        Args:
            content: JSON content to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        import json

        try:
            # Serialize to check validity
            json_str = json.dumps(content)

            # Check size
            if len(json_str) > self.MAX_FILE_SIZE_MB * 1024 * 1024:
                return False, "JSON content too large"

            # Check depth
            def check_depth(obj, depth=0):
                if depth > self.MAX_JSON_DEPTH:
                    return False
                if isinstance(obj, dict):
                    return all(check_depth(v, depth + 1) for v in obj.values())
                elif isinstance(obj, list):
                    return all(check_depth(item, depth + 1) for item in obj)
                return True

            if not check_depth(content):
                return False, f"JSON nesting too deep (>{self.MAX_JSON_DEPTH})"

            return True, None

        except (TypeError, ValueError) as e:
            return False, f"Invalid JSON content: {e}"

    def _validate_csv_content(self, content: Any) -> tuple[bool, str | None]:
        """
        Validate CSV content for safety.

        Args:
            content: CSV content to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(content, list):
            return False, "CSV content must be a list"

        if not content:
            return True, None  # Empty CSV is valid

        # Check first row for column count
        first_row = content[0]
        if not isinstance(first_row, dict):
            return False, "CSV rows must be dictionaries"

        if len(first_row) > self.MAX_CSV_COLUMNS:
            return False, f"Too many columns: {len(first_row)} > {self.MAX_CSV_COLUMNS}"

        # Check for CSV injection
        for row in content[:100]:  # Check first 100 rows
            for _key, value in row.items():
                if isinstance(value, str) and value.startswith(('=', '+', '-', '@', '\t', '\r')):
                    return False, f"Potential CSV injection in value: {value[:20]}"

        return True, None

    def _validate_overwrite(self, path: Path) -> tuple[bool, str | None]:
        """
        Validate overwrite safety.

        Args:
            path: Path that will be overwritten

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if it's a critical file
        critical_files = {
            'config.json', 'manifest.json', 'index.json',
            'embeddings.npy', 'mapping.json'
        }

        if path.name in critical_files:
            # Create backup
            backup_path = path.with_suffix(f'{path.suffix}.backup')
            try:
                shutil.copy2(path, backup_path)
                logger.info(f"Created backup: {backup_path}")
            except Exception as e:
                return False, f"Cannot create backup for critical file: {e}"

        return True, None

    def _is_system_file(self, path: Path) -> bool:
        """
        Check if file is a system file that shouldn't be deleted.

        Args:
            path: Path to check

        Returns:
            True if system file
        """
        system_patterns = {
            '.git', '.gitignore', '__pycache__',
            'config.json', 'manifest.json'
        }

        # Check file name
        if path.name in system_patterns:
            return True

        # Check if in system directory
        return any(part in system_patterns for part in path.parts)

    def compute_checksum(self, path: Path) -> str | None:
        """
        Compute SHA-256 checksum for integrity verification.

        Args:
            path: File path

        Returns:
            Checksum string or None if failed
        """
        try:
            sha256 = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Checksum computation failed: {e}")
            return None
