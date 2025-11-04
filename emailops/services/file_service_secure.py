"""
File Service Module - Security Hardened Version

Handles all file I/O operations with comprehensive security validation.
Prevents path traversal, symlink attacks, and resource exhaustion.
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


class SecureFileService:
    """Service for handling file operations with security validation."""

    # Security limits
    MAX_FILE_SIZE_MB: ClassVar[int] = 100  # Maximum file size for read/write operations
    MAX_PATH_LENGTH: ClassVar[int] = 260  # Windows MAX_PATH limit
    MAX_CSV_ROWS: ClassVar[int] = 100000  # Maximum rows in CSV to prevent memory exhaustion
    ALLOWED_EXTENSIONS: ClassVar[set[str]] = {
        '.txt', '.json', '.csv', '.log', '.md', '.html', '.xml',
        '.pdf', '.docx', '.xlsx', '.eml', '.msg'
    }

    def __init__(self, export_root: str):
        """
        Initialize the secure file service.

        Args:
            export_root: Root directory for email exports
        """
        self.export_root = Path(export_root).resolve()

        # Ensure export root exists and is a directory
        if not self.export_root.exists():
            self.export_root.mkdir(parents=True, exist_ok=True)
        elif not self.export_root.is_dir():
            raise ValueError(f"Export root must be a directory: {export_root}")

        # Store canonical path for security checks
        self._canonical_root = os.path.realpath(str(self.export_root))
        logger.info(f"SecureFileService initialized with root: {self._canonical_root}")

    def _validate_path(self, path: Path, check_extension: bool = False) -> Path:
        """
        Validate and sanitize a path for security.

        Args:
            path: Path to validate
            check_extension: Whether to check file extension

        Returns:
            Validated absolute path

        Raises:
            SecurityError: If path is invalid or insecure
        """
        # Convert to Path object if string
        if isinstance(path, str):
            path = Path(path)

        # Check path length
        if len(str(path)) > self.MAX_PATH_LENGTH:
            raise SecurityError(f"Path too long (>{self.MAX_PATH_LENGTH} chars): {path}")

        # Resolve to absolute path (follows symlinks)
        try:
            resolved = path.resolve(strict=False)
        except Exception as e:
            raise SecurityError(f"Invalid path: {path}") from e

        # Get canonical path (resolves symlinks on all platforms)
        canonical = os.path.realpath(str(resolved))

        # Ensure path is within export root
        try:
            # Both paths must be canonical for proper comparison
            Path(canonical).relative_to(self._canonical_root)
        except ValueError as exc:
            raise SecurityError(
                f"Path traversal detected - path outside export root: {path}"
            ) from exc

        # Check for null bytes (injection attack)
        if '\x00' in str(path):
            raise SecurityError("Null byte injection detected in path")

        # Check for dangerous path components
        path_parts = Path(canonical).parts
        dangerous = {'..', '.', '~', '$', '|', '>', '<', '&', ';', '`'}
        for part in path_parts:
            if part in dangerous or part.startswith('.'):
                raise SecurityError(f"Dangerous path component: {part}")

        # Check file extension if requested
        if check_extension and resolved.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise SecurityError(
                f"File extension not allowed: {resolved.suffix}. "
                f"Allowed: {', '.join(sorted(self.ALLOWED_EXTENSIONS))}"
            )

        # Check if path exists and is a symlink
        if resolved.exists() and resolved.is_symlink():
            # Symlinks are dangerous - could point outside root
            link_target = resolved.readlink()
            link_canonical = Path(os.path.realpath(str(link_target)))
            try:
                link_canonical.relative_to(self._canonical_root)
            except ValueError as exc:
                raise SecurityError(
                    f"Symlink points outside export root: {resolved} -> {link_target}"
                ) from exc

        return resolved

    def _check_file_size(self, path: Path, max_mb: float | None = None) -> None:
        """
        Check if file size is within limits.

        Args:
            path: File path to check
            max_mb: Maximum size in MB (uses default if None)

        Raises:
            SecurityError: If file is too large
        """
        if not path.exists() or not path.is_file():
            return

        max_bytes = (max_mb or self.MAX_FILE_SIZE_MB) * 1024 * 1024
        size = path.stat().st_size

        if size > max_bytes:
            size_mb = size / (1024 * 1024)
            raise SecurityError(
                f"File too large: {size_mb:.1f}MB > {max_mb or self.MAX_FILE_SIZE_MB}MB"
            )

    def get_conversation_path(self, conv_id: str) -> Path | None:
        """
        Get and validate conversation path with security checks.

        Args:
            conv_id: Conversation ID

        Returns:
            Validated path if exists and valid, None otherwise
        """
        # Sanitize conversation ID
        if not conv_id or not conv_id.replace('_', '').replace('-', '').isalnum():
            logger.warning(f"Invalid conversation ID format: {conv_id}")
            return None

        try:
            # Build path and validate
            candidate = self.export_root / conv_id
            validated = self._validate_path(candidate)

            # Check if it's a valid conversation directory
            if not validated.exists() or not validated.is_dir():
                return None

            # Must contain Conversation.txt
            conv_file = validated / "Conversation.txt"
            if not conv_file.exists():
                return None

            # Validate conversation file
            self._validate_path(conv_file)
            self._check_file_size(conv_file)

            return validated

        except SecurityError as e:
            logger.error(f"Security violation accessing conversation {conv_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get conversation path: {e}")
            return None

    def save_text_file(
        self,
        content: str,
        path: Path,
        encoding: str = "utf-8",
        validate_extension: bool = True
    ) -> bool:
        """
        Save text content to a file with security validation.

        Args:
            content: Text content to save
            path: Output file path
            encoding: Text encoding
            validate_extension: Whether to validate file extension

        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate path
            validated_path = self._validate_path(path, check_extension=validate_extension)

            # Check content size
            content_bytes = content.encode(encoding)
            max_bytes = self.MAX_FILE_SIZE_MB * 1024 * 1024
            if len(content_bytes) > max_bytes:
                raise SecurityError(
                    f"Content too large: {len(content_bytes)/1024/1024:.1f}MB > {self.MAX_FILE_SIZE_MB}MB"
                )

            # Create parent directory if needed
            validated_path.parent.mkdir(parents=True, exist_ok=True)

            # Write atomically to prevent partial writes
            temp_path = validated_path.with_suffix(validated_path.suffix + '.tmp')
            try:
                temp_path.write_text(content, encoding=encoding)
                temp_path.replace(validated_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

            logger.info(f"Saved text file: {validated_path}")
            return True

        except SecurityError as e:
            logger.error(f"Security violation saving file: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to save text file: {e}", exc_info=True)
            return False

    def read_text_file(
        self,
        path: Path,
        encoding: str = "utf-8",
        max_chars: int | None = None,
        validate_extension: bool = True
    ) -> str | None:
        """
        Read text content from a file with security validation.

        Args:
            path: File path to read
            encoding: Text encoding
            max_chars: Maximum characters to read
            validate_extension: Whether to validate file extension

        Returns:
            File content or None if failed
        """
        try:
            # Validate path
            validated_path = self._validate_path(path, check_extension=validate_extension)

            if not validated_path.exists():
                logger.warning(f"File not found: {validated_path}")
                return None

            # Check file size before reading
            self._check_file_size(validated_path)

            # Read content
            content = validated_path.read_text(encoding=encoding)

            # Apply character limit if specified
            if max_chars is not None and len(content) > max_chars:
                content = content[:max_chars]

            logger.debug(f"Read text file: {validated_path}")
            return content

        except SecurityError as e:
            logger.error(f"Security violation reading file: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to read text file: {e}", exc_info=True)
            return None

    def save_json(
        self,
        data: Any,
        path: Path,
        indent: int = 2,
        ensure_ascii: bool = False
    ) -> bool:
        """
        Save data as JSON file with security validation.

        Args:
            data: Data to serialize
            path: Output file path
            indent: JSON indentation
            ensure_ascii: Whether to escape non-ASCII characters

        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate path (JSON files are allowed)
            validated_path = self._validate_path(path)

            # Enforce .json extension
            if validated_path.suffix.lower() != '.json':
                validated_path = validated_path.with_suffix('.json')
                validated_path = self._validate_path(validated_path)

            # Serialize to check size
            json_str = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)
            json_bytes = json_str.encode('utf-8')

            max_bytes = self.MAX_FILE_SIZE_MB * 1024 * 1024
            if len(json_bytes) > max_bytes:
                raise SecurityError(
                    f"JSON too large: {len(json_bytes)/1024/1024:.1f}MB > {self.MAX_FILE_SIZE_MB}MB"
                )

            # Create parent directory if needed
            validated_path.parent.mkdir(parents=True, exist_ok=True)

            # Write atomically
            temp_path = validated_path.with_suffix('.json.tmp')
            try:
                with temp_path.open('w', encoding='utf-8') as f:
                    f.write(json_str)
                temp_path.replace(validated_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

            logger.info(f"Saved JSON file: {validated_path}")
            return True

        except SecurityError as e:
            logger.error(f"Security violation saving JSON: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to save JSON file: {e}", exc_info=True)
            return False

    def load_json(self, path: Path) -> Any | None:
        """
        Load data from JSON file with security validation.

        Args:
            path: JSON file path

        Returns:
            Loaded data or None if failed
        """
        try:
            # Validate path
            validated_path = self._validate_path(path)

            if not validated_path.exists():
                logger.warning(f"JSON file not found: {validated_path}")
                return None

            # Check file size
            self._check_file_size(validated_path)

            # Load JSON with size limit
            with validated_path.open('r', encoding='utf-8') as f:
                # Read with size limit
                max_bytes = self.MAX_FILE_SIZE_MB * 1024 * 1024
                content = f.read(max_bytes + 1)
                if len(content) > max_bytes:
                    raise SecurityError("JSON file too large")

                data = json.loads(content)

            logger.debug(f"Loaded JSON file: {validated_path}")
            return data

        except SecurityError as e:
            logger.error(f"Security violation loading JSON: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load JSON file: {e}", exc_info=True)
            return None

    def export_csv(
        self,
        data: list[dict[str, Any]],
        path: Path,
        headers: list[str] | None = None
    ) -> bool:
        """
        Export data to CSV file with security validation.

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

            # Check row count limit
            if len(data) > self.MAX_CSV_ROWS:
                raise SecurityError(
                    f"Too many CSV rows: {len(data)} > {self.MAX_CSV_ROWS}"
                )

            # Validate path
            validated_path = self._validate_path(path)

            # Enforce .csv extension
            if validated_path.suffix.lower() != '.csv':
                validated_path = validated_path.with_suffix('.csv')
                validated_path = self._validate_path(validated_path)

            # Create parent directory if needed
            validated_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine headers
            if headers is None:
                headers = list(data[0].keys())

            # Validate headers (prevent CSV injection)
            for header in headers:
                if isinstance(header, str) and header.startswith(('=', '+', '-', '@')):
                    raise SecurityError(f"Potential CSV injection in header: {header}")

            # Write atomically
            temp_path = validated_path.with_suffix('.csv.tmp')
            try:
                with temp_path.open('w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()

                    # Validate and write each row
                    for row in data:
                        # Check for CSV injection in values
                        for key, value in row.items():
                            if isinstance(value, str) and value.startswith(('=', '+', '-', '@')):
                                # Escape formula injection by prepending with '
                                row[key] = "'" + value

                        writer.writerow(row)

                temp_path.replace(validated_path)

            finally:
                if temp_path.exists():
                    temp_path.unlink()

            logger.info(f"Exported {len(data)} rows to CSV: {validated_path}")
            return True

        except SecurityError as e:
            logger.error(f"Security violation exporting CSV: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}", exc_info=True)
            return False

    def delete_file(self, path: Path) -> bool:
        """
        Delete a file with security validation.

        Args:
            path: File path to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate path
            validated_path = self._validate_path(path)

            if not validated_path.exists():
                logger.debug(f"File already doesn't exist: {validated_path}")
                return True

            if not validated_path.is_file():
                raise SecurityError(f"Path is not a file: {validated_path}")

            validated_path.unlink()
            logger.info(f"Deleted file: {validated_path}")
            return True

        except SecurityError as e:
            logger.error(f"Security violation deleting file: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete file: {e}", exc_info=True)
            return False

    def validate_path_security(self, path: Path) -> bool:
        """
        Validate that a path is secure and within allowed boundaries.

        Args:
            path: Path to validate

        Returns:
            True if path is valid and secure
        """
        try:
            self._validate_path(path)
            return True
        except (SecurityError, Exception):
            return False


class SecurityError(Exception):
    """Raised when a security violation is detected."""
    pass
