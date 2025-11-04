"""
File Service Module

Handles all file I/O operations, abstracting file system interactions from the GUI.
"""

import asyncio
import csv
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FileService:
    """Service for handling file operations."""

    def __init__(self, export_root: str):
        """
        Initialize the file service.

        Args:
            export_root: Root directory for email exports
        """
        self.export_root = Path(export_root)

    def get_conversation_path(self, conv_id: str) -> Path | None:
        """
        Get and validate conversation path.

        Args:
            conv_id: Conversation ID

        Returns:
            Path if valid, None otherwise
        """
        try:
            root = self.export_root.resolve()
            candidate = (root / conv_id).resolve()

            # Check if path is within export root
            try:
                candidate.relative_to(root)
            except ValueError:
                return None

            if candidate.exists() and candidate.is_dir():
                # Check for required conversation files
                conv_file = candidate / "Conversation.txt"
                if conv_file.exists():
                    return candidate

            return None

        except Exception as e:
            logger.error(f"Failed to get conversation path: {e}")
            return None

    def save_text_file(self, content: str, path: Path, encoding: str = "utf-8") -> bool:
        """
        Save text content to a file.

        Args:
            content: Text content to save
            path: Output file path
            encoding: Text encoding

        Returns:
            True if successful, False otherwise
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding=encoding)
            logger.info(f"Saved text file: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save text file: {e}", exc_info=True)
            return False

    def save_binary_file(self, content: bytes, path: Path) -> bool:
        """
        Save binary content to a file.

        Args:
            content: Binary content to save
            path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            logger.info(f"Saved binary file: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save binary file: {e}", exc_info=True)
            return False

    def read_text_file(self, path: Path, encoding: str = "utf-8", max_chars: int | None = None) -> str | None:
        """
        Read text content from a file.

        Args:
            path: File path to read
            encoding: Text encoding
            max_chars: Optional maximum characters to read

        Returns:
            File content or None if failed
        """
        try:
            if not path.exists():
                logger.warning(f"File not found: {path}")
                return None

            content = path.read_text(encoding=encoding)
            if max_chars is not None and len(content) > max_chars:
                content = content[:max_chars]
            logger.debug(f"Read text file: {path}")
            return content
        except Exception as e:
            logger.error(f"Failed to read text file: {e}", exc_info=True)
            return None

    async def read_text_file_async(self, path: Path, encoding: str = "utf-8", max_chars: int | None = None) -> str | None:
        """
        Asynchronously read text content from a file.

        Args:
            path: File path to read
            encoding: Text encoding
            max_chars: Optional maximum characters to read

        Returns:
            File content or None if failed
        """
        try:
            if not path.exists():
                logger.warning(f"File not found: {path}")
                return None

            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(None, lambda: path.read_text(encoding=encoding))

            if max_chars is not None and len(content) > max_chars:
                content = content[:max_chars]
            logger.debug(f"Read text file asynchronously: {path}")
            return content
        except Exception as e:
            logger.error(f"Failed to read text file asynchronously: {e}", exc_info=True)
            return None

    def read_binary_file(self, path: Path) -> bytes | None:
        """
        Read binary content from a file.

        Args:
            path: File path to read

        Returns:
            File content or None if failed
        """
        try:
            if not path.exists():
                logger.warning(f"File not found: {path}")
                return None

            content = path.read_bytes()
            logger.debug(f"Read binary file: {path}")
            return content
        except Exception as e:
            logger.error(f"Failed to read binary file: {e}", exc_info=True)
            return None

    def save_json(
        self, data: Any, path: Path, indent: int = 2, ensure_ascii: bool = False
    ) -> bool:
        """
        Save data as JSON file.

        Args:
            data: Data to serialize
            path: Output file path
            indent: JSON indentation
            ensure_ascii: Whether to escape non-ASCII characters

        Returns:
            True if successful, False otherwise
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            logger.info(f"Saved JSON file: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON file: {e}", exc_info=True)
            return False

    def load_json(self, path: Path) -> Any | None:
        """
        Load data from JSON file.

        Args:
            path: JSON file path

        Returns:
            Loaded data or None if failed
        """
        try:
            if not path.exists():
                logger.warning(f"JSON file not found: {path}")
                return None

            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug(f"Loaded JSON file: {path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load JSON file: {e}", exc_info=True)
            return None

    def export_csv(
        self, data: list[dict[str, Any]], path: Path, headers: list[str] | None = None
    ) -> bool:
        """
        Export data to CSV file.

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

            path.parent.mkdir(parents=True, exist_ok=True)

            # Determine headers
            if headers is None:
                headers = list(data[0].keys())

            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(data)

            logger.info(f"Exported {len(data)} rows to CSV: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export CSV: {e}", exc_info=True)
            return False

    def read_csv(self, path: Path) -> list[dict[str, Any]] | None:
        """
        Read data from CSV file.

        Args:
            path: CSV file path

        Returns:
            List of dictionaries or None if failed
        """
        try:
            if not path.exists():
                logger.warning(f"CSV file not found: {path}")
                return None

            data = []
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(dict(row))

            logger.debug(f"Read {len(data)} rows from CSV: {path}")
            return data

        except Exception as e:
            logger.error(f"Failed to read CSV: {e}", exc_info=True)
            return None

    def copy_file(self, source: Path, destination: Path) -> bool:
        """
        Copy a file to a new location.

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

            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            logger.info(f"Copied {source} to {destination}")
            return True

        except Exception as e:
            logger.error(f"Failed to copy file: {e}", exc_info=True)
            return False

    def move_file(self, source: Path, destination: Path) -> bool:
        """
        Move a file to a new location.

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

            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            logger.info(f"Moved {source} to {destination}")
            return True

        except Exception as e:
            logger.error(f"Failed to move file: {e}", exc_info=True)
            return False

    def delete_file(self, path: Path) -> bool:
        """
        Delete a file.

        Args:
            path: File path to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            if not path.exists():
                logger.debug(f"File already doesn't exist: {path}")
                return True

            path.unlink()
            logger.info(f"Deleted file: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete file: {e}", exc_info=True)
            return False

    def delete_directory(self, path: Path, recursive: bool = False) -> bool:
        """
        Delete a directory.

        Args:
            path: Directory path to delete
            recursive: Whether to delete recursively

        Returns:
            True if successful, False otherwise
        """
        try:
            if not path.exists():
                logger.debug(f"Directory already doesn't exist: {path}")
                return True

            if recursive:
                shutil.rmtree(path)
            else:
                path.rmdir()

            logger.info(f"Deleted directory: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete directory: {e}", exc_info=True)
            return False

    def create_directory(self, path: Path) -> bool:
        """
        Create a directory.

        Args:
            path: Directory path to create

        Returns:
            True if successful, False otherwise
        """
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directory: {e}", exc_info=True)
            return False

    def list_files(
        self, directory: Path, pattern: str = "*", recursive: bool = False
    ) -> list[Path]:
        """
        List files in a directory.

        Args:
            directory: Directory to list
            pattern: File pattern to match
            recursive: Whether to search recursively

        Returns:
            List of file paths
        """
        try:
            if not directory.exists():
                logger.warning(f"Directory not found: {directory}")
                return []

            if recursive:
                files = list(directory.rglob(pattern))
            else:
                files = list(directory.glob(pattern))

            # Filter to only files
            files = [f for f in files if f.is_file()]

            logger.debug(f"Found {len(files)} files in {directory}")
            return files

        except Exception as e:
            logger.error(f"Failed to list files: {e}", exc_info=True)
            return []

    def get_file_info(self, path: Path) -> dict[str, Any]:
        """
        Get information about a file.

        Args:
            path: File path

        Returns:
            Dictionary with file information
        """
        info = {
            "exists": False,
            "path": str(path),
            "name": path.name,
            "size": 0,
            "modified": None,
            "created": None,
            "is_file": False,
            "is_dir": False,
        }

        try:
            if path.exists():
                info["exists"] = True
                info["is_file"] = path.is_file()
                info["is_dir"] = path.is_dir()

                stat = path.stat()
                info["size"] = stat.st_size
                info["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                info["created"] = datetime.fromtimestamp(stat.st_ctime).isoformat()
        except Exception as e:
            logger.warning(f"Failed to get file info: {e}")

        return info

    def validate_path_security(
        self, path: Path, allowed_root: Path | None = None
    ) -> bool:
        """
        Validate that a path is secure and within allowed boundaries.

        Args:
            path: Path to validate
            allowed_root: Optional root directory to restrict paths within

        Returns:
            True if path is valid and secure
        """
        try:
            # Resolve to absolute path
            resolved_path = path.resolve()

            # Check if within allowed root
            if allowed_root:
                allowed_root = allowed_root.resolve()
                try:
                    resolved_path.relative_to(allowed_root)
                except ValueError:
                    logger.warning(
                        f"Path {path} is outside allowed root {allowed_root}"
                    )
                    return False

            # Check for dangerous patterns
            dangerous_patterns = ["../", "..\\", "~", "$"]
            path_str = str(path)
            for pattern in dangerous_patterns:
                if pattern in path_str:
                    logger.warning(
                        f"Dangerous pattern '{pattern}' found in path: {path}"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Path validation failed: {e}")
            return False

    def create_backup(
        self, source: Path, backup_dir: Path | None = None
    ) -> Path | None:
        """
        Create a backup of a file or directory.

        Args:
            source: Source file or directory
            backup_dir: Optional backup directory (defaults to source.parent)

        Returns:
            Path to backup or None if failed
        """
        try:
            if not source.exists():
                logger.warning(f"Source not found for backup: {source}")
                return None

            # Determine backup location
            if backup_dir is None:
                backup_dir = source.parent

            # Create timestamped backup name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{source.stem}_backup_{timestamp}{source.suffix}"
            backup_path = backup_dir / backup_name

            # Perform backup
            if source.is_file():
                shutil.copy2(source, backup_path)
            else:
                shutil.copytree(source, backup_path)

            logger.info(f"Created backup: {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Failed to create backup: {e}", exc_info=True)
            return None
