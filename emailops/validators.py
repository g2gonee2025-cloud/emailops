#!/usr/bin/env python3
"""
Input validation and sanitization utilities for EmailOps.
Provides security checks for paths, commands, and user inputs.
"""

import re
import shlex
from pathlib import Path


def validate_directory_path(
    path: str | Path, must_exist: bool = True, allow_parent_traversal: bool = False
) -> tuple[bool, str]:
    """
    Validate directory path with security checks.

    Args:
        path: Directory path to validate
        must_exist: Whether the directory must exist
        allow_parent_traversal: Whether to allow '..' in paths (default: False for security)

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    try:
        # Convert to Path object and resolve
        p = Path(path).expanduser().resolve()

        # Security: Check for parent directory traversal attempts
        if not allow_parent_traversal:
            try:
                # Check if the resolved path contains attempts to escape
                path_str = str(path)
                if '..' in path_str or path_str.startswith('/..'):
                    return False, "Path traversal detected (.. not allowed)"
            except Exception:
                pass

        # Check existence if required
        if must_exist and not p.exists():
            return False, f"Directory does not exist: {p}"

        # Check if it's actually a directory (when it exists)
        if p.exists() and not p.is_dir():
            return False, f"Path exists but is not a directory: {p}"

        # Additional security: ensure path is absolute after resolution
        if not p.is_absolute():
            return False, f"Path must be absolute after resolution: {p}"

        return True, "Valid"

    except ValueError as e:
        return False, f"Invalid path value: {e}"
    except OSError as e:
        return False, f"OS error accessing path: {e}"
    except Exception as e:
        return False, f"Path validation error: {e}"


def validate_file_path(
    path: str | Path,
    must_exist: bool = True,
    allowed_extensions: list[str] | None = None,
    allow_parent_traversal: bool = False,
) -> tuple[bool, str]:
    """
    Validate file path with extension checks.

    Args:
        path: File path to validate
        must_exist: Whether the file must exist
        allowed_extensions: List of allowed extensions (e.g., ['.txt', '.pdf'])
        allow_parent_traversal: Whether to allow '..' in paths (default: False for security)

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    try:
        # Convert to Path object and resolve
        p = Path(path).expanduser().resolve()

        # Security: Check for parent directory traversal attempts
        if not allow_parent_traversal:
            try:
                path_str = str(path)
                if '..' in path_str or path_str.startswith('/..'):
                    return False, "Path traversal detected (.. not allowed)"
            except Exception:
                pass

        # Check existence if required
        if must_exist and not p.exists():
            return False, f"File does not exist: {p}"

        # Check if it's actually a file (when it exists)
        if p.exists() and not p.is_file():
            return False, f"Path exists but is not a file: {p}"

        # Check file extension if specified
        if allowed_extensions:
            ext = p.suffix.lower()
            if ext not in [e.lower() for e in allowed_extensions]:
                return False, f"File extension '{ext}' not in allowed list: {allowed_extensions}"

        # Additional security: ensure path is absolute after resolution
        if not p.is_absolute():
            return False, f"Path must be absolute after resolution: {p}"

        return True, "Valid"

    except ValueError as e:
        return False, f"Invalid path value: {e}"
    except OSError as e:
        return False, f"OS error accessing path: {e}"
    except Exception as e:
        return False, f"Path validation error: {e}"


def sanitize_path_input(path_input: str) -> str:
    """
    Sanitize user-provided path input by removing dangerous characters.

    Args:
        path_input: Raw path string from user

    Returns:
        Sanitized path string
    """
    if not path_input:
        return ""

    # Remove null bytes (security risk)
    sanitized = path_input.replace('\0', '')

    # Remove leading/trailing whitespace
    sanitized = sanitized.strip()

    # Remove any shell metacharacters for extra safety
    # Keep only: alphanumeric, ., _, -, /, \, :, space
    sanitized = re.sub(r'[^a-zA-Z0-9._\-/\\: ]', '', sanitized)

    return sanitized


def validate_command_args(
    command: str, args: list[str], allowed_commands: list[str] | None = None
) -> tuple[bool, str]:
    """
    Validate command and arguments for safe execution.

    Args:
        command: Command name (e.g., 'python', 'git')
        args: List of command arguments
        allowed_commands: Optional whitelist of allowed commands

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    # Check if command is in whitelist (if provided)
    if allowed_commands and command not in allowed_commands:
        return False, f"Command '{command}' not in allowed list"

    # Check for shell injection attempts in command
    dangerous_patterns = [';', '|', '&', '$', '`', '\n', '\r']
    for pattern in dangerous_patterns:
        if pattern in command:
            return False, f"Dangerous character '{pattern}' detected in command"

    # Validate each argument
    for arg in args:
        # Check for null bytes
        if '\0' in arg:
            return False, "Null byte detected in argument"

        # Check for command chaining attempts
        for pattern in dangerous_patterns:
            if pattern in arg:
                return False, f"Dangerous character '{pattern}' detected in argument: {arg}"

    return True, "Valid"


def quote_shell_arg(arg: str) -> str:
    """
    Safely quote a shell argument to prevent injection.

    Args:
        arg: Argument to quote

    Returns:
        Safely quoted argument string
    """
    return shlex.quote(str(arg))


def validate_project_id(project_id: str) -> tuple[bool, str]:
    """
    Validate Google Cloud project ID format.

    Args:
        project_id: GCP project ID to validate

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not project_id:
        return False, "Project ID cannot be empty"

    # GCP project IDs must:
    # - Be 6-30 characters long
    # - Contain only lowercase letters, numbers, and hyphens
    # - Start with a lowercase letter
    # - Not end with a hyphen

    if len(project_id) < 6 or len(project_id) > 30:
        return False, "Project ID must be 6-30 characters long"

    if not project_id[0].islower() or not project_id[0].isalpha():
        return False, "Project ID must start with a lowercase letter"

    if project_id.endswith('-'):
        return False, "Project ID cannot end with a hyphen"

    if not re.match(r'^[a-z][a-z0-9-]*$', project_id):
        return False, "Project ID can only contain lowercase letters, numbers, and hyphens"

    return True, "Valid"


def validate_environment_variable(name: str, value: str) -> tuple[bool, str]:
    """
    Validate environment variable name and value for security.

    Args:
        name: Environment variable name
        value: Environment variable value

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    # Validate name
    if not name:
        return False, "Environment variable name cannot be empty"

    if not re.match(r'^[A-Z_][A-Z0-9_]*$', name):
        return False, "Environment variable name must contain only uppercase letters, numbers, and underscores"

    # Check for null bytes in value
    if '\0' in value:
        return False, "Environment variable value contains null byte"

    return True, "Valid"
