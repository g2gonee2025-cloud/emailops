#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

"""
Input validation and sanitization utilities for EmailOps.
Provides security checks for paths, commands, and user inputs.

NOTE: This module preserves the existing public API (functions returning
(tuple[bool, str])) and adds *ergonomic* variants that return normalized
values. See `validate_directory_path_info` and `validate_file_path_info`.
"""



T = TypeVar("T")

# -------------------------
# Pre-compiled regex patterns for performance
# -------------------------

# Email validation pattern (RFC 5322 compliant)
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

# GCP project ID pattern
PROJECT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")

# Environment variable name patterns
ENV_VAR_UPPERCASE_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
ENV_VAR_MIXED_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Dangerous characters for path sanitization
DANGEROUS_PATH_CHARS = {"\0", "\r", "\n", "|", "&", ";", "$", "`", "<", ">", "(", ")", "[", "]", "{", "}"}

# Shell dangerous patterns
SHELL_DANGEROUS_PATTERNS = frozenset([";", "|", "&", "$", "`", "\n", "\r", "\0"])

# Blocklist of dangerous commands
DANGEROUS_COMMANDS = frozenset(["rm", "mv", "dd", "mkfs", "shutdown", "reboot", "halt"])

# URL validation pattern
URL_PATTERN = re.compile(
    r"^(?:http|https|ftp)://"  # Protocol
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # Domain
    r"localhost|"  # Localhost
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # IP address
    r"(?::\d+)?"  # Optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)

# SQL identifier pattern (basic - alphanumeric plus underscore)
SQL_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# JSON key pattern
JSON_KEY_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_\-\.]*$")


# -------------------------
# Existing public API (unchanged)
# -------------------------


def validate_directory_path(
    path: str | Path, must_exist: bool = True, allow_parent_traversal: bool = False
) -> tuple[bool, str]:
    """Validate directory path with security checks.

    MEDIUM #22: TOCTOU vulnerability note - files can change between validation and use.
    Callers should handle exceptions during file access as proper mitigation.
    Validation provides early feedback but doesn't guarantee availability at use time.

    Args:
        path: Directory path to validate
        must_exist: Whether the directory must exist
        allow_parent_traversal: Whether to allow '..' in paths (default: False for security)

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    try:
        # Convert to Path object (do not resolve yet so we can inspect raw segments)
        p_raw = Path(path).expanduser()

        # Security: Check for explicit parent-directory traversal segments ('..')
        # Use Path.parts so that only real path segments are inspected (not substrings in filenames)
        if not allow_parent_traversal and any(part == ".." for part in p_raw.parts):
            return False, "Path traversal detected ('..' segments are not allowed)"

        # Now resolve to an absolute canonical path for subsequent checks
        p = p_raw.resolve()

        # Check existence if required (TOCTOU warning: state may change after check)
        if must_exist and not p.exists():
            return False, f"Directory does not exist: {p}"

        # Check if it's actually a directory (when it exists)
        if p.exists() and not p.is_dir():
            return False, f"Path exists but is not a directory: {p}"

        # Additional security: ensure path is absolute after resolution
        if not p.is_absolute():
            return False, f"Path must be absolute after resolution: {p}"

        # Check for symlink attacks
        if p.exists() and p.is_symlink():
            real_path = p.resolve()
            if real_path != p:
                return False, f"Symlink detected pointing to: {real_path}"

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
    """Validate file path with extension checks.

    NOTE: To avoid TOCTOU issues, callers should use the normalized path
    returned by validate_file_path_info() immediately after validation.

    Args:
        path: File path to validate
        must_exist: Whether the file must exist
        allowed_extensions: List of allowed extensions (e.g., ['.txt', '.pdf'])
        allow_parent_traversal: Whether to allow '..' in paths (default: False for security)

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    try:
        # Convert to Path object (do not resolve yet so we can inspect raw segments)
        p_raw = Path(path).expanduser()

        # Security: Check for explicit parent-directory traversal segments ('..')
        # Use Path.parts so that only real path segments are inspected (not substrings in filenames)
        if not allow_parent_traversal and any(part == ".." for part in p_raw.parts):
            return False, "Path traversal detected ('..' segments are not allowed)"

        # Now resolve to an absolute canonical path for subsequent checks
        p = p_raw.resolve()

        # Check existence if required (TOCTOU warning: state may change after check)
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

        # Check for symlink attacks
        if p.exists() and p.is_symlink():
            real_path = p.resolve()
            if real_path != p:
                return False, f"Symlink detected pointing to: {real_path}"

        return True, "Valid"

    except ValueError as e:
        return False, f"Invalid path value: {e}"
    except OSError as e:
        return False, f"OS error accessing path: {e}"
    except Exception as e:
        return False, f"Path validation error: {e}"


def sanitize_path_input(path_input: str) -> str:
    """Sanitize user-provided path input by removing dangerous characters.

    Args:
        path_input: Raw path string from user

    Returns:
        Sanitized path string
    """
    if not path_input:
        return ""

    # Remove dangerous characters while preserving valid path characters
    # This is less aggressive than before - allows more valid paths
    sanitized = "".join(c for c in path_input if c not in DANGEROUS_PATH_CHARS)

    # Remove leading/trailing whitespace
    sanitized = sanitized.strip()

    return sanitized


def validate_command_args(command: str, args: list[str], allowed_commands: list[str] | None = None) -> tuple[bool, str]:
    """Validate command and arguments for safe execution.

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

    # If no allowlist, check against a blocklist of dangerous commands
    if not allowed_commands and command in DANGEROUS_COMMANDS:
        return False, f"Command '{command}' is blocked for security reasons"

    # Check for shell injection attempts in command using pre-compiled set
    if any(pattern in command for pattern in SHELL_DANGEROUS_PATTERNS):
        return False, "Dangerous character detected in command"

    # Validate each argument
    for arg in args:
        # Check for dangerous patterns using pre-compiled set
        if any(pattern in arg for pattern in SHELL_DANGEROUS_PATTERNS):
            return False, f"Dangerous character detected in argument: {arg}"

    return True, "Valid"


def quote_shell_arg(arg: str) -> str:
    """Safely quote a shell argument to prevent injection.

    This uses POSIX quoting via :func:`shlex.quote`. On Windows (`cmd.exe`/PowerShell),
    prefer calling :func:`subprocess.run` with ``shell=False`` and an argument list.

    Args:
        arg: Argument to quote

    Returns:
        Safely quoted argument string
    """
    return shlex.quote(str(arg))


def validate_project_id(project_id: str) -> tuple[bool, str]:
    """Validate Google Cloud project ID format.

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

    if project_id.endswith("-"):
        return False, "Project ID cannot end with a hyphen"

    # Use pre-compiled regex pattern
    if not PROJECT_ID_PATTERN.match(project_id):
        return False, "Project ID can only contain lowercase letters, numbers, and hyphens"

    return True, "Valid"


def validate_environment_variable(name: str, value: str, *, require_uppercase: bool = True) -> tuple[bool, str]:
    """Validate environment variable name and value for security.

    Args:
        name: Environment variable name
        value: Environment variable value
        require_uppercase: When True (default), enforce uppercase-only names.
            When False, allow mixed/lowercase names.

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    # Validate name
    if not name:
        return False, "Environment variable name cannot be empty"

    # Use pre-compiled regex patterns
    pattern = ENV_VAR_UPPERCASE_PATTERN if require_uppercase else ENV_VAR_MIXED_PATTERN
    if not pattern.match(name):
        policy = (
            "uppercase letters, numbers, and underscores" if require_uppercase else "letters, numbers, and underscores"
        )
        return False, f"Environment variable name must contain only {policy}"

    # Check for null bytes in value
    if "\0" in value:
        return False, "Environment variable value contains null byte"

    return True, "Valid"


def validate_email_format(email: str) -> tuple[bool, str]:
    """Validate email address format.

    Performs basic RFC 5322-compliant validation with additional
    security checks for email length and format.

    NOTE: For production use, consider using the email-validator library
    for more comprehensive RFC compliance and internationalization support.

    Args:
        email: Email address to validate

    Returns:
        Tuple of (is_valid: bool, message: str)

    Example:
        >>> validate_email_format("user@example.com")
        (True, 'Valid')
        >>> validate_email_format("invalid email")
        (False, 'Invalid email format: invalid email')
    """
    if not email or not email.strip():
        return False, "Email cannot be empty"

    email = email.strip()

    # Use pre-compiled regex pattern
    if not EMAIL_PATTERN.match(email):
        return False, f"Invalid email format: {email}"

    # RFC 5321 limit: 320 characters total
    if len(email) > 320:
        return False, "Email address too long (max 320 chars)"

    # Local part (before @) should not exceed 64 characters
    local_part = email.split("@")[0]
    if len(local_part) > 64:
        return False, "Email local part too long (max 64 chars)"

    # Domain part validation
    domain_part = email.split("@")[1] if "@" in email else ""
    if len(domain_part) > 253:
        return False, "Email domain too long (max 253 chars)"

    return True, "Valid"


# -------------------------
# New ergonomic, typed variants (additive)
# -------------------------


@dataclass(frozen=True)
class ValidationResult(Generic[T]):
    ok: bool
    msg: str
    value: T | None = None


def sanitize_path_input_report(path_input: str) -> tuple[str, bool]:
    """Return (sanitized_value, changed_flag) for transparency."""
    sanitized = sanitize_path_input(path_input)
    return sanitized, sanitized != (path_input or "")


def _maybe_expand_vars(p: str | Path, expand_vars: bool) -> str:
    s = str(p)
    return os.path.expandvars(s) if expand_vars else s


def validate_directory_path_info(
    path: str | Path,
    *,
    must_exist: bool = True,
    allow_parent_traversal: bool = False,
    expand_vars: bool = False,
) -> ValidationResult[Path]:
    """Validate a directory path and return a normalized Path on success.

    Returns a :class:`ValidationResult` carrying the resolved absolute :class:`Path`
    when validation succeeds
    otherwise ``value`` is ``None``.
    """
    expanded = _maybe_expand_vars(path, expand_vars)
    ok, msg = validate_directory_path(expanded, must_exist=must_exist, allow_parent_traversal=allow_parent_traversal)
    if not ok:
        return ValidationResult(False, msg, None)
    return ValidationResult(True, "Valid", Path(expanded).expanduser().resolve())


def validate_file_path_info(
    path: str | Path,
    *,
    must_exist: bool = True,
    allowed_extensions: list[str] | None = None,
    allowed_multi_suffixes: list[str] | None = None,
    allow_parent_traversal: bool = False,
    expand_vars: bool = False,
) -> ValidationResult[Path]:
    """Validate a file path with support for single- and multi-suffix allowlists.

    This wrapper accepts both ``allowed_extensions`` (single-suffix, e.g. ".pdf")
    and ``allowed_multi_suffixes`` (e.g. ".tar.gz"). If either list is provided,
    the file is accepted when it matches **any** of the provided patterns.
    """
    expanded = _maybe_expand_vars(path, expand_vars)
    # Bypass extension checks in the base function to perform union logic here.
    ok, msg = validate_file_path(
        expanded, must_exist=must_exist, allowed_extensions=None, allow_parent_traversal=allow_parent_traversal
    )
    if not ok:
        return ValidationResult(False, msg, None)

    p = Path(expanded).expanduser().resolve()
    if allowed_extensions or allowed_multi_suffixes:
        ext_ok = False
        if allowed_extensions:
            ext_ok = p.suffix.lower() in [e.lower() for e in allowed_extensions]
        multi_ok = False
        if allowed_multi_suffixes:
            combined = "".join(p.suffixes).lower()
            multi_ok = any(combined == s.lower() for s in allowed_multi_suffixes)
        if not (ext_ok or multi_ok):
            return ValidationResult(
                False, f"File extension '{p.suffix}' not allowed (combined suffix '{''.join(p.suffixes)}')", None
            )

    return ValidationResult(True, "Valid", p)
