"""
Security Validators.

Implements §11.3 of the Canonical Blueprint.
"""

from __future__ import annotations

import logging
import re
import shlex
from pathlib import Path
from typing import List, Optional, Set

from cortex.common.types import Err, Ok, Result

logger = logging.getLogger(__name__)

# Basic email regex (can be replaced with a more robust library if needed)
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

# -----------------------------------------------------------------------------
# Prompt Injection Defense (§11.4)
# -----------------------------------------------------------------------------

# Comprehensive prompt injection patterns based on OWASP + real-world attacks
INJECTION_PATTERNS: list[str] = [
    # Classic jailbreak attempts
    "ignore previous instruction",
    "disregard earlier instruction",
    "override these rules",
    "forget all previous",
    "disregard all prior",
    "new instructions:",
    "updated instructions:",
    # System prompt manipulation
    "system prompt:",
    "### instruction",
    "### system",
    # Identity confusion
    "you are chatgpt",
    "you are now",
    "pretend you are",
    "as an ai language model",
    "as a large language model",
    # Code execution attempts
    "run code:",
    "execute:",
    "eval(",
    "exec(",
    "import os",
    "import sys",
    "subprocess",
    "__import__",
    # Mode switching
    "developer mode",
    "jailbreak",
    "debug mode",
    "admin mode",
    "god mode",
    "dan mode",
    # Prompt leaking
    "show me your prompt",
    "what are your instructions",
    "reveal your system prompt",
    "print your instructions",
    # Context injection
    "{{",
    "${",
    "<!--",
    "<script",
    "javascript:",
    # Role confusion markers
    "user:",
    "human:",
    "assistant:",
    # Base64/encoding tricks
    "base64",
    "decode(",
    "atob(",
    # Instruction termination
    "stop output",
    "end instructions",
    "ignore above",
]

# Compiled regex for performance
_INJECTION_PATTERN_RE = re.compile(
    "|".join(re.escape(p) for p in INJECTION_PATTERNS), re.IGNORECASE
)


def is_prompt_injection(text: str) -> bool:
    """
    Check if text contains potential prompt injection patterns.

    Args:
        text: Text to check

    Returns:
        True if injection pattern detected
    """
    if not text:
        return False
    return bool(_INJECTION_PATTERN_RE.search(text))


def sanitize_retrieved_content(text: str) -> str:
    """
    Sanitize retrieved content before sending to LLM.

    Removes lines that contain potential prompt injection patterns.
    This is defense-in-depth for content retrieved from user documents.

    Args:
        text: Raw retrieved text (email body, attachment content, etc.)

    Returns:
        Sanitized text with injection-like lines removed
    """
    if not text:
        return ""

    safe_lines: list[str] = []
    for line in text.splitlines():
        line_lower = line.strip().lower()

        # Skip empty lines (keep them for formatting)
        if not line_lower:
            safe_lines.append(line)
            continue

        # Check for injection patterns
        if _INJECTION_PATTERN_RE.search(line_lower):
            logger.debug("Removed injection-like line: %s", line[:50])
            continue

        # Skip lines that look like system/role markers
        if line_lower.startswith(
            ("system:", "assistant:", "user:", "instruction:", "### instruction", "```")
        ):
            logger.debug("Removed role-marker line: %s", line[:50])
            continue

        safe_lines.append(line)

    return "\n".join(safe_lines)


# Safe path characters (alphanumeric, dots, underscores, hyphens, slashes, colons for Windows)
_SAFE_PATH_CHARS = re.compile(r"[^a-zA-Z0-9._\-/\\: ]")

# Dangerous shell characters
_DANGEROUS_SHELL_CHARS: Set[str] = {"&", "|", ";", "`", "$", "\n", "\r", "\x00"}

# Default allowed file extensions (lowercase, with dot)
DEFAULT_ALLOWED_EXTENSIONS: Set[str] = {
    ".txt",
    ".json",
    ".md",
    ".csv",
    ".xml",
    ".yaml",
    ".yml",
    ".py",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".eml",
    ".msg",
    ".mbox",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".log",
    ".cfg",
    ".ini",
    ".env",
}


# -----------------------------------------------------------------------------
# Path Sanitization (§11.3)
# -----------------------------------------------------------------------------


def sanitize_path_input(path_input: str) -> str:
    """
    Sanitize a path input by removing dangerous characters.

    This is the FIRST line of defense - clean before validation.

    Removes:
    * Null bytes (prevents string termination attacks)
    * Shell metacharacters
    * Control characters

    Args:
        path_input: Raw path string from user input

    Returns:
        Cleaned path string with only safe characters
    """
    if not path_input:
        return ""

    # Remove null bytes first (critical security issue)
    cleaned = path_input.replace("\x00", "")

    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()

    # Remove unsafe characters, keeping only alphanumeric, dots, underscores, hyphens, slashes, spaces
    cleaned = _SAFE_PATH_CHARS.sub("", cleaned)

    return cleaned


# -----------------------------------------------------------------------------
# Symlink Detection (§11.3)
# -----------------------------------------------------------------------------


def is_dangerous_symlink(
    path: Path, allowed_roots: Optional[List[Path]] = None
) -> bool:
    """
    Check if a path is a symlink pointing outside allowed directories.

    TOCTOU Warning: This check can race with symlink creation.
    Callers should still handle OSError/PermissionError at use time.

    Args:
        path: Path to check
        allowed_roots: List of allowed root directories. If None, only checks
                      if symlink points to parent directories.

    Returns:
        True if the path is a dangerous symlink, False otherwise
    """
    try:
        if not path.is_symlink():
            return False

        # Resolve the symlink target
        resolved = path.resolve()

        # Check for parent traversal (symlink escapes its directory)
        try:
            # The symlink's parent should contain the resolved path
            symlink_parent = path.parent.resolve()
            resolved.relative_to(symlink_parent)
        except ValueError:
            # resolved is not under symlink_parent - potentially dangerous
            if allowed_roots:
                # Check if it's under any allowed root
                for root in allowed_roots:
                    try:
                        resolved.relative_to(root.resolve())
                        return False  # It's under an allowed root
                    except ValueError:
                        continue
                return True  # Not under any allowed root
            return True

        return False

    except (OSError, PermissionError) as e:
        logger.warning("Failed to check symlink %s: %s", path, e)
        return True  # Treat errors as dangerous


# -----------------------------------------------------------------------------
# Path Validation (§11.3 - Enhanced)
# -----------------------------------------------------------------------------


def validate_directory_result(
    path: str,
    must_exist: bool = True,
    allow_parent_traversal: bool = False,
    check_symlinks: bool = True,
    allowed_roots: Optional[List[Path]] = None,
) -> Result[Path, str]:
    """
    Validate a directory path.

    Requirements:
    * Must be absolute or resolvable to absolute
    * Must exist and be a directory (if must_exist=True)
    * No parent traversal (..) allowed in input string (by default)
    * Symlinks checked for escaping (by default)

    Args:
        path: Path string to validate
        must_exist: Whether the directory must exist
        allow_parent_traversal: Whether to allow .. in path
        check_symlinks: Whether to check for dangerous symlinks
        allowed_roots: List of allowed root directories for symlinks

    Returns:
        Result containing resolved Path or error message
    """
    if not allow_parent_traversal and ".." in path:
        return Err("Parent traversal (..) not allowed")

    try:
        # Create path object but don't resolve yet to allow symlink check
        p_unresolved = Path(path).expanduser()

        # Perform symlink check on the unresolved path
        if check_symlinks and is_dangerous_symlink(
            p_unresolved,
            allowed_roots=allowed_roots,
        ):
            return Err(f"Dangerous symlink detected: {path}")

        # Now resolve the path for existence/type checks
        p = p_unresolved.resolve()

        if must_exist:
            if not p.exists():
                return Err(f"Directory does not exist: {p}")
            if not p.is_dir():
                return Err(f"Path is not a directory: {p}")

        return Ok(p)
    except Exception as e:
        return Err(f"Invalid path: {e}")


def validate_file_result(
    path: str,
    must_exist: bool = True,
    allow_parent_traversal: bool = False,
    allowed_extensions: Optional[Set[str]] = None,
    check_symlinks: bool = True,
    allowed_roots: Optional[List[Path]] = None,
) -> Result[Path, str]:
    """
    Validate a file path.

    Requirements:
    * Must be absolute or resolvable to absolute
    * Must exist and be a file (if must_exist=True)
    * No parent traversal (..) allowed in input string (by default)
    * Extension must be in allowed list (if provided)
    * Symlinks checked for escaping (by default)

    Args:
        path: Path string to validate
        must_exist: Whether the file must exist
        allow_parent_traversal: Whether to allow .. in path
        allowed_extensions: Set of allowed extensions (lowercase, with dot)
        check_symlinks: Whether to check for dangerous symlinks
        allowed_roots: List of allowed root directories for symlinks

    Returns:
        Result containing resolved Path or error message
    """
    if not allow_parent_traversal and ".." in path:
        return Err("Parent traversal (..) not allowed")

    try:
        # Create path object but don't resolve yet to allow symlink check
        p_unresolved = Path(path).expanduser()

        if check_symlinks and is_dangerous_symlink(
            p_unresolved,
            allowed_roots=allowed_roots,
        ):
            return Err(f"Dangerous symlink detected: {path}")

        # Now resolve the path for existence/type checks
        p = p_unresolved.resolve()

        # Check extension if whitelist provided
        if allowed_extensions is not None:
            ext = p.suffix.lower()
            if ext not in allowed_extensions:
                return Err(
                    f"File extension '{ext}' not allowed. Allowed: {sorted(allowed_extensions)}"
                )

        if must_exist:
            if not p.exists():
                return Err(f"File does not exist: {p}")
            if not p.is_file():
                return Err(f"Path is not a file: {p}")

        return Ok(p)
    except Exception as e:
        return Err(f"Invalid path: {e}")


# -----------------------------------------------------------------------------
# Command Validation (§11.3 - Enhanced)
# -----------------------------------------------------------------------------


def validate_command_args(
    command: str,
    args: List[str],
    allowed_commands: Optional[List[str]] = None,
) -> Result[List[str], str]:
    """
    Validate command arguments.

    Requirements:
    * Command must be in allowed_commands (if provided)
    * Args must not contain shell injection characters
    * No null bytes allowed

    Args:
        command: The command to execute
        args: List of command arguments
        allowed_commands: Whitelist of allowed commands (None = all allowed)

    Returns:
        Result containing validated args or error message
    """
    if allowed_commands is not None and command not in allowed_commands:
        return Err(f"Command not allowed: {command}")

    # Check command for dangerous characters
    if any(c in _DANGEROUS_SHELL_CHARS for c in command):
        return Err(f"Command contains dangerous characters: {command}")

    # Check each argument
    for arg in args:
        if any(c in _DANGEROUS_SHELL_CHARS for c in arg):
            return Err(f"Argument contains dangerous characters: {arg}")

    return Ok(args)


def quote_shell_arg(arg: str) -> str:
    """
    Safely quote a shell argument for use in subprocess calls.

    Uses shlex.quote() for POSIX-compliant quoting.

    Args:
        arg: The argument to quote

    Returns:
        Safely quoted argument
    """
    return shlex.quote(arg)


# -----------------------------------------------------------------------------
# Email Validation (§11.3)
# -----------------------------------------------------------------------------


def validate_email_format(email: str) -> Result[str, str]:
    """
    Validate email format.

    Args:
        email: Email string to validate

    Returns:
        Result containing validated email or error message
    """
    email = email.strip()
    if not EMAIL_REGEX.match(email):
        return Err(f"Invalid email format: {email}")
    return Ok(email)


# -----------------------------------------------------------------------------
# Environment Variable Validation (§11.3)
# -----------------------------------------------------------------------------

_ENV_VAR_NAME_REGEX = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def validate_environment_variable(
    name: str, value: str
) -> Result[tuple[str, str], str]:
    """
    Validate an environment variable name and value.

    Requirements:
    * Name must match [A-Z_][A-Z0-9_]* pattern
    * Value must not contain null bytes

    Args:
        name: Environment variable name
        value: Environment variable value

    Returns:
        Result containing (name, value) tuple or error message
    """
    if not _ENV_VAR_NAME_REGEX.match(name):
        return Err(
            f"Invalid environment variable name: {name}. "
            "Must match pattern [A-Z_][A-Z0-9_]*"
        )

    if "\x00" in value:
        return Err("Environment variable value contains null byte")

    return Ok((name, value))
