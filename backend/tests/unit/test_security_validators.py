
from __future__ import annotations

import os
import unittest
from pathlib import Path

from cortex.security.validators import (
    is_dangerous_symlink,
    validate_directory_result,
    validate_file_result,
)


class TestSecurityValidators(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path("/tmp/test_security_validators")
        self.test_dir.mkdir(exist_ok=True)
        self.allowed_root = self.test_dir / "allowed"
        self.allowed_root.mkdir(exist_ok=True)
        self.disallowed_root = self.test_dir / "disallowed"
        self.disallowed_root.mkdir(exist_ok=True)
        self.test_file = self.allowed_root / "test_file.txt"
        self.test_file.touch()
        self.symlink = self.allowed_root / "symlink"

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.test_dir)

    def test_is_dangerous_symlink(self) -> None:
        # Create a symlink to a file outside the allowed root
        self.symlink.symlink_to(self.disallowed_root / "dangerous_file.txt")
        self.assertTrue(is_dangerous_symlink(self.symlink, [self.allowed_root]))

        # Create a symlink to a file inside the allowed root
        self.symlink.unlink()
        self.symlink.symlink_to(self.test_file)
        self.assertFalse(is_dangerous_symlink(self.symlink, [self.allowed_root]))

    def test_validate_directory_result(self) -> None:
        # Test with a valid directory
        result = validate_directory_result(str(self.allowed_root))
        self.assertTrue(result.is_ok())
        self.assertEqual(result.value, self.allowed_root.resolve())

        # Test with a non-existent directory
        result = validate_directory_result(str(self.allowed_root / "non_existent"))
        self.assertTrue(result.is_err())

        # Test with a dangerous symlink
        self.symlink.symlink_to(self.disallowed_root)
        result = validate_directory_result(str(self.symlink))
        self.assertTrue(result.is_err())

    def test_validate_file_result(self) -> None:
        # Test with a valid file
        result = validate_file_result(str(self.test_file))
        self.assertTrue(result.is_ok())
        self.assertEqual(result.value, self.test_file.resolve())

        # Test with a non-existent file
        result = validate_file_result(str(self.allowed_root / "non_existent.txt"))
        self.assertTrue(result.is_err())

        # Test with a dangerous symlink
        self.symlink.symlink_to(self.disallowed_root / "dangerous_file.txt")
        result = validate_file_result(str(self.symlink))
        self.assertTrue(result.is_err())


if __name__ == "__main__":
    unittest.main()
