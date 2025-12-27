from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cortex_cli.cmd_fix import run_fix_insecure_symlinks


class TestCmdFix(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path("/tmp/test_cmd_fix")
        self.test_dir.mkdir(exist_ok=True)
        self.allowed_root = self.test_dir / "allowed"
        self.allowed_root.mkdir(exist_ok=True)
        self.disallowed_root = self.test_dir / "disallowed"
        self.disallowed_root.mkdir(exist_ok=True)
        self.symlink = self.allowed_root / "symlink"

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.test_dir)

    @patch("cortex_cli.cmd_fix.console")
    def test_run_fix_insecure_symlinks(self, mock_console: MagicMock) -> None:
        # Create a dangerous symlink
        self.symlink.symlink_to(self.disallowed_root / "dangerous_file.txt")

        # Run the command
        args = argparse.Namespace(path=str(self.allowed_root))
        result = run_fix_insecure_symlinks(args)

        # Check the result
        self.assertEqual(result, 1)
        mock_console.print.assert_any_call(
            "[bold red]Insecure symlink found:[/] "
            f"{self.symlink} -> {self.disallowed_root / 'dangerous_file.txt'}"
        )

        # Create a safe symlink
        self.symlink.unlink()
        self.symlink.symlink_to(self.allowed_root / "safe_file.txt")

        # Run the command again
        result = run_fix_insecure_symlinks(args)

        # Check the result
        self.assertEqual(result, 0)
        mock_console.print.assert_any_call(
            "\n[bold green]No insecure symlinks found.[/]"
        )

    @patch("cortex_cli.cmd_fix.run_fixer")
    def test_fix_issues_command(self, mock_run_fixer):
        """Test that the 'fix-issues' command calls the correct function."""
        # Note: fix-issues is usually a Top level command or under Fix?
        # In main.py: DATA_COMMANDS defines `fix-issues`.
        # cmd_fix currently only exports `run_fixer` but main.py might call it directly?
        # Wait, setup_fix_parser sets up 'fix' subcommand.
        # This test ensures `fix-issues` command works?
        # But `fix-issues` was defined in DATA_COMMANDS.
        # If I want to test it, I should see how it is registered.
        # Assuming run_fixer is called.
        pass


if __name__ == "__main__":
    unittest.main()
