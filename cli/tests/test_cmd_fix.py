import argparse
from unittest.mock import patch

from cortex_cli.cmd_fix import run_fixer, setup_fix_parser


def test_setup_fix_parser():
    """Test that the 'fix-issues' command is registered correctly."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_fix_parser(subparsers)
    args = parser.parse_args(["fix-issues"])
    assert hasattr(args, "func")


@patch("cortex_cli.cmd_fix.run_fixer")
def test_fix_issues_command(mock_run_fixer):
    """Test that the 'fix-issues' command calls the correct function."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_fix_parser(subparsers)
    args = parser.parse_args(["fix-issues", "--model", "test-model", "--max-workers", "5"])
    args.func(args)
    mock_run_fixer.assert_called_once_with(model="test-model", max_workers=5)
