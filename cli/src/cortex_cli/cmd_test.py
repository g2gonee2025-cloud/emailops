
import subprocess
import sys
from argparse import ArgumentParser, Namespace

from cortex_cli.style import colorize


def setup_test_parser(subparsers) -> ArgumentParser:
    """
    Setup the subparser for the 'test' command.
    """
    test_parser = subparsers.add_parser(
        "test",
        help="Run tests using pytest",
        description="""
Run the test suite using pytest.

This command acts as a wrapper around pytest, allowing you to pass additional
arguments to the pytest runner.

Examples:
  cortex test
  cortex test backend/tests/test_bare_except.py
  cortex test -k "bare_except"
        """,
    )
    test_parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Arguments to pass to pytest",
    )
    test_parser.set_defaults(func=run_tests)
    return test_parser


def run_tests(args: Namespace) -> None:
    """
    Execute pytest with the given arguments.
    """
    print(colorize("▶ RUNNING TESTS", "bold"))
    command = [sys.executable, "-m", "pytest"] + args.pytest_args
    try:
        subprocess.run(command, check=True)
        print(colorize("✓ Tests passed", "green"))
    except subprocess.CalledProcessError:
        print(colorize("✗ Tests failed", "red"))
        sys.exit(1)
    except FileNotFoundError:
        print(colorize("✗ Error: pytest not found. Please install it with 'pip install pytest'", "red"))
        sys.exit(1)
