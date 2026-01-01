import subprocess
import sys
from argparse import ArgumentParser, Namespace
from typing import Any

from cortex_cli.style import colorize


def setup_test_parser(subparsers: Any) -> ArgumentParser:
    """
    Setup the subparser for the 'test' command.
    """
    test_parser: ArgumentParser = subparsers.add_parser(
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
    print(colorize("RUNNING TESTS", "bold"))
    pytest_args = getattr(args, "pytest_args", None) or []
    if not isinstance(pytest_args, list):
        pytest_args = [str(pytest_args)]
    command = [sys.executable, "-m", "pytest", *pytest_args]
    try:
        result = subprocess.run(command, check=False)
    except KeyboardInterrupt:
        print(colorize("Tests interrupted", "yellow"))
        sys.exit(130)
    except FileNotFoundError:
        print(
            colorize(
                "Error: pytest not found. Please install it with 'pip install pytest'",
                "red",
            )
        )
        sys.exit(1)
    if result.returncode == 0:
        print(colorize("Tests passed", "green"))
        return
    print(colorize(f"Tests failed (exit code {result.returncode})", "red"))
    sys.exit(result.returncode)
