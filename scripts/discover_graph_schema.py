#!/usr/bin/env python3
"""
Discovers the graph schema from a sample of conversations.

This script is a wrapper around the `cortex graph discover-schema` CLI command.
"""

import subprocess
import sys
from pathlib import Path


def find_project_root() -> Path:
    """Find the project root by searching upwards for a marker file."""
    current_path = Path(__file__).resolve()
    while not (current_path / "pyproject.toml").exists():
        if current_path.parent == current_path:
            raise FileNotFoundError("Could not find project root.")
        current_path = current_path.parent
    return current_path


def main():
    """Run the `cortex graph discover-schema` command."""
    try:
        project_root = find_project_root()
        cli_path = (project_root / "cli/src/cortex_cli/main.py").resolve()

        if not cli_path.exists():
            print(
                f"Error: Could not find CLI entrypoint at {cli_path}", file=sys.stderr
            )
            sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Forward all arguments to the CLI
    if sys.executable is None:
        print("Error: Could not find Python executable.", file=sys.stderr)
        sys.exit(1)

    command = [
        sys.executable,
        str(cli_path),
        "graph",
        "discover-schema",
        *sys.argv[1:],
    ]

    try:
        subprocess.run(command, check=True, text=True)
    except FileNotFoundError:
        print(
            f"Error: '{sys.executable}' not found.",
            file=sys.stderr,
        )
        sys.exit(1)
    except (OSError, PermissionError) as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(
            "Error: CLI command failed.",
            file=sys.stderr,
        )
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
