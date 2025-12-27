#! /usr/bin/env python
"""
Discovers the graph schema from a sample of conversations.

This script is a wrapper around the `cortex graph discover-schema` CLI command.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the `cortex graph discover-schema` command."""
    # Construct the path to the CLI entrypoint
    cli_path = (
        Path(__file__).parent.parent / "cli/src/cortex_cli/main.py"
    ).resolve()

    if not cli_path.exists():
        print(
            f"Error: Could not find CLI entrypoint at {cli_path}", file=sys.stderr
        )
        sys.exit(1)

    # Forward all arguments to the CLI
    command = [
        sys.executable,
        str(cli_path),
        "graph",
        "discover-schema",
        *sys.argv[1:],
    ]

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print(
            f"Error: '{sys.executable}' not found.",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(
            f"Error executing command: {' '.join(command)}",
            file=sys.stderr,
        )
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
