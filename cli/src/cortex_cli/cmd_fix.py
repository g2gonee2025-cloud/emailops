from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cortex.security.validators import is_dangerous_symlink
from rich.console import Console

logger = logging.getLogger(__name__)

console = Console()
PATCHES_DIR = Path("patches")

# -----------------------------------------------------------------------------
# Symlink Fixer (from HEAD)
# -----------------------------------------------------------------------------


def _scan_for_insecure_symlinks(roots: list[Path]) -> int:
    """Scan for and report insecure symlinks."""
    insecure_symlinks_found = 0
    allowed_roots = [r.resolve() for r in roots]

    console.print(f"Scanning for insecure symlinks in: {[str(r) for r in roots]}")
    console.print(f"Allowed destination roots: {[str(r) for r in allowed_roots]}")

    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                path = Path(dirpath) / name
                if path.is_symlink():
                    if is_dangerous_symlink(path, allowed_roots=allowed_roots):
                        insecure_symlinks_found += 1
                        console.print(
                            f"[bold red]Insecure symlink found:[/] {path} -> {os.readlink(path)}"
                        )

    return insecure_symlinks_found


def run_fix_insecure_symlinks(args: argparse.Namespace) -> int:
    """Run the insecure symlinks fix."""
    try:
        scan_path = args.path or Path.cwd()
        scan_paths = [Path(scan_path).expanduser().resolve()]
        insecure_symlinks_found = _scan_for_insecure_symlinks(scan_paths)
        if insecure_symlinks_found > 0:
            console.print(
                f"\n[bold yellow]Found {insecure_symlinks_found} insecure symlink(s).[/]"
            )
            console.print(
                "To fix, remove the symlink and replace with a copy of the file if needed."
            )
            return 1
        else:
            console.print("\n[bold green]No insecure symlinks found.[/]")
            return 0
    except Exception as e:
        logger.error(f"Error scanning for insecure symlinks: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/] {e}")
        return 1


def setup_fix_parser(
    parser: argparse._SubParsersAction,
) -> None:
    """Set up the fix command parser."""
    fix_parser = parser.add_parser("fix", help="Fix common issues")
    fix_subparsers = fix_parser.add_subparsers(dest="fix_command", help="Fix commands")

    # Command: insecure-symlinks
    insecure_symlinks_parser = fix_subparsers.add_parser(
        "insecure-symlinks", help="Scan for and report insecure symlinks"
    )
    insecure_symlinks_parser.add_argument(
        "--path", type=str, help="The path to scan. Defaults to the current directory."
    )
    insecure_symlinks_parser.set_defaults(func=run_fix_insecure_symlinks)
