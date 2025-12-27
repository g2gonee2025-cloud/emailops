
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from cortex_cli.style import colorize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PATCHES_DIR = PROJECT_ROOT / "patches"

def apply_patch(patch_file: Path, dry_run: bool = False, revert: bool = False, patch_level: int = 1) -> tuple[bool, str]:
    """Apply a single patch file."""
    if not shutil.which("patch"):
        return False, "The 'patch' command is not installed or not in your PATH."

    command = ["patch", f"-p{patch_level}", "-i", str(patch_file)]
    if dry_run:
        command.append("--dry-run")
    if revert:
        command.append("-R")

    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True, "Success"
        else:
            return False, result.stderr or result.stdout
    except FileNotFoundError:
        return False, "The 'patch' command could not be found."
    except Exception as e:
        return False, str(e)

def cmd_patch_run(dry_run: bool, revert: bool, patch_level: int) -> None:
    """Run the patch command."""
    if not PATCHES_DIR.exists():
        print(colorize("ERROR:", "red") + f" Patches directory not found: {PATCHES_DIR}")
        sys.exit(1)

    patches = sorted(PATCHES_DIR.glob("*.diff"))
    if not patches:
        print(colorize("INFO:", "blue") + " No patches found to apply.")
        return

    print(f"Found {len(patches)} patches to apply\n")

    success_count = 0
    failed_patches = []

    for i, patch in enumerate(patches, 1):
        print(f"[{i}/{len(patches)}] Applying {patch.name}...", end=" ")
        success, msg = apply_patch(patch, dry_run=dry_run, revert=revert, patch_level=patch_level)

        if success:
            print(colorize("✅", "green"))
            success_count += 1
        else:
            print(colorize("❌", "red") + f"\n  Error: {msg}")
            failed_patches.append((patch.name, msg))

    print(f"\n{'=' * 60}")
    print(f"Results: {success_count}/{len(patches)} succeeded")

    if failed_patches:
        print(f"\nFailed patches ({len(failed_patches)}):")
        for name, _ in failed_patches:
            print(f"  - {name}")
        sys.exit(1)

def setup_patch_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up the parser for the 'patch' command."""
    patch_parser = subparsers.add_parser(
        "patch",
        help="Apply patches to the codebase",
        description="Apply all .diff files found in the 'patches' directory.",
    )
    patch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate patch application without making changes.",
    )
    patch_parser.add_argument(
        "--revert",
        action="store_true",
        help="Revert applied patches.",
    )
    patch_parser.add_argument(
        "--level",
        type=int,
        default=1,
        dest="patch_level",
        help="Set the patch level for the patch command (default: 1).",
    )
    patch_parser.set_defaults(func=lambda args: cmd_patch_run(args.dry_run, args.revert, args.patch_level))
