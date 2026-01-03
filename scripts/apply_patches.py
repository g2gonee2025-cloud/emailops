#!/usr/bin/env python3
"""
Apply all generated patches sequentially.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# --- Constants ---
# STYLE: Replace magic numbers with named constants
PATCH_TIMEOUT_SECONDS = 30
ERROR_PREVIEW_LENGTH = 200
SEPARATOR_WIDTH = 60
DEFAULT_STRIP_LEVEL = 1

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PATCHES_DIR = PROJECT_ROOT / "patches"
PATCH_EXECUTABLE = shutil.which("patch")


def is_patch_safe(patch_file: Path) -> bool:
    """Check if the patch file contains potentially unsafe paths."""
    try:
        with open(patch_file, errors="ignore") as f:
            for line in f:
                # Security: Check for absolute paths or directory traversal in patch headers
                if line.startswith("--- ") or line.startswith("+++ "):
                    parts = line.split(maxsplit=2)
                    if len(parts) > 1:
                        path_str = parts[1]
                        # Check for absolute paths or path traversal components
                        if path_str.startswith("/") or ".." in Path(path_str).parts:
                            print(
                                f"\n[SECURITY] Unsafe path detected in {patch_file.name}: '{path_str}'",
                                file=sys.stderr,
                            )
                            return False
    except FileNotFoundError:
        return False  # This will be handled more gracefully in apply_patch
    return True


def apply_patch(patch_file: Path, strip_level: int) -> tuple[bool, str]:
    """Apply a single patch file."""
    # SECURITY: Validate patch content before applying
    if not is_patch_safe(patch_file):
        return False, "Security risk: Patch contains absolute or traversal paths."

    try:
        # SECURITY: Use absolute path to 'patch' executable
        # PERFORMANCE: Prevent blocking on interactive prompts by redirecting stdin
        result = subprocess.run(
            [PATCH_EXECUTABLE, f"-p{strip_level}", "-i", str(patch_file)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=PATCH_TIMEOUT_SECONDS,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return True, "Success"
        else:
            return False, result.stderr or result.stdout
    # EXCEPTION_HANDLING: Catch specific exceptions instead of a broad one
    except FileNotFoundError:
        return False, f"'{PATCH_EXECUTABLE}' command not found. Please install 'patch'."
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {PATCH_TIMEOUT_SECONDS} seconds."
    except OSError as e:
        return False, f"An OS error occurred: {e}"


def main():
    # LOGIC_ERRORS: Make patch strip level configurable, defaulting to 1
    parser = argparse.ArgumentParser(
        description="Apply all patches in the patches directory."
    )
    parser.add_argument(
        "-p",
        "--strip",
        type=int,
        default=DEFAULT_STRIP_LEVEL,
        help=f"The strip level to pass to the patch command (default: {DEFAULT_STRIP_LEVEL}).",
    )
    args = parser.parse_args()

    # SECURITY: Exit if 'patch' command is not found
    if not PATCH_EXECUTABLE:
        print(
            "Error: 'patch' command not found in PATH. Please install it.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not PATCHES_DIR.exists():
        print(f"Patches directory not found: {PATCHES_DIR}")
        sys.exit(1)

    # STYLE: Discover both '.diff' and '.patch' files
    patches = sorted(
        list(PATCHES_DIR.glob("*.diff")) + list(PATCHES_DIR.glob("*.patch"))
    )
    print(f"Found {len(patches)} patches to apply\n")

    success_count = 0
    # PERFORMANCE: Only store the names of failed patches, not the full error messages
    failed_patches = []

    for i, patch in enumerate(patches, 1):
        print(f"[{i}/{len(patches)}] Applying {patch.name}...", end=" ")
        success, msg = apply_patch(patch, args.strip)

        if success:
            # STYLE: Remove Unicode emojis for better terminal compatibility
            print("[OK]")
            success_count += 1
        else:
            print(f"[FAIL]\n  Error: {msg[:ERROR_PREVIEW_LENGTH]}")
            failed_patches.append(patch.name)

    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print(f"Results: {success_count}/{len(patches)} succeeded")

    if failed_patches:
        print(f"\nFailed patches ({len(failed_patches)}):")
        for name in failed_patches:
            print(f"  - {name}")

    return 0 if not failed_patches else 1


if __name__ == "__main__":
    sys.exit(main())
