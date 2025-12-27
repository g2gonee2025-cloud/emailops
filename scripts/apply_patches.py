#!/usr/bin/env python3
"""
Apply all generated patches sequentially.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PATCHES_DIR = PROJECT_ROOT / "patches"


def apply_patch(patch_file: Path) -> tuple[bool, str]:
    """Apply a single patch file."""
    try:
        result = subprocess.run(
            ["patch", "-p0", "-i", str(patch_file)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True, "Success"
        else:
            return False, result.stderr or result.stdout
    except Exception as e:
        return False, str(e)


def main():
    if not PATCHES_DIR.exists():
        print(f"Patches directory not found: {PATCHES_DIR}")
        sys.exit(1)

    patches = sorted(PATCHES_DIR.glob("*.diff"))
    print(f"Found {len(patches)} patches to apply\n")

    success_count = 0
    failed = []

    for i, patch in enumerate(patches, 1):
        print(f"[{i}/{len(patches)}] Applying {patch.name}...", end=" ")
        success, msg = apply_patch(patch)

        if success:
            print("✅")
            success_count += 1
        else:
            print(f"❌\n  Error: {msg[:200]}")
            failed.append((patch.name, msg))

    print(f"\n{'=' * 60}")
    print(f"Results: {success_count}/{len(patches)} succeeded")

    if failed:
        print(f"\nFailed patches ({len(failed)}):")
        for name, _ in failed:
            print(f"  - {name}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
