#!/usr/bin/env python3
"""
Intelligent Patch Applier.

Groups patches by target file and applies them sequentially to avoid conflicts.
Uses LLM fixing layer (openai-gpt-5) for malformed patches.
"""

import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests
from dotenv import load_dotenv

load_dotenv(".env")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PATCHES_DIR = PROJECT_ROOT / "patches_per_issue"

# LLM Config for fixing
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DO_API_KEY")
BASE_URL = (os.getenv("LLM_ENDPOINT") or "https://inference.do-ai.run/v1").rstrip("/")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
FIX_MODEL = "openai-gpt-5"

# File locks to prevent concurrent modification
file_locks: dict[str, Lock] = defaultdict(Lock)


def extract_file_path(patch_content: str) -> str | None:
    """Extract target file path from patch."""
    for line in patch_content.split("\n"):
        if line.startswith("--- "):
            return line[4:].split("\t")[0].strip()
    return None


def is_valid_patch(content: str) -> bool:
    """Check if patch has valid hunk headers."""
    return bool(
        re.search(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", content, re.MULTILINE)
    )


def call_llm_fix(original: str, patch: str) -> str:
    """Use LLM to fix malformed patch."""
    prompt = f"""Fix this malformed unified diff patch.

ORIGINAL FILE:
```
{original[:8000]}
```

MALFORMED PATCH:
```
{patch}
```

Output ONLY a valid unified diff. Include proper @@ hunk headers like:
@@ -10,5 +10,6 @@
"""
    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": FIX_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 2000,
            },
            headers=HEADERS,
            timeout=120,
        )
        if resp.status_code == 200:
            content = (
                resp.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            # Extract code block if present
            match = re.search(r"```(?:diff)?\s*\n(.*?)\n```", content, re.DOTALL)
            if match:
                content = match.group(1)

            # Determine start of diff
            # Look for "--- " at start of a line
            diff_start = re.search(r"^--- .*$", content, re.MULTILINE)
            if diff_start:
                content = content[diff_start.start() :]

            return content.strip()
    except Exception:
        pass
    return ""


def try_apply(patch_path: Path, dry_run: bool = True) -> tuple[bool, str]:
    """Try to apply a patch."""
    cmd = ["patch", "-p0"]
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(["-i", str(patch_path)])

    result = subprocess.run(
        cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30
    )
    return result.returncode == 0, result.stderr or result.stdout


def _handle_llm_fix(file_path: str, patch_path: Path, content: str) -> tuple[bool, str]:
    """Attempt to fix a patch using an LLM."""
    original_path = PROJECT_ROOT / file_path
    if not original_path.exists():
        return False, "Original file not found for LLM fix."

    with original_path.open() as f:
        original_content = f.read()

    fixed_patch = call_llm_fix(original_content, content)
    if fixed_patch and is_valid_patch(fixed_patch):
        with patch_path.open("w") as f:
            f.write(fixed_patch)
        return try_apply(patch_path, dry_run=True)

    return False, "LLM fix failed or produced invalid patch."


def _process_single_patch(patch_path: Path, file_path: str, use_llm: bool, dry_run: bool) -> dict:
    """Process a single patch file."""
    result = {"path": patch_path.name, "applied": False, "fixed": False}
    try:
        with patch_path.open() as f:
            content = f.read()

        # Check if already applied (reverse dry-run succeeds)
        proc_reverse = subprocess.run(
            ["patch", "-p0", "--dry-run", "-R", "-i", str(patch_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc_reverse.returncode == 0:
            result.update({"applied": True, "note": "already_applied"})
            return result

        # Try dry-run first
        success, msg = try_apply(patch_path, dry_run=True)

        # If fails and LLM enabled, try to fix
        if not success and use_llm:
            fixed_success, fix_msg = _handle_llm_fix(file_path, patch_path, content)
            if fixed_success:
                result["fixed"] = True
                success, msg = True, "Applied after LLM fix."

        if not success:
            result["error"] = msg[:80]
            return result

        # Actually apply if not a global dry-run
        if not dry_run:
            apply_success, apply_msg = try_apply(patch_path, dry_run=False)
            result["applied"] = apply_success
            if not apply_success:
                result["error"] = apply_msg[:80]
        else:
            result["applied"] = True  # Mark as success for dry-run reporting

    except Exception as e:
        result["error"] = str(e)

    return result


def process_file_patches(
    file_path: str, patches: list[Path], use_llm: bool, dry_run: bool
) -> list[dict]:
    """Process all patches for a single file sequentially."""
    patches.sort(
        key=lambda p: (
            int(re.search(r"__L(\d+)__", p.name).group(1))
            if re.search(r"__L(\d+)__", p.name)
            else 0
        )
    )
    results = [
        _process_single_patch(patch_path, file_path, use_llm, dry_run)
        for patch_path in patches
    ]
    return results


def group_patches_by_file(patches: list[Path]) -> dict[str, list[Path]]:
    """Group patches by their target file."""
    groups = defaultdict(list)
    for p in patches:
        try:
            with p.open() as f:
                content = f.read()
            file_path = extract_file_path(content)
            if file_path:
                groups[file_path].append(p)
        except Exception:
            pass
    return groups


def _parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Intelligent Patch Applier")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM fixing layer")
    parser.add_argument("--dry-run", action="store_true", help="Only validate patches, do not apply them")
    return parser.parse_args()


def _prepare_patches():
    """Find and group patches by target file."""
    if not PATCHES_DIR.exists():
        print(f"Patches directory not found: {PATCHES_DIR}")
        sys.exit(1)
    patches = sorted(PATCHES_DIR.glob("*.diff"))
    print(f"Found {len(patches)} patches")
    groups = group_patches_by_file(patches)
    print(f"Grouped into {len(groups)} target files")
    return patches, groups


def _process_patches_in_parallel(groups, use_llm, dry_run):
    """Process patch groups in parallel using a thread pool."""
    applied = fixed = failed = 0
    total_processed = 0
    max_workers = 5 if use_llm else 10

    with Path("patch_failures.log").open("w") as log_file, ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file_patches, file_path, file_patches, use_llm, dry_run): file_path
            for file_path, file_patches in groups.items()
        }

        for future in as_completed(futures):
            results = future.result()
            for res in results:
                total_processed += 1
                if res.get("applied"):
                    applied += 1
                if res.get("fixed"):
                    fixed += 1
                if res.get("error"):
                    failed += 1
                    log_file.write(f"FAILED {res['path']}: {res.get('error', '')}\n")

            if total_processed % 10 == 0 and total_processed > 0:
                print(f"Progress: {applied} applied, {failed} failed...")
                log_file.flush()

    return applied, fixed, failed


def _print_summary(total_patches, applied, fixed, failed):
    """Print the final summary of the patch application process."""
    print(f"\n{'=' * 60}")
    print(f"Total patches: {total_patches}")
    print(f"Applied: {applied}")
    print(f"Fixed by LLM: {fixed}")
    print(f"Failed: {failed}")


def main():
    """Main function to orchestrate the patch application process."""
    args = _parse_args()
    patches, groups = _prepare_patches()

    print(f"LLM Fix: {'disabled' if args.no_llm else FIX_MODEL}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'APPLY'}\n")

    applied, fixed, failed = _process_patches_in_parallel(groups, not args.no_llm, args.dry_run)
    _print_summary(len(patches), applied, fixed, failed)


if __name__ == "__main__":
    main()
