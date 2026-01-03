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


def _is_patch_applied(patch_path: Path) -> bool:
    """Check if a patch has already been applied by attempting a dry-run in reverse."""
    proc_reverse = subprocess.run(
        ["patch", "-p0", "--dry-run", "-R", "-i", str(patch_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return proc_reverse.returncode == 0


def _try_fix_patch_with_llm(
    patch_path: Path, file_path: str, content: str
) -> tuple[bool, str, bool]:
    """Attempt to fix a patch with an LLM and returns the new application status."""
    original_path = PROJECT_ROOT / file_path
    if not original_path.exists():
        return False, "Original file not found", False

    with original_path.open() as f:
        original = f.read()

    fixed_content = call_llm_fix(original, content)
    if not fixed_content or not is_valid_patch(fixed_content):
        return False, "LLM fix failed or produced invalid patch", False

    with patch_path.open("w") as f:
        f.write(fixed_content)

    success, msg = try_apply(patch_path, dry_run=True)
    return success, msg, True


def process_file_patches(
    file_path: str, patches: list[Path], use_llm: bool, dry_run: bool
) -> list[dict]:
    """Process all patches for a single file sequentially."""
    results = []
    patches.sort(
        key=lambda p: (
            int(re.search(r"__L(\d+)__", p.name).group(1))
            if re.search(r"__L(\d+)__", p.name)
            else 0
        )
    )

    for patch_path in patches:
        result = {"path": patch_path.name, "applied": False, "fixed": False}
        try:
            with patch_path.open() as f:
                content = f.read()

            if _is_patch_applied(patch_path):
                result.update({"applied": True, "note": "already_applied"})
                results.append(result)
                continue

            success, msg = try_apply(patch_path, dry_run=True)
            if not success and use_llm:
                success, msg, fixed = _try_fix_patch_with_llm(
                    patch_path, file_path, content
                )
                if fixed:
                    result["fixed"] = True

            if not success:
                result["error"] = msg[:80]
            elif not dry_run:
                applied_successfully, apply_msg = try_apply(patch_path, dry_run=False)
                result["applied"] = applied_successfully
                if not applied_successfully:
                    result["error"] = apply_msg[:80]
            else:
                result["applied"] = True  # Dry-run success

        except Exception as e:
            result["error"] = str(e)
        results.append(result)

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


def _process_future_result(future, counters: dict, log_file):
    """Process the result from a completed future, updating counters and logs."""
    try:
        results = future.result()
        for res in results:
            counters["total"] += 1
            if res.get("applied"):
                counters["applied"] += 1
            if res.get("fixed"):
                counters["fixed"] += 1
            if res.get("error"):
                counters["failed"] += 1
                log_file.write(f"FAILED {res['path']}: {res.get('error', '')}\n")

        if counters["total"] % 10 == 0:
            print(
                f"Progress: {counters['applied']} applied, {counters['failed']} failed..."
            )
            log_file.flush()
    except Exception as e:
        counters["failed"] += 1
        log_file.write(f"ERROR processing future: {e}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM fixing")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only validate, don't apply"
    )
    args = parser.parse_args()

    if not PATCHES_DIR.exists():
        print(f"Patches directory not found: {PATCHES_DIR}")
        sys.exit(1)

    patches = sorted(PATCHES_DIR.glob("*.diff"))
    print(f"Found {len(patches)} patches")

    # Group by file
    groups = group_patches_by_file(patches)
    print(f"Grouped into {len(groups)} target files")
    print(f"LLM Fix: {'disabled' if args.no_llm else FIX_MODEL}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'APPLY'}\n")

    counters = {"applied": 0, "fixed": 0, "failed": 0, "total": 0}
    max_workers = 5 if not args.no_llm else 10

    with (
        Path("patch_failures.log").open("w") as log_file,
        ThreadPoolExecutor(max_workers=max_workers) as executor,
    ):
        futures = [
            executor.submit(
                process_file_patches,
                file_path,
                file_patches,
                not args.no_llm,
                args.dry_run,
            )
            for file_path, file_patches in groups.items()
        ]

        for future in as_completed(futures):
            _process_future_result(future, counters, log_file)

    print(f"\n{'=' * 60}")
    print(f"Total patches: {len(patches)}")
    print(f"Applied: {counters['applied']}")
    print(f"Fixed by LLM: {counters['fixed']}")
    print(f"Failed: {counters['failed']}")


if __name__ == "__main__":
    main()
