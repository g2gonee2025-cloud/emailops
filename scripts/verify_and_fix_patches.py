#!/usr/bin/env python3
"""
Verify and fix patches before applying.

Uses gpt-oss-120b to fix malformed patches, then applies them.
"""

import os
import re
import subprocess
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(".env")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PATCHES_DIR = PROJECT_ROOT / "patches"

# LLM Config
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DO_API_KEY")
BASE_URL = (os.getenv("LLM_ENDPOINT") or "https://inference.do-ai.run/v1").rstrip("/")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
FIX_MODEL = "openai-gpt-5"

FIX_PATCH_PROMPT = """
You are given a malformed unified diff patch. Fix it to be valid.

ORIGINAL FILE:
```
{original_content}
```

MALFORMED PATCH:
```
{patch_content}
```

RULES:
1. Output a valid unified diff that can be applied with `patch -p0`
2. Replace "@@ ... @@" with proper hunk headers like "@@ -10,5 +10,6 @@"
3. Keep --- and +++ headers exactly as they are
4. Include 3 lines of context around changes
5. Output ONLY the fixed diff, no explanations
"""


def extract_file_path(patch_content: str) -> str | None:
    """Extract target file path from patch."""
    for line in patch_content.split("\n"):
        if line.startswith("--- "):
            return line[4:].split("\t")[0].strip()
    return None


def is_valid_hunk_header(line: str) -> bool:
    """Check if line is a valid unified diff hunk header."""
    return bool(re.match(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", line))


def validate_patch(patch_content: str) -> tuple[bool, str]:
    """Validate patch format."""
    lines = patch_content.split("\n")
    if len(lines) < 3:
        return False, "Patch too short"

    has_from = any(line.startswith("--- ") for line in lines[:5])
    has_to = any(line.startswith("+++ ") for line in lines[:5])
    if not has_from or not has_to:
        return False, "Missing headers"

    has_valid_hunk = any(is_valid_hunk_header(line) for line in lines)
    if not has_valid_hunk:
        if any("@@ ... @@" in line for line in lines):
            return False, "Placeholder hunks"
        return False, "No valid hunks"

    return True, ""


def call_llm_fix(original_content: str, patch_content: str) -> str:
    """Call LLM to fix malformed patch."""
    prompt = FIX_PATCH_PROMPT.format(
        original_content=original_content[:10000], patch_content=patch_content
    )

    payload = {
        "model": FIX_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4000,
    }

    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions", json=payload, headers=HEADERS, timeout=120
        )
        if resp.status_code == 200:
            content = (
                resp.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            # Strip markdown
            lines = content.strip().split("\n")
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines)
        return ""
    except Exception:
        return ""


def try_apply_patch(patch_path: Path, dry_run: bool = True) -> tuple[bool, str]:
    """Try to apply a patch."""
    cmd = ["patch", "-p0"]
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(["-i", str(patch_path)])

    result = subprocess.run(
        cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30
    )
    return (result.returncode == 0, result.stderr or result.stdout)


def process_patch(patch_path: Path, apply: bool = False, use_llm: bool = True) -> dict:
    """Process a patch: validate, fix with LLM if needed, apply."""
    result = {"file": patch_path.name, "valid": False, "applied": False, "fixed": False}

    try:
        with patch_path.open() as f:
            content = f.read()

        file_path = extract_file_path(content)
        original_content = None
        if file_path:
            original_path = PROJECT_ROOT / file_path
            if original_path.exists():
                with original_path.open() as f:
                    original_content = f.read()

        is_valid, error = validate_patch(content)

        # Try LLM fix if format invalid
        if not is_valid and use_llm and original_content:
            fixed = call_llm_fix(original_content, content)
            if fixed:
                is_valid_fixed, _ = validate_patch(fixed)
                if is_valid_fixed:
                    content = fixed
                    with patch_path.open("w") as f:
                        f.write(fixed)
                    is_valid = True
                    result["fixed"] = True

        if not is_valid:
            result["error"] = error
            return result

        # Dry-run
        success, msg = try_apply_patch(patch_path, dry_run=True)

        # If dry-run fails, try LLM fix
        if not success and use_llm and original_content and not result.get("fixed"):
            fixed = call_llm_fix(original_content, content)
            if fixed:
                is_valid_fixed, _ = validate_patch(fixed)
                if is_valid_fixed:
                    with patch_path.open("w") as f:
                        f.write(fixed)
                    result["fixed"] = True
                    # Retry dry-run
                    success, msg = try_apply_patch(patch_path, dry_run=True)

        if not success:
            result["error"] = f"Dry-run failed: {msg[:80]}"
            result["valid"] = False
            return result

        result["valid"] = True

        # Apply
        if apply:
            success, msg = try_apply_patch(patch_path, dry_run=False)
            result["applied"] = success
            if not success:
                result["error"] = f"Apply failed: {msg[:80]}"

        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def main():
    import argparse
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply valid patches")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM fixing")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    patches = sorted(PATCHES_DIR.glob("*.diff"))
    print(f"Found {len(patches)} patches")
    print(f"LLM Fix: {'disabled' if args.no_llm else FIX_MODEL}")
    print("Workers: 10\n")

    valid = applied = fixed = 0
    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process_patch, p, args.apply, not args.no_llm): p
            for p in patches
        }

        for future in as_completed(futures):
            patch = futures[future]
            result = future.result()
            results.append(result)

            if result["valid"]:
                valid += 1
                status = "✅"
            else:
                status = "❌"

            if result.get("fixed"):
                fixed += 1
                status = "🔧"

            if result.get("applied"):
                applied += 1

            if args.verbose or not result["valid"]:
                print(f"{status} {patch.name}")
                if result.get("error"):
                    print(f"   → {result['error']}")

    print(f"\n{'=' * 60}")
    print(f"Valid: {valid}/{len(patches)}")
    print(f"Fixed by LLM: {fixed}")
    if args.apply:
        print(f"Applied: {applied}")


if __name__ == "__main__":
    main()
