#!/usr/bin/env python3
"""
Use LLM to fix malformed patches.

Sends each broken patch to gpt-oss-120b to generate a proper unified diff.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(".env")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("patch_fixer")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PATCHES_DIR = PROJECT_ROOT / "patches"
FIXED_DIR = PROJECT_ROOT / "patches_fixed"
FIXED_DIR.mkdir(exist_ok=True)

API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DO_API_KEY")
BASE_URL = (os.getenv("LLM_ENDPOINT") or "https://inference.do-ai.run/v1").rstrip("/")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

MAX_WORKERS = 10
FIX_MODEL = "openai-gpt-oss-120b"

FIX_PATCH_PROMPT = """
You are given a malformed unified diff patch that has invalid hunk headers like "@@ ... @@".
Your task is to fix it into a valid unified diff format.

RULES:
1. Keep the same --- and +++ headers
2. Replace "@@ ... @@" with proper hunk headers like "@@ -1,10 +1,12 @@"
3. The hunk header format is: @@ -START,COUNT +START,COUNT @@
4. Analyze the - and + lines to determine correct line numbers
5. Output ONLY the fixed unified diff, nothing else

<original_file_content>
{original_content}
</original_file_content>

<malformed_patch>
{patch_content}
</malformed_patch>

Output the corrected unified diff:
"""


def get_file_path_from_patch(patch_content: str) -> str | None:
    """Extract file path from patch."""
    for line in patch_content.split("\n"):
        if line.startswith("--- "):
            return line[4:].split("\t")[0].strip()
    return None


def call_llm(prompt: str) -> str:
    """Call LLM to fix patch."""
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": FIX_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4000,
    }
    try:
        resp = requests.post(url, json=payload, headers=HEADERS, timeout=120)
        if resp.status_code == 200:
            content = (
                resp.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            lines = content.strip().split("\n")
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines)
        else:
            logger.error(f"LLM Error {resp.status_code}")
            return ""
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return ""


def fix_patch(patch_path: Path) -> dict:
    """Fix a single malformed patch."""
    try:
        with open(patch_path) as f:
            patch_content = f.read()

        # Check if it has placeholder headers
        if "@@ ... @@" not in patch_content:
            # Already has proper headers, just copy
            fixed_path = FIXED_DIR / patch_path.name
            with open(fixed_path, "w") as f:
                f.write(patch_content)
            return {"file": patch_path.name, "status": "already_valid"}

        # Get original file to provide context
        file_path = get_file_path_from_patch(patch_content)
        if not file_path:
            return {"file": patch_path.name, "status": "no_file_path"}

        original_path = PROJECT_ROOT / file_path
        if not original_path.exists():
            return {"file": patch_path.name, "status": "original_missing"}

        with open(original_path) as f:
            original_content = f.read()

        prompt = FIX_PATCH_PROMPT.format(
            original_content=original_content, patch_content=patch_content
        )

        fixed = call_llm(prompt)
        if not fixed:
            return {"file": patch_path.name, "status": "llm_failed"}

        # Save fixed patch
        fixed_path = FIXED_DIR / patch_path.name
        with open(fixed_path, "w") as f:
            f.write(fixed)

        return {"file": patch_path.name, "status": "fixed", "path": str(fixed_path)}

    except Exception as e:
        return {"file": patch_path.name, "status": "error", "error": str(e)}


def main():
    patches = sorted(PATCHES_DIR.glob("*.diff"))
    logger.info(f"Found {len(patches)} patches to fix")
    logger.info(f"Using model: {FIX_MODEL}")

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fix_patch, p): p for p in patches}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = result["status"]
            if status == "fixed":
                logger.info(f"✅ Fixed {result['file']}")
            elif status == "already_valid":
                logger.info(f"✓ Already valid: {result['file']}")
            else:
                logger.warning(f"⚠️ {result['file']}: {status}")

    fixed = sum(1 for r in results if r["status"] in ("fixed", "already_valid"))
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Fixed patches saved to: {FIXED_DIR}")
    logger.info(f"Result: {fixed}/{len(patches)} patches ready")


if __name__ == "__main__":
    main()
