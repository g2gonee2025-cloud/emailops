#!/usr/bin/env python3
"""
Bulk Code Fixer Script.

Takes `bulk_review_report_grouped.json` and generates unified diff patches
to fix issues. Uses a single worker per file (concurrently).
Model: gpt-5
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("bulk_fixer")

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PATCHES_DIR = PROJECT_ROOT / "patches"
PATCHES_DIR.mkdir(parents=True, exist_ok=True)

# API Configuration
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DO_API_KEY")
BASE_URL = (os.getenv("LLM_ENDPOINT") or "https://inference.do-ai.run/v1").rstrip("/")

if not API_KEY:
    print("Missing API key. Set LLM_API_KEY or DO_API_KEY.")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Concurrency
MAX_WORKERS = 10
FIX_MODEL = "openai-gpt-5"

# Prompt - Request FIXED FILE instead of diff
FIX_PROMPT = """
<task>
You are an expert software engineer tasked with fixing code issues.
You are given a Python file and a list of issues identified by code review.

OUTPUT THE COMPLETE FIXED FILE CONTENT with all issues resolved.
Do NOT output diffs, markdown, or explanations.
Output ONLY the fixed Python code, starting from the first line.
</task>

<context>
File Path: {file_path}

Issues to Fix:
{issues_list}
</context>

<original_file>
{file_content}
</original_file>

<instructions>
1. Fix ALL issues listed above
2. Maintain all existing functionality
3. Preserve code style and formatting
4. Output ONLY the complete fixed Python file
5. Do NOT add markdown code blocks or explanations
6. Start output with the first line of code (imports, shebang, or docstring)
</instructions>
"""


def load_grouped_report() -> dict:
    path = PROJECT_ROOT / "bulk_review_report_grouped.json"
    if not path.exists():
        logger.error(f"Report not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def call_llm_for_fixed_content(model: str, prompt: str) -> str:
    """Make LLM API call to get fixed file content."""
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 8000,  # Increased for full file content
    }
    try:
        resp = requests.post(url, json=payload, headers=HEADERS, timeout=240)
        if resp.status_code == 200:
            content = (
                resp.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            # Strip markdown code blocks if present
            lines = content.strip().split("\n")
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines)
        else:
            logger.error(f"LLM Error {resp.status_code}: {resp.text[:200]}")
            return ""
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return ""


def process_file(file_path_str: str, issues: list[dict]) -> dict:
    """Generate a fix patch for a single file using difflib."""
    import difflib

    full_path = PROJECT_ROOT / file_path_str
    if not full_path.exists():
        return {"file": file_path_str, "status": "missing"}

    try:
        with full_path.open(encoding="utf-8") as f:
            original_content = f.read()

        if not original_content.strip():
            return {"file": file_path_str, "status": "skipped_empty"}

        # Format issues string
        issues_text = []
        for i in issues:
            line = f"Line {i.get('line')}" if i.get("line") else "General"
            issues_text.append(
                f"- [{i.get('category')}] {line}: {i.get('description')}"
            )

        prompt = FIX_PROMPT.format(
            file_path=file_path_str,
            issues_list="\n".join(issues_text),
            file_content=original_content,
        )

        fixed_content = call_llm_for_fixed_content(FIX_MODEL, prompt)

        if not fixed_content or fixed_content == original_content:
            return {"file": file_path_str, "status": "no_changes"}

        # Generate proper unified diff using difflib
        original_lines = original_content.splitlines(keepends=True)
        fixed_lines = fixed_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            fixed_lines,
            fromfile=file_path_str,
            tofile=file_path_str,
            lineterm="",
        )

        diff_text = "\n".join(diff)

        if not diff_text:
            return {"file": file_path_str, "status": "no_changes"}

        # Save patch
        patch_name = file_path_str.replace("/", "__") + ".diff"
        patch_path = PATCHES_DIR / patch_name
        with open(patch_path, "w") as f:
            f.write(diff_text)

        return {"file": file_path_str, "status": "success", "patch": str(patch_path)}

    except Exception as e:
        logger.error(f"Error processing {file_path_str}: {e}")
        return {"file": file_path_str, "status": "error", "error": str(e)}


def main():
    logger.info("Loading issues report...")
    report = load_grouped_report()
    files_map = report.get("files", {})

    # Only process files that have issues
    files_to_process = [f for f, issues in files_map.items() if issues]

    logger.info(f"Found {len(files_to_process)} files with issues to fix.")
    logger.info(f"Starting repair with model: {FIX_MODEL} (Workers: {MAX_WORKERS})")

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(process_file, f, files_map[f]): f for f in files_to_process
        }

        for future in as_completed(future_to_file):
            f = future_to_file[future]
            try:
                res = future.result()
                results.append(res)
                status = res.get("status")
                if status == "success":
                    logger.info(f"✅ Generated patch for {f}")
                else:
                    logger.warning(f"⚠️  Failed {f}: {status}")
            except Exception as e:
                logger.error(f"❌ Exception for {f}: {e}")

    logger.info("=" * 60)
    logger.info(f"Patches generated in: {PATCHES_DIR}")
    logger.info(f"Total processed: {len(results)}")

    success_count = sum(1 for r in results if r.get("status") == "success")
    logger.info(f"Success: {success_count}/{len(results)}")


if __name__ == "__main__":
    main()
