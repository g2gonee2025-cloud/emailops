#!/usr/bin/env python3
"""
Per-Issue Patch Generator.

Generates one micro-patch per individual error from bulk_review_report_v2.json.
Each patch fixes exactly ONE issue, making them simpler and more reliable.
Uses the same patterns as bulk_code_review.py.
"""
import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv(".env")

# Configure logging (same as bulk_code_review.py)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("per_issue_fixer")

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
PATCHES_DIR = PROJECT_ROOT / "patches_per_issue"


def setup_fix_parser(subparsers: Any):
    """Setup the 'fix-issues' command parser."""
    fix_parser = subparsers.add_parser(
        "fix-issues",
        help="Generate patches for issues in bulk_review_report_v2.json",
        description="""
Generates one micro-patch per individual error from bulk_review_report_v2.json.
Each patch fixes exactly ONE issue, making them simpler and more reliable.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    fix_parser.add_argument(
        "--model",
        default=os.getenv("FIXER_MODEL", "gpt-4"),
        help="The model to use for generating patches (default: gpt-4, or FIXER_MODEL env var)",
    )
    fix_parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of concurrent workers (default: 10)",
    )
    fix_parser.set_defaults(
        func=lambda args: run_fixer(model=args.model, max_workers=args.max_workers)
    )


def run_fixer(model: str, max_workers: int):
    """
    Main function to generate patches for issues.
    """
    PATCHES_DIR.mkdir(exist_ok=True)

    # API Configuration (same as bulk_code_review.py)
    API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DO_API_KEY")
    BASE_URL = (os.getenv("LLM_ENDPOINT") or "https://inference.do-ai.run/v1").rstrip(
        "/"
    )

    if not API_KEY:
        print("Missing API key. Set LLM_API_KEY or DO_API_KEY.")
        sys.exit(1)

    HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    logger.info("Loading issues from bulk_review_report_v2.json...")
    issues = load_report()
    logger.info(f"Found {len(issues)} issues to process")
    logger.info(f"Model: {model}")
    logger.info(f"Workers: {max_workers}")
    logger.info(f"Output: {PATCHES_DIR}\n")

    results = []
    success_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_issue, issue, i, model, BASE_URL, HEADERS): i
            for i, issue in enumerate(issues)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append(result)

                if result["status"] == "success":
                    success_count += 1
                    if success_count % 50 == 0:
                        logger.info(f"Progress: {success_count} patches generated...")
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                logger.error(f"Exception for issue {idx}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("PER-ISSUE PATCH GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total Issues: {len(issues)}")
    print(f"Patches Generated: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Output Directory: {PATCHES_DIR}")
    print("=" * 60)


# Prompt for fixing a single issue
FIX_PROMPT = """
<task>
You are an expert Python developer. Fix this ONE specific issue.
Output ONLY a valid unified diff patch. No explanations.
</task>

<file>{file_path}</file>
<issue>[{category}] Line {line}: {description}</issue>

<code_context>
{code_context}
</code_context>

<output_format>
--- {file_path}
+++ {file_path}
@@ -{hunk_start},{hunk_old} @@
 context line (space prefix)
-removed line (minus prefix)
+added line (plus prefix)
 context line (space prefix)
</output_format>

Output the unified diff now:
"""


def load_report() -> list[dict]:
    """Load issues from bulk_review_report_v2.json."""
    path = PROJECT_ROOT / "bulk_review_report_v2.json"
    if not path.exists():
        logger.error(f"Report not found: {path}")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    return data.get("issues", [])


def get_code_context(
    file_path: str, line: int, context: int = 5
) -> tuple[str, int, int]:
    """Get code context around a specific line."""
    full_path = PROJECT_ROOT / file_path
    if not full_path.exists():
        return "", 0, 0

    try:
        with open(full_path, encoding="utf-8") as f:
            lines = f.readlines()
    except (IOError, OSError) as e:
        logger.error(f"Error reading file {full_path}: {e}")
        return "", 0, 0

    start = max(0, line - context - 1)
    end = min(len(lines), line + context)

    # Number each line for context
    numbered = []
    for i in range(start, end):
        numbered.append(f"{i + 1:4d}: {lines[i].rstrip()}")

    return "\n".join(numbered), start + 1, end


def call_llm(prompt: str, model: str, base_url: str, headers: Dict[str, str]) -> str:
    """Make a single LLM API call (same pattern as bulk_code_review.py)."""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1000,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            content = (
                data.get("choices", [{}])[0].get("message", {}).get("content") or ""
            )
            # Strip markdown code blocks if present
            lines = content.strip().split("\n")
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines)
        else:
            return ""
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return ""


def process_issue(
    issue: dict, idx: int, model: str, base_url: str, headers: Dict[str, str]
) -> dict:
    """Generate a patch for a single issue."""
    file_path = issue.get("file", "")
    line = issue.get("line") or 1
    category = issue.get("category", "UNKNOWN")
    description = issue.get("description", "")

    result = {"idx": idx, "file": file_path, "line": line, "status": "unknown"}

    if not file_path:
        result["status"] = "no_file"
        return result

    code_context, start_line, end_line = get_code_context(file_path, line)
    if not code_context:
        result["status"] = "file_missing"
        return result

    prompt = FIX_PROMPT.format(
        file_path=file_path,
        category=category,
        line=line,
        description=description,
        code_context=code_context,
        hunk_start=start_line,
        hunk_old=end_line - start_line + 1,
    )

    patch = call_llm(prompt, model, base_url, headers)
    if not patch:
        result["status"] = "llm_failed"
        return result

    # Save patch with unique name
    safe_file = file_path.replace("/", "__").replace(".py", "")
    patch_name = f"{safe_file}__L{line}__I{idx}.diff"
    patch_path = PATCHES_DIR / patch_name

    with open(patch_path, "w") as f:
        f.write(patch)

    result["status"] = "success"
    result["patch"] = str(patch_path)
    return result
