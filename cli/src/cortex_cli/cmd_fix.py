from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from cortex.security.validators import is_dangerous_symlink

logger = logging.getLogger(__name__)

console = Console()
PATCHES_DIR = Path("patches")

# -----------------------------------------------------------------------------
# Symlink Fixer (from HEAD)
# -----------------------------------------------------------------------------


def _scan_for_insecure_symlinks(roots: List[Path]) -> int:
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


# -----------------------------------------------------------------------------
# Issue Fixer (from Origin/Main)
# -----------------------------------------------------------------------------


def load_report(path: str = "bulk_review_report_v2.json") -> list[dict]:
    import json

    if not os.path.exists(path):
        print(f"Report file not found: {path}")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    return data.get("issues", [])


def get_code_context(file_path: str, line: int, context_lines: int = 5):
    if not os.path.exists(file_path):
        return None, 0, 0
    with open(file_path) as f:
        lines = f.readlines()

    start_line = max(0, line - context_lines - 1)
    end_line = min(len(lines), line + context_lines)

    return "".join(lines[start_line:end_line]), start_line + 1, end_line


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

    # Call LLM (Pseudo-code as explicit call logic missing in snippet, recreating standard request)
    import requests

    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        response = requests.post(
            f"{base_url}/chat/completions", headers=headers, json=payload, timeout=30
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        # Save patch
        patch_file = PATCHES_DIR / f"fix_{idx:03d}.patch"
        with open(patch_file, "w") as f:
            f.write(content)

        result["status"] = "success"
        result["patch"] = str(patch_file)
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)

    return result


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
    logger.info(f"Output: {PATCHES_DIR}")

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


def setup_fix_parser(parser: argparse.ArgumentParser) -> None:
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
