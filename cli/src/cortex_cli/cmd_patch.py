from __future__ import annotations

import argparse
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from dotenv import load_dotenv

if TYPE_CHECKING:
    from collections.abc import Sequence

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants ---

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

# --- Parser Setup ---


def setup_patch_parser(subparsers: Any) -> None:
    """Setup the 'patch' command parser."""
    parser = subparsers.add_parser(
        "patch",
        help="Fix malformed patch files using an LLM",
        description="Fixes malformed unified diff patches by regenerating hunk headers.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the malformed patch files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to save the fixed patch files.",
    )
    parser.add_argument(
        "--model",
        default="openai-gpt-oss-120b",
        help="The LLM to use for fixing the patches.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="The maximum number of concurrent workers.",
    )
    parser.set_defaults(func=main)


# --- Core Logic ---


def _is_valid_diff(content: str) -> bool:
    """Perform a basic validation of the unified diff format."""
    return (
        "--- " in content
        and "+++ " in content
        and re.search(r"^@@ -\d+,\d+ \+\d+,\d+ @@", content, re.MULTILINE) is not None
    )


def get_file_path_from_patch(patch_content: str) -> str | None:
    """Extract file path from patch, handling paths with spaces."""
    for line in patch_content.split("\n"):
        if line.startswith("--- "):
            path_part = line[4:]
            return path_part.split("\t")[0].strip() if "\t" in path_part else path_part.strip()
    return None


def call_llm(prompt: str, model: str) -> str:
    """Call LLM to fix patch."""
    api_key = os.getenv("LLM_API_KEY") or os.getenv("DO_API_KEY")
    base_url = (os.getenv("LLM_ENDPOINT") or "https://inference.do-ai.run/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{base_url}/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4000,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()

        data = resp.json()
        if (
            not isinstance(data, dict)
            or "choices" not in data
            or not isinstance(data["choices"], list)
            or not data["choices"]
        ):
            logger.error(f"LLM response has unexpected structure: {data}")
            return ""

        message = data["choices"][0].get("message", {})
        content = message.get("content", "")

        # Strip markdown code blocks if present
        lines = content.strip().split("\n")
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        return "\n".join(lines)

    except requests.exceptions.RequestException as e:
        logger.error(f"LLM request failed: {e}")
        return ""
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Failed to parse LLM response: {e}")
        return ""


def fix_patch(
    patch_path: Path, output_dir: Path, project_root: Path, model: str
) -> dict:
    """Fix a single malformed patch."""
    try:
        patch_content = patch_path.read_text()

        if "@@ ... @@" not in patch_content:
            fixed_path = output_dir / patch_path.name
            fixed_path.write_text(patch_content)
            return {"file": patch_path.name, "status": "already_valid"}

        file_path = get_file_path_from_patch(patch_content)
        if not file_path:
            return {"file": patch_path.name, "status": "no_file_path"}

        original_path = project_root / file_path
        if not original_path.exists():
            return {"file": patch_path.name, "status": "original_missing"}

        original_content = original_path.read_text()
        prompt = FIX_PATCH_PROMPT.format(
            original_content=original_content, patch_content=patch_content
        )

        fixed_content = call_llm(prompt, model)
        if not fixed_content:
            return {"file": patch_path.name, "status": "llm_failed"}

        if not _is_valid_diff(fixed_content):
            return {"file": patch_path.name, "status": "invalid_output"}

        fixed_path = output_dir / patch_path.name
        fixed_path.write_text(fixed_content)

        return {"file": patch_path.name, "status": "fixed", "path": str(fixed_path)}

    except Exception as e:
        logger.exception(f"Error processing {patch_path.name}")
        return {"file": patch_path.name, "status": "error", "error": str(e)}


def main(args: argparse.Namespace) -> None:
    """Main function for the 'patch' command."""
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    project_root = Path.cwd()

    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return

    output_dir.mkdir(exist_ok=True)
    patches = sorted(input_dir.glob("*.diff"))

    logger.info(f"Found {len(patches)} patches to fix in {input_dir}")
    logger.info(f"Using model: {args.model}")
    logger.info(f"Saving fixed patches to: {output_dir}")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                fix_patch, p, output_dir, project_root, args.model
            ): p
            for p in patches
        }

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

    fixed_count = sum(
        1 for r in results if r["status"] in ("fixed", "already_valid")
    )
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Result: {fixed_count}/{len(patches)} patches are ready in {output_dir}")
