import argparse
import asyncio
import glob
import json
import logging
import os
import re
import sys
from pathlib import Path

import aiohttp
import aiofiles

# Default Configuration
DEFAULT_REPO_OWNER = "g2gonee2025-cloud"
DEFAULT_REPO_NAME = "emailops"
DEFAULT_API_URL = "https://jules.googleapis.com/v1alpha/sessions"
DEFAULT_CONCURRENCY_LIMIT = 50
DEFAULT_MIN_LOC = 50

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Bulk code review using Jules API.")
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("JULES_API_KEY"),
        help="Jules API key. Can also be set via JULES_API_KEY env var.",
    )
    parser.add_argument(
        "--owner",
        type=str,
        default=DEFAULT_REPO_OWNER,
        help=f"Repository owner. Default: {DEFAULT_REPO_OWNER}",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=DEFAULT_REPO_NAME,
        help=f"Repository name. Default: {DEFAULT_REPO_NAME}",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"Jules API URL. Default: {DEFAULT_API_URL}",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY_LIMIT,
        help=f"Concurrency limit for API calls. Default: {DEFAULT_CONCURRENCY_LIMIT}",
    )
    parser.add_argument(
        "--min-loc",
        type=int,
        default=DEFAULT_MIN_LOC,
        help=f"Minimum lines of code for a file to be eligible. Default: {DEFAULT_MIN_LOC}",
    )
    return parser.parse_args()


def count_lines(filepath: Path) -> int:
    """Counts lines of code in a file, with specific error handling."""
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except OSError as e:
        logger.warning(f"Could not read file {filepath}: {e}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while counting lines for {filepath}: {e}",
            exc_info=True,
        )
    return 0


def _find_python_imports(content: str, filepath: Path) -> set[Path]:
    """Finds Python imports in the given content."""
    imports = set()
    matches = re.findall(r"^(?:from|import)\s+([\w\.]+)", content, re.MULTILINE)
    for match in matches:
        relative_path = match.replace(".", "/") + ".py"
        possible_path = Path.cwd() / "backend/src" / relative_path
        if not possible_path.exists():
            possible_path = Path.cwd() / "cli/src" / relative_path
        if possible_path.exists() and possible_path != filepath:
            imports.add(possible_path)
    return imports


def _find_js_imports(content: str, filepath: Path) -> set[Path]:
    """Finds JS/TS imports in the given content."""
    imports = set()
    matches = re.findall(r'from\s+[\'"]([^\'"]+)[\'"]', content)
    for match in matches:
        if match.startswith("."):
            possible_path = (filepath.parent / match).resolve()
            for ext in [".ts", ".tsx", ".js", ".jsx", ".json"]:
                p = possible_path.with_suffix(ext)
                if p.exists() and p != filepath:
                    imports.add(p)
                    break
    return imports


def find_imports(filepath: Path) -> set[Path]:
    """
    Very basic import detection for context resolution.
    Returns a set of absolute paths to imported files if they exist in the project.
    """
    imports = set()
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        if filepath.suffix == ".py":
            imports.update(_find_python_imports(content, filepath))
        elif filepath.suffix in [".ts", ".tsx", ".js", ".jsx"]:
            imports.update(_find_js_imports(content, filepath))
    except OSError as e:
        logger.warning(f"Could not read file {filepath} for import parsing: {e}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while parsing imports for {filepath}: {e}",
            exc_info=True,
        )
    return imports


async def _execute_api_call(
    session: aiohttp.ClientSession,
    api_url: str,
    api_key: str,
    payload: dict,
    relative_path: str,
):
    """Executes the API call to create a session."""
    try:
        async with session.post(
            api_url,
            json=payload,
            headers={"X-Goog-Api-Key": api_key, "Content-Type": "application/json"},
            timeout=300,
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                logger.info(f"Created session for {relative_path}: {data.get('name')}")
                return {
                    "file": str(relative_path),
                    "session_id": data.get("name"),
                    "status": "success",
                }
            logger.error(
                f"Failed to create session for {relative_path}: {resp.status} - {resp.reason}"
            )
            return {
                "file": str(relative_path),
                "error": f"{resp.status} - {resp.reason}",
                "status": "failed",
            }
    except aiohttp.ClientError as e:
        logger.error(f"API request failed for {relative_path}: {e}")
        return {"file": str(relative_path), "error": str(e), "status": "api_error"}
    except TimeoutError:
        logger.error(f"API request timed out for {relative_path}")
        return {"file": str(relative_path), "error": "Timeout", "status": "timeout"}
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during session creation for {relative_path}: {e}",
            exc_info=True,
        )
        return {
            "file": str(relative_path),
            "error": "Unexpected error",
            "status": "error",
        }


async def create_session(
    session: aiohttp.ClientSession,
    file_path: Path,
    context_files: set[Path],
    sem: asyncio.Semaphore,
    api_url: str,
    api_key: str,
    source_id: str,
):
    """Prepares and initiates a code review session."""
    async with sem:
        relative_path = file_path.relative_to(Path.cwd())
        context_str = "\n".join([str(p.relative_to(Path.cwd())) for p in context_files])
        prompt_text = (
            f"Analyze the file '{relative_path}' for errors, mismatches, logic errors, and syntactical errors.\n"
            f"Fix any errors found, considering edge cases and unintended negative consequences.\n"
            f"Check if it is correctly integrated with the 'frontend_cli' (likely cortex_cli).\n"
            f"Context files:\n{context_str}"
        )
        payload = {
            "title": f"Review: {relative_path}",
            "prompt": prompt_text,
            "sourceContext": {
                "source": source_id,
                "githubRepoContext": {"startingBranch": "main"},
            },
            "automationMode": "AUTO_CREATE_PR",
        }

        return await _execute_api_call(
            session, api_url, api_key, payload, str(relative_path)
        )


async def main():
    args = parse_args()

    if not args.api_key:
        logger.error(
            "JULES_API_KEY environment variable or --api-key flag must be set."
        )
        sys.exit(1)

    source_id = f"sources/github/{args.owner}/{args.repo}"
    root_dir = Path.cwd()
    ignore_dirs = {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "temp_validation_s3_20",
        "__pycache__",
    }

    tasks = []
    sem = asyncio.Semaphore(args.concurrency)

    logger.info(f"Scanning files in {root_dir}...")

    eligible_files = []

    # 1. Discovery
    for ext in ["*.py", "*.ts", "*.tsx", "*.js", "*.sh"]:
        for path in root_dir.rglob(ext):
            if any(part in ignore_dirs for part in path.parts):
                continue

            if path.is_file():
                loc = await asyncio.to_thread(count_lines, path)
                if loc > args.min_loc:
                    eligible_files.append(path)

    logger.info(f"Found {len(eligible_files)} files > {args.min_loc} LOC.")

    # 2. Execution
    async with aiohttp.ClientSession() as session:
        for f in eligible_files:
            context = await asyncio.to_thread(find_imports, f)
            tasks.append(
                create_session(
                    session, f, context, sem, args.api_url, args.api_key, source_id
                )
            )

        results = await asyncio.gather(*tasks)

    # 3. Report
    successful_reviews = [r for r in results if r and r.get("status") == "success"]
    failed_reviews = [r for r in results if r and r.get("status") != "success"]

    logger.info("-" * 50)
    logger.info("Bulk Review Summary")
    logger.info("-" * 50)
    logger.info(f"Successfully created {len(successful_reviews)} review sessions.")
    for res in successful_reviews:
        logger.info(f"  [SUCCESS] {res['file']} -> Session ID: {res['session_id']}")

    if failed_reviews:
        logger.warning(f"\nFailed to create {len(failed_reviews)} review sessions.")
        for res in failed_reviews:
            logger.warning(f"  [FAILURE] {res['file']} -> Error: {res['error']}")

    # Save detailed JSON report
    async with aiofiles.open("jules_batch_report.json", "w") as f:
        await f.write(json.dumps(results, indent=2))

    logger.info("\nDetailed report saved to jules_batch_report.json")
    logger.info("Processing complete.")


if __name__ == "__main__":
    asyncio.run(main())
