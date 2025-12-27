import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

import aiohttp

# Configuration
API_KEY = os.environ.get("JULES_API_KEY")
REPO_OWNER = "g2gonee2025-cloud"
REPO_NAME = "emailops"
SOURCE_ID = f"sources/github/{REPO_OWNER}/{REPO_NAME}"
API_URL = "https://jules.googleapis.com/v1alpha/sessions"
CONCURRENCY_LIMIT = 95
MIN_LOC = 50

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if not API_KEY:
    logger.error("JULES_API_KEY environment variable not set.")
    sys.exit(1)


async def count_lines(filepath: Path) -> int:
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception as e:
        logger.warning(f"Could not count lines for {filepath}: {e}")
        return 0


def find_imports(filepath: Path) -> set[Path]:
    """
    Very basic import detection for context resolution.
    Returns a set of absolute paths to imported files if they exist in the project.
    """
    imports = set()
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")

        # Simple regex for Python imports
        if filepath.suffix == ".py":
            # import x.y.z
            # from x.y import z
            matches = re.findall(r"^(?:from|import)\s+([\w\.]+)", content, re.MULTILINE)
            for match in matches:
                # rough conversion: foo.bar -> foo/bar.py
                relative_path = match.replace(".", "/") + ".py"

                # Check varying depths basically
                # This is heuristic and simple as requested
                possible_path = Path.cwd() / "backend/src" / relative_path
                if not possible_path.exists():
                    possible_path = Path.cwd() / "cli/src" / relative_path

                if possible_path.exists() and possible_path != filepath:
                    imports.add(possible_path)

        # Simple regex for JS/TS
        elif filepath.suffix in [".ts", ".tsx", ".js", ".jsx"]:
            # import ... from './foo'
            matches = re.findall(r'from\s+[\'"]([^\'"]+)[\'"]', content)
            for match in matches:
                if match.startswith("."):
                    possible_path = (filepath.parent / match).resolve()
                    # Try extensions
                    for ext in [".ts", ".tsx", ".js", ".jsx", ".json"]:
                        p = possible_path.with_suffix(ext)
                        if p.exists() and p != filepath:
                            imports.add(p)
                            break

    except Exception as e:
        logger.warning(f"Error parsing imports for {filepath}: {e}")

    return imports


async def create_session(
    session: aiohttp.ClientSession,
    file_path: Path,
    context_files: set[Path],
    sem: asyncio.Semaphore,
):
    async with sem:
        relative_path = file_path.relative_to(Path.cwd())

        # Construct Prompt
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
                "source": SOURCE_ID,
                "githubRepoContext": {
                    "startingBranch": "main"  # Assuming main, can be adjusted
                },
            },
            "automationMode": "AUTO_CREATE_PR",  # Optional, purely for action
        }

        try:
            async with session.post(
                API_URL,
                json=payload,
                headers={"X-Goog-Api-Key": API_KEY, "Content-Type": "application/json"},
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(
                        f"Created session for {relative_path}: {data.get('name')}"
                    )
                    return {
                        "file": str(relative_path),
                        "session_id": data.get("name"),
                        "status": "success",
                    }
                else:
                    text = await resp.text()
                    logger.error(
                        f"Failed to create session for {relative_path}: {resp.status} - {text}"
                    )
                    return {
                        "file": str(relative_path),
                        "error": text,
                        "status": "failed",
                    }
        except Exception as e:
            logger.error(f"Exception for {relative_path}: {e}")
            return {"file": str(relative_path), "error": str(e), "status": "error"}


async def main():
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
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    logger.info(f"Scanning files in {root_dir}...")

    eligible_files = []

    # 1. Discovery
    for ext in ["*.py", "*.ts", "*.tsx", "*.js", "*.sh"]:
        for path in root_dir.rglob(ext):
            # Skip ignored dirs
            if any(part in ignore_dirs for part in path.parts):
                continue

            if path.is_file():
                loc = await count_lines(path)
                if loc > MIN_LOC:
                    eligible_files.append(path)

    logger.info(f"Found {len(eligible_files)} files > {MIN_LOC} LOC.")

    # 2. Execution
    async with aiohttp.ClientSession() as session:
        for f in eligible_files:
            context = find_imports(f)
            tasks.append(create_session(session, f, context, sem))

        results = await asyncio.gather(*tasks)

    # 3. Report
    with open("jules_batch_report.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Batch processing complete. Report saved to jules_batch_report.json")


if __name__ == "__main__":
    asyncio.run(main())
