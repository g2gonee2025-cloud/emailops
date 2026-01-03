#!/usr/bin/env python3
"""
Fix SonarQube Issues using Jules API.
Reads sonarqube_issues.json and creates Jules sessions.
"""

import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# Load env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fix_sonar_issues.log"),
    ],
)
logger = logging.getLogger("fix_sonar")

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ISSUES_FILE = PROJECT_ROOT / "sonarqube_issues.json"
REPORT_FILE = PROJECT_ROOT / "jules_sonar_batch_report.json"

# API Configuration
JULES_API_KEY = os.getenv("JULES_API_KEY") or os.getenv("JULES_API_KEY_ALT")
API_URL = "https://jules.googleapis.com/v1alpha/sessions"
REPO_OWNER = "g2gonee2025-cloud"
REPO_NAME = "emailops"
SOURCE_ID = f"sources/github/{REPO_OWNER}/{REPO_NAME}"
STARTING_BRANCH = "main"
AUTOMATION_MODE = "AUTO_CREATE_PR"  # or "AUTOMATION_MODE_UNSPECIFIED"

# Limits
SESSION_LIMIT = 70  # User specified limit
CONCURRENCY = 5

if not JULES_API_KEY:
    logger.error("Missing JULES_API_KEY. Please set it in .env")
    sys.exit(1)


def load_issues():
    if not ISSUES_FILE.exists():
        logger.error(f"Issues file not found: {ISSUES_FILE}")
        sys.exit(1)

    with open(ISSUES_FILE) as f:
        data = json.load(f)

    issues = data.get("issues", [])
    logger.info(f"Loaded {len(issues)} issues from {ISSUES_FILE}")
    return issues


def group_issues_by_file(issues):
    grouped = defaultdict(list)
    for issue in issues:
        # SonarQube component format: "projectKey:path/to/file"
        # JSON report form SonarQube API usually has 'component' or 'file' depending on projection
        component = issue.get("file") or issue.get("component") or ""

        if ":" in component:
            file_path = component.split(":", 1)[1]
        else:
            file_path = component

        if not file_path:
            continue

        # Skip if file doesn't exist locally (might be deleted or excluded)
        if not (PROJECT_ROOT / file_path).exists():
            continue

        grouped[file_path].append(issue)

    return grouped


def load_previous_batches():
    if REPORT_FILE.exists():
        with open(REPORT_FILE) as f:
            return json.load(f)
    return {}


def save_batch_report(report):
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)


async def create_jules_session(session, file_path, issues):
    """Create a Jules session for a file."""

    # Construct Prompt
    issue_descriptions = []
    for i, issue in enumerate(issues, 1):
        line = issue.get("line", "General")
        msg = issue.get("message", "")
        rule = issue.get("rule", "")
        desc = f"{i}. Line {line}: [{rule}] {msg}"
        issue_descriptions.append(desc)

    issues_text = "\n".join(issue_descriptions)
    prompt = (
        f"Fix the following SonarQube issues in '{file_path}'.\n\n"
        f"Issues:\n{issues_text}\n\n"
        "Instructions:\n"
        "1. Analyze the provided issues and the file content.\n"
        "2. Apply necessary fixes to resolve the reported issues.\n"
        "3. Ensure the code remains functional and follows best practices.\n"
        "4. Do not introduce new issues.\n"
    )

    payload = {
        "title": f"Fix SonarQube Issues: {file_path}",
        "prompt": prompt,
        "sourceContext": {
            "source": SOURCE_ID,
            "githubRepoContext": {"startingBranch": STARTING_BRANCH},
        },
        "automationMode": AUTOMATION_MODE,
    }

    headers = {
        "X-Goog-Api-Key": JULES_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        async with session.post(API_URL, json=payload, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data
            else:
                text = await resp.text()
                logger.error(f"Failed {file_path}: {resp.status} - {text}")
                return None
    except Exception as e:
        logger.error(f"Exception for {file_path}: {e}")
        return None


async def process_batch(file_paths, grouped_issues, report):
    async with aiohttp.ClientSession() as session:
        created_count = 0
        tasks = []

        # Determine which files need processing
        pending_files = []
        for file_path in file_paths:
            if file_path in report:
                logger.info(
                    f"Skipping {file_path} (already processed: {report[file_path].get('name')})"
                )
                continue
            pending_files.append(file_path)

        # Respect Session Limit
        target_files = pending_files[:SESSION_LIMIT]
        if len(pending_files) > SESSION_LIMIT:
            logger.warning(
                f"Limiting to {SESSION_LIMIT} sessions (Quota limit). Pending: {len(pending_files)}"
            )

        logger.info(f"Creating {len(target_files)} sessions...")

        semaphore = asyncio.Semaphore(CONCURRENCY)

        async def _sem_task(fp, iss):
            async with semaphore:
                return await create_jules_session(session, fp, iss)

        for file_path in target_files:
            issues = grouped_issues[file_path]
            task = asyncio.create_task(_sem_task(file_path, issues))
            tasks.append((file_path, task))

        # Process in chunks to limit concurrency if needed, but aiohttp handles it well.
        # We'll just wait for all.
        for file_path, task in tasks:
            result = await task
            if result:
                name = result.get("name")
                logger.info(f"Created Session: {file_path} -> {name}")
                report[file_path] = {
                    "name": name,
                    "status": "created",
                    "issue_count": len(grouped_issues[file_path]),
                }
                created_count += 1
                save_batch_report(report)  # Incremental save
            else:
                logger.error(f"Failed to create session for {file_path}")

        return created_count


def main():
    issues = load_issues()
    grouped_issues = group_issues_by_file(issues)

    # Sort files by number of critical issues or just issue count?
    # Let's sort by severity/priority implicitly by list order if sorted in JSON?
    # Or just number of issues (descending) to tackle heaviest files first.
    sorted_files = sorted(
        grouped_issues.keys(), key=lambda k: len(grouped_issues[k]), reverse=True
    )

    report = load_previous_batches()

    filtered_files = []
    for f in sorted_files:
        if "kube-state-metrics" in f:
            continue  # Skip vendored
        if "node_modules" in f:
            continue
        filtered_files.append(f)

    logger.info(f"Files to process (excluding vendored): {len(filtered_files)}")

    processed = asyncio.run(process_batch(filtered_files, grouped_issues, report))

    logger.info(f"Done. Created {processed} sessions.")


if __name__ == "__main__":
    main()
