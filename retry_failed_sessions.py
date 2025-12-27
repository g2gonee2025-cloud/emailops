import asyncio
import aiohttp
import json
import os
import time
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("smart_retry.log"), logging.StreamHandler()],
)

# Constants based on Google AI Pro Tier for Jules
CONCURRENCY_LIMIT = 5
REQUEST_DELAY = 1.0
STOP_ON_RATE_LIMIT = False  # Changed to False for persistent mode

# Payload Config
REPO_OWNER = "g2gonee2025-cloud"
REPO_NAME = "emailops"
SOURCE_ID = f"sources/github/{REPO_OWNER}/{REPO_NAME}"
API_URL = "https://jules.googleapis.com/v1alpha/sessions"

JULES_API_KEY = os.getenv("JULES_API_KEY")

if not JULES_API_KEY:
    logging.error("JULES_API_KEY not found in environment variables.")
    try:
        with open(".env") as f:
            for line in f:
                if line.startswith("JULES_API_KEY="):
                    JULES_API_KEY = line.strip().split("=", 1)[1]
                    logging.info("Loaded JULES_API_KEY from .env")
                    break
    except Exception as e:
        logging.warning(f"Could not load .env file: {e}")


async def create_session(
    session: aiohttp.ClientSession, file_path: str, context_files: List[str] = None
) -> Dict[str, Any]:
    headers = {"X-Goog-Api-Key": JULES_API_KEY, "Content-Type": "application/json"}

    prompt_text = (
        f"Analyze the file '{file_path}' for errors, mismatches, logic errors, and syntactical errors.\n"
        f"Fix any errors found, considering edge cases and unintended negative consequences.\n"
        f"Check if it is correctly integrated with the 'frontend_cli' (likely cortex_cli)."
    )

    payload = {
        "title": f"Review: {file_path}",
        "prompt": prompt_text,
        "sourceContext": {
            "source": SOURCE_ID,
            "githubRepoContext": {"startingBranch": "main"},
        },
        "automationMode": "AUTO_CREATE_PR",
    }

    try:
        async with session.post(API_URL, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                logging.info(f"SUCCESS: {file_path}")
                return {
                    "file": file_path,
                    "status": "success",
                    "session_id": data.get("name"),
                    "response": data,
                }
            elif response.status == 429:
                error_text = await response.text()
                # logging.error(f"RATE LIMIT (429) for {file_path}") # Logged in worker now
                return {
                    "file": file_path,
                    "status": "rate_limited",
                    "error": error_text,
                    "code": 429,
                }
            else:
                error_text = await response.text()
                logging.error(f"FAILED {file_path} ({response.status}): {error_text}")
                return {
                    "file": file_path,
                    "status": "failed",
                    "error": error_text,
                    "code": response.status,
                }
    except Exception as e:
        logging.error(f"EXCEPTION for {file_path}: {e}")
        return {"file": file_path, "status": "error", "error": str(e)}


async def process_files(failed_items: List[Dict[str, Any]]):
    results = []

    # Use a Queue for better dynamic processing
    queue = asyncio.Queue()
    for item in failed_items:
        queue.put_nowait(item)

    concurrency_sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    rate_limit_event = asyncio.Event()
    rate_limit_event.set()  # Green light

    async with aiohttp.ClientSession() as session:

        async def worker():
            while not queue.empty():
                try:
                    item = await queue.get()
                    file_path = item["file"]

                    while True:  # Retry loop for this item
                        # Wait if we are in a "penalty box"
                        await rate_limit_event.wait()

                        async with concurrency_sem:
                            # Double check logic inside semaphore
                            if not rate_limit_event.is_set():
                                continue

                            await asyncio.sleep(REQUEST_DELAY)

                            result = await create_session(session, file_path)

                            if result.get("code") == 429:
                                logging.warning(
                                    f"429 Hit on {file_path}. Sleeping 3 minutes..."
                                )

                                # Only one worker needs to trigger the global pause
                                if rate_limit_event.is_set():
                                    rate_limit_event.clear()
                                    logging.info("Global Pause initiated.")
                                    await asyncio.sleep(180)
                                    rate_limit_event.set()
                                    logging.info("Resuming after rate limit pause.")

                                continue  # Retry this item

                            results.append(result)
                            queue.task_done()
                            break

                except Exception as e:
                    logging.error(f"Worker exception: {e}")
                    if queue:
                        queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(CONCURRENCY_LIMIT)]
        await asyncio.gather(*workers)

    return results


def get_priority_score(item):
    path = item["file"]
    if "backend/src/cortex/" in path and "test" not in path:
        return 3
    if "frontend/src/components/" in path or "frontend/src/views/" in path:
        return 2
    if "test" in path or "migration" in path or "scripts/" in path:
        return 0
    return 1


def main():
    report_file = "jules_batch_report.json"
    retry_report_file = "jules_persistent_report.json"

    if not os.path.exists(report_file):
        print(f"No report file found at {report_file}")
        return

    print(f"Reading failures from {report_file}...")
    try:
        with open(report_file, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                past_results = data.get("results", [])
            else:
                past_results = data
    except Exception as e:
        print(f"Error reading report: {e}")
        return

    # Identify failures
    failed_items = [r for r in past_results if r.get("status") != "success"]

    # Priority Logic
    failed_items.sort(key=get_priority_score, reverse=True)
    high_priority_items = [x for x in failed_items if get_priority_score(x) > 0]

    # Process ALL high priority items now, since we are persistent
    items_to_process = high_priority_items

    print(f"Found {len(failed_items)} total failures.")
    print(
        f"Starting PERSISTENT processing of {len(items_to_process)} high-value files."
    )
    print(f"Strategy: 5 Concurrent, Wait 3m on 429.")

    try:
        results = asyncio.run(process_files(items_to_process))
    except KeyboardInterrupt:
        print("\nStopped by user.")
        results = []  # Incomplete state handling would be better but simple for now

    successes = [r for r in results if r["status"] == "success"]

    print("-" * 40)
    print(f"Persistent Retry Complete.")
    print(f"Successes: {len(successes)}")
    print("-" * 40)

    final_output = {
        "timestamp": time.time(),
        "config": {"mode": "persistent"},
        "results": results,
    }

    with open(retry_report_file, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"Saved persistent report to {retry_report_file}")


if __name__ == "__main__":
    main()
