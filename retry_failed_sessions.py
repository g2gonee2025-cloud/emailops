import asyncio
import json
import logging
import os
import time
from typing import Any

import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("persistent_retry.log"), logging.StreamHandler()],
)

# Constants
CONCURRENCY_LIMIT = (
    1  # Strict Limit to avoid burst 429s -> User mentioned concurrency limit
)
REQUEST_DELAY = 1.0  # Delay between checks
RATE_LIMIT_SLEEP = 180  # 3 minutes sleep on 429
QUEUE_MAX_SIZE = 500  # Bounded queue to prevent memory spikes

# Payload Config
REPO_OWNER = "g2gonee2025-cloud"
REPO_NAME = "emailops"
SOURCE_ID = f"sources/github/{REPO_OWNER}/{REPO_NAME}"
API_URL = "https://jules.googleapis.com/v1alpha/sessions"

# API Key Loader
JULES_API_KEY = os.getenv("JULES_API_KEY_ALT")
if not JULES_API_KEY:
    try:
        with open(".env") as f:
            for line in f:
                if line.startswith("JULES_API_KEY_ALT="):
                    JULES_API_KEY = line.strip().split("=", 1)[1]
                    break
    except Exception:
        pass

if not JULES_API_KEY:
    JULES_API_KEY = os.getenv("JULES_API_KEY")


async def create_session(
    session: aiohttp.ClientSession, file_path: str
) -> dict[str, Any]:
    headers = {"X-Goog-Api-Key": JULES_API_KEY, "Content-Type": "application/json"}

    # Best-in-Class Prompt v2 (Context Aware & Utility Maximized)
    prompt_text = (
        f"Act as a Principal Software Architect for 'Cortex', an enterprise EmailOps platform.\n"
        f"Review the file '{file_path}' with extreme rigor.\n\n"
        "**System Context:**\n"
        "- **Stack:** Python 3.11 (FastAPI, Async), React (TypeScript, Tailwind), PostgreSQL (pgvector).\n"
        "- **Infra:** DigitalOcean (Kubernetes, Spaces), Redis (Queue/Cache).\n"
        "- **Core Domain:** High-volume email processing, RAG (Retrieval Augmented Generation), Security.\n\n"
        "**Review Objectives (Maximize Utility):**\n"
        "1.  **Security Hardening:** Detect Injection (SQL/Prompt), Auth bypass, and PII leakage. (Critical)\n"
        "2.  **Resilience:** Identify unhandled async exceptions, race conditions, or connection leaks.\n"
        "3.  **Performance:** Spot N+1 DB queries, expensive loops, or memory bloat.\n"
        "4.  **Architectural Fit:** Ensure separation of concerns (Service Layer vs API Layer).\n\n"
        "**Actionable Output:**\n"
        "- Only report **High-Value Findings** (Bugs, Security, Perf). Ignore trivial nitpicks.\n"
        "- **PROVIDE CODE FIXES:** For every issue, write the *corrected* code.\n"
    )

    payload = {
        "title": f"Deep Review: {file_path}",
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
                }
            elif response.status == 429:
                return {"file": file_path, "status": "rate_limited", "code": 429}
            else:
                error = await response.text()
                logging.error(f"FAIL {file_path}: {error}")
                return {"file": file_path, "status": "failed", "error": error}
    except Exception as e:
        logging.error(f"ERR {file_path}: {e}")
        return {"file": file_path, "status": "error", "error": str(e)}


def get_completed_files():
    completed = set()
    for fname in [
        "jules_batch_report.json",
        "jules_smart_retry_report.json",
        "jules_surgical_report.json",
    ]:
        if os.path.exists(fname):
            try:
                d = json.load(open(fname))
                if isinstance(d, list):
                    completed.update(
                        {x["file"] for x in d if x.get("status") == "success"}
                    )
                elif isinstance(d, dict):
                    completed.update(
                        {
                            x["file"]
                            for x in d.get("results", [])
                            if x.get("status") == "success"
                        }
                    )
            except Exception:
                pass
    return completed


def get_priority_score(path):
    # Tier 1: Security, Auth, Critical API Routes
    if "security" in path or "auth" in path or "routes_" in path:
        return 5
    # Tier 2: Core Intelligence/Retrieval/Ingestion
    if (
        "intelligence" in path
        or "retrieval" in path
        or "ingestion" in path
        or "rag_api" in path
    ):
        return 4
    # Tier 3: Core Backend (Models, DB, General)
    if "backend/src/cortex/" in path and "test" not in path:
        return 3
    # Tier 4: Frontend Components (UI)
    if "frontend/src/components/" in path or "frontend/src/views/" in path:
        return 2
    # Tier 5: Everything else non-test
    if "test" not in path and "migration" not in path and "scripts/" not in path:
        return 1
    # Tier 0: Ignore/Last
    return 0


async def worker(queue, session, rate_limit_event, rate_limit_lock):
    """Worker that processes items from queue with atomic rate-limit handling."""
    while not queue.empty():
        item = await queue.get()
        file_path = item["file"]

        while True:
            await rate_limit_event.wait()

            # Simple per-worker delay
            await asyncio.sleep(REQUEST_DELAY)

            res = await create_session(session, file_path)

            if res.get("status") == "rate_limited":
                logging.warning(
                    f"429 Hit on {file_path}. Sleeping {RATE_LIMIT_SLEEP}s..."
                )

                # ATOMIC: Use lock to ensure only one worker enters critical section
                # This prevents multiple concurrent sleep periods (race condition)
                async with rate_limit_lock:
                    if rate_limit_event.is_set():
                        rate_limit_event.clear()
                        await asyncio.sleep(RATE_LIMIT_SLEEP)
                        rate_limit_event.set()
                        logging.info("Resuming...")
                continue  # Retry same file

            # Success or hard fail
            queue.task_done()
            break


async def safe_worker(queue, session, rate_limit_event, rate_limit_lock, worker_id):
    """Wrapper for worker with error recovery and logging."""
    try:
        await worker(queue, session, rate_limit_event, rate_limit_lock)
    except Exception as e:
        logging.error(f"Worker {worker_id} crashed with exception: {e}", exc_info=True)
        raise


async def main():
    # 1. Load initial failures
    if not os.path.exists("jules_batch_report.json"):
        print("No jules_batch_report.json found.")
        return

    with open("jules_batch_report.json") as f:
        data = json.load(f)
        all_items = data if isinstance(data, list) else data.get("results", [])

    # 2. Filter completed
    completed = get_completed_files()
    pending = [x for x in all_items if x["file"] not in completed]

    # 3. Filter out Tier 0 items (tests, migrations, scripts)
    pending = [x for x in pending if get_priority_score(x["file"]) > 0]

    # 4. Sort by priority
    pending.sort(key=lambda x: get_priority_score(x["file"]), reverse=True)

    print(f"Total Pending: {len(pending)}")
    print(f"Already Completed: {len(completed)}")
    print("Starting Persistent Background Worker...")

    # Create bounded queue to prevent memory spikes
    queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
    
    # Enqueue all pending items (respects maxsize)
    for p in pending:
        await queue.put(p)

    rate_limit_event = asyncio.Event()
    rate_limit_event.set()
    rate_limit_lock = asyncio.Lock()  # Protect critical section

    async with aiohttp.ClientSession() as session:
        workers = [
            asyncio.create_task(
                safe_worker(queue, session, rate_limit_event, rate_limit_lock, i)
            )
            for i in range(CONCURRENCY_LIMIT)
        ]
        
        # Gather with exception handling
        results = await asyncio.gather(*workers, return_exceptions=True)
        
        # Check for worker failures
        failed = [r for r in results if isinstance(r, Exception)]
        if failed:
            logging.error(f"Worker failures detected: {len(failed)}")
            for exc in failed:
                logging.error(f"  - {exc}")
            raise failed[0]


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Main execution failed: {e}", exc_info=True)
        raise
