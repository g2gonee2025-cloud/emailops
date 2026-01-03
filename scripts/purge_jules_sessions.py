#!/usr/bin/env python3
"""
Purge all Jules sessions to free up quota.
"""

import asyncio
import logging
import os
import sys

import aiohttp
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("purge_sessions")

JULES_API_KEY = os.getenv("JULES_API_KEY") or os.getenv("JULES_API_KEY_ALT")
API_URL = "https://jules.googleapis.com/v1alpha/sessions"

if not JULES_API_KEY:
    logger.error("Missing JULES_API_KEY")
    sys.exit(1)


async def list_sessions(session):
    sessions = []
    url = f"{API_URL}?pageSize=100"

    while url:
        async with session.get(url, headers={"X-Goog-Api-Key": JULES_API_KEY}) as resp:
            if resp.status != 200:
                logger.error(
                    f"Failed to list sessions: {resp.status} {await resp.text()}"
                )
                break

            data = await resp.json()
            page_sessions = data.get("sessions", [])
            sessions.extend(page_sessions)

            next_token = data.get("nextPageToken")
            if next_token:
                url = f"{API_URL}?pageSize=100&pageToken={next_token}"
            else:
                url = None

    return sessions


from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
async def delete_session_safe(session, session_name, semaphore):
    async with semaphore:
        url = f"https://jules.googleapis.com/v1alpha/{session_name}"
        async with session.delete(
            url, headers={"X-Goog-Api-Key": JULES_API_KEY}
        ) as resp:
            if resp.status == 200:
                logger.info(f"Deleted {session_name}")
                return True
            elif resp.status == 429:
                raise Exception("Rate limit")
            elif resp.status == 404:
                return True  # Already gone
            else:
                logger.error(f"Failed to delete {session_name}: {resp.status}")
                return False


async def purge_all():
    async with aiohttp.ClientSession() as session:
        logger.info("Listing sessions...")
        sessions = await list_sessions(session)
        logger.info(f"Found {len(sessions)} active sessions.")

        if not sessions:
            return

        semaphore = asyncio.Semaphore(5)
        tasks = []
        for s in sessions:
            name = s.get("name")
            if name:
                tasks.append(delete_session_safe(session, name, semaphore))

        await asyncio.gather(*tasks)
        logger.info("Purge complete.")


if __name__ == "__main__":
    asyncio.run(purge_all())
