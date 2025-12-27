import asyncio
import json
import logging
import os

import aiohttp

# Config
REPO_OWNER = "g2gonee2025-cloud"
REPO_NAME = "emailops"
API_URL = "https://jules.googleapis.com/v1alpha/sessions"

JULES_API_KEY = os.getenv("JULES_API_KEY_ALT")
if not JULES_API_KEY:
    JULES_API_KEY = os.getenv("JULES_API_KEY")

if not JULES_API_KEY:
    try:
        with open(".env") as f:
            for line in f:
                if line.startswith("JULES_API_KEY_ALT="):
                    JULES_API_KEY = line.strip().split("=", 1)[1]
                    break
    except Exception:
        pass


async def fetch_sessions():
    async with aiohttp.ClientSession() as session:
        headers = {"X-Goog-Api-Key": JULES_API_KEY, "Content-Type": "application/json"}

        # List sessions (pagination might be needed if > 50, but we likely have few recent ones)
        # Assuming pageSize default is reasonable or max.
        url = f"{API_URL}?pageSize=100"

        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                print(f"Error fetching sessions: {resp.status} {await resp.text()}")
                return []

            data = await resp.json()
            return data.get("sessions", [])


def main():
    print("Fetching sessions from Jules API...")
    sessions = asyncio.run(fetch_sessions())
    print(f"Found {len(sessions)} total sessions.")

    # Filter for our Surgical Strike sessions
    # Title format: "Deep Review: {file_path}"
    surgical_results = []

    for s in sessions:
        title = s.get("title", "")
        if title.startswith("Deep Review:"):
            file_path = title.replace("Deep Review: ", "")
            surgical_results.append(
                {
                    "file": file_path,
                    "status": "success",
                    "session_id": s.get("name"),
                    "response": s,  # Store full object including state/replies if available?
                    # Note: list sessions might be shallow.
                    # Ideally we want the result, but listing might just give metadata.
                    # However, for the report, just knowing ID + status + metadata is good start.
                    # We can print the 'latestMessage' or similar if present.
                }
            )

    print(f"Recovered {len(surgical_results)} surgical review sessions.")

    with open("jules_surgical_report.json", "w") as f:
        json.dump(surgical_results, f, indent=2)
    print("Saved recovered report to jules_surgical_report.json")


if __name__ == "__main__":
    main()
