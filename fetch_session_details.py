import asyncio
import json
import os

import aiohttp

SESSION_ID = "sessions/113369174496315368"
API_URL = f"https://jules.googleapis.com/v1alpha/{SESSION_ID}"

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
    except:
        pass


async def get_session():
    async with aiohttp.ClientSession() as session:
        headers = {"X-Goog-Api-Key": JULES_API_KEY}

        print(f"Fetching {API_URL}...")
        async with session.get(API_URL, headers=headers) as resp:
            print(f"Status: {resp.status}")
            if resp.status == 200:
                data = await resp.json()
                print(json.dumps(data, indent=2))
            else:
                print(await resp.text())


if __name__ == "__main__":
    asyncio.run(get_session())
