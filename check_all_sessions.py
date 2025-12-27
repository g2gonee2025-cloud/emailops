import asyncio
import json
import os

import aiohttp

# Load report
with open("jules_surgical_report.json") as f:
    sessions = json.load(f)

# Config
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


async def check_session(session, http_sess):
    sid = session.get("session_id")
    file = session.get("file")
    if not sid:
        return {"file": file, "status": "no_id"}

    url = f"https://jules.googleapis.com/v1alpha/{sid}"
    headers = {"X-Goog-Api-Key": JULES_API_KEY}

    async with http_sess.get(url, headers=headers) as resp:
        if resp.status == 200:
            data = await resp.json()
            return {"file": file, "state": data.get("state"), "data": data}
        return {"file": file, "state": f"HTTP {resp.status}"}


async def main():
    async with aiohttp.ClientSession() as http_sess:
        tasks = [check_session(s, http_sess) for s in sessions]
        results = await asyncio.gather(*tasks)

        print(f"{'File':<50} | {'State':<15}")
        print("-" * 70)
        for r in results:
            print(f"{r['file']:<50} | {r.get('state', 'Unknown'):<15}")

        # Check messages for one
        sample_sid = sessions[0].get("session_id")
        if sample_sid:
            msg_url = f"https://jules.googleapis.com/v1alpha/{sample_sid}/messages"
            print(f"\nChecking messages for {sample_sid}...")
            async with http_sess.get(
                msg_url, headers={"X-Goog-Api-Key": JULES_API_KEY}
            ) as resp:
                print(f"Messages Status: {resp.status}")
                if resp.status == 200:
                    data = await resp.json()
                    msgs = data.get("messages", [])
                    print(f"Found {len(msgs)} messages.")
                    for m in msgs:
                        print("-" * 20)
                        print(m.get("text", "No text"))


if __name__ == "__main__":
    asyncio.run(main())
