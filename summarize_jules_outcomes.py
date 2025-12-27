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
    except Exception:
        pass


async def get_outcome(session, http_sess):
    sid = session.get("session_id")
    file = session.get("file")
    if not sid:
        return {"file": file, "status": "No ID", "pr": "-"}

    url = f"https://jules.googleapis.com/v1alpha/{sid}"
    headers = {"X-Goog-Api-Key": JULES_API_KEY}

    async with http_sess.get(url, headers=headers) as resp:
        if resp.status == 200:
            data = await resp.json()
            state = data.get("state")
            outputs = data.get("outputs", [])
            pr_link = "-"
            pr_title = "-"

            if outputs:
                for o in outputs:
                    if "pullRequest" in o:
                        pr_link = o["pullRequest"].get("url", "-")
                        pr_title = o["pullRequest"].get("title", "-")
                        break

            return {
                "file": file,
                "state": state,
                "pr_link": pr_link,
                "pr_title": pr_title,
            }
        return {"file": file, "state": f"HTTP {resp.status}", "pr": "Error"}


async def main():
    async with aiohttp.ClientSession() as http_sess:
        tasks = [get_outcome(s, http_sess) for s in sessions]
        results = await asyncio.gather(*tasks)

        print(f"{'File':<50} | {'State':<12} | {'PR Title':<30} | {'PR Link'}")
        print("-" * 120)

        prs_found = 0
        for r in results:
            print(
                f"{r['file']:<50} | {r['state']:<12} | {r.get('pr_title', '-')[:30]:<30} | {r.get('pr_link', '-')}"
            )
            if r.get("pr_link") != "-":
                prs_found += 1

        print("-" * 120)
        print(f"Total PRs Created: {prs_found}")

        # Save summary
        with open("jules_final_summary.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
