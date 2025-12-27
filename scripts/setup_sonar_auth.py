import os
from pathlib import Path

import requests

SONAR_URL = "http://localhost:9000"
PWD = "!Sharjah2085"
TOKEN_NAME = f"agent-token-{os.getpid()}"

try:
    print(f"Generating token '{TOKEN_NAME}'...")
    r = requests.post(
        f"{SONAR_URL}/api/user_tokens/generate",
        params={"name": TOKEN_NAME},
        auth=("admin", PWD),
    )
    if r.status_code != 200:
        print(f"Error: {r.status_code} {r.text}")
        exit(1)

    token = r.json()["token"]
    print("Token generated successfully.")

    # Append to .env
    env_path = "/root/workspace/emailops-vertex-ai/.env"
    with Path(env_path).open("a") as f:
        f.write("\n# SonarQube Credentials (Generated)\n")
        f.write(f"SONAR_TOKEN={token}\n")
        f.write(f"SONAR_HOST_URL={SONAR_URL}\n")

    print("Updated " + env_path)

except (requests.exceptions.RequestException, OSError, ValueError, KeyError) as e:
    print(f"Error setting up Sonar auth: {e}")
    exit(1)
