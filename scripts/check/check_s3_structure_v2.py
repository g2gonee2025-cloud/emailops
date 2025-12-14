import json
import os
import random
from collections import defaultdict
from pathlib import Path

import boto3
from botocore.config import Config

# Load .env explicitly
env_path = Path(".env")
if env_path.exists():
    print(f"Loading .env from {env_path.absolute()}")
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()
else:
    print(".env file not found!")

# Configuration from .env
ENDPOINT = os.getenv("S3_ENDPOINT")
REGION = os.getenv("S3_REGION")
BUCKET = os.getenv("S3_BUCKET_RAW")
ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
SECRET_KEY = os.getenv("S3_SECRET_KEY")
PREFIX = "raw/outlook/"
SAMPLE = 20

if not ACCESS_KEY or not SECRET_KEY:
    print("Error: Missing S3_ACCESS_KEY or S3_SECRET_KEY in .env")
    exit(1)

print(f"Connecting to {ENDPOINT} bucket {BUCKET}...")

client = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    region_name=REGION,
    config=Config(signature_version="s3v4"),
)

print(f"Listing objects in {PREFIX}...")
folders = set()
try:
    count = 0
    paginator = client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX, Delimiter="/")

    for page in page_iterator:
        if "CommonPrefixes" in page:
            for prefix in page["CommonPrefixes"]:
                folder_path = prefix["Prefix"]
                folder_name = folder_path[len(PREFIX) :].strip("/")
                if folder_name:
                    folders.add(folder_name)
                    count += 1

        if count >= 100:
            break

except Exception as e:
    print(f"Error listing objects: {e}")
    exit(1)

print(f"Found {len(folders)} folders (stopped early if > 100).")

if not folders:
    print("No folders found.")
    exit(0)

sample = random.sample(sorted(folders), min(SAMPLE, len(folders)))
expected = {"Conversation.txt", "manifest.json", "attachments/"}


def list_folder_contents(folder: str) -> list[str]:
    folder_prefix = f"{PREFIX}{folder}/"
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=folder_prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                keys.append(obj["Key"])
    return keys


def has_key(keys: list[str], folder: str, expected_entry: str) -> bool:
    pref = f"{PREFIX}{folder}/{expected_entry}"
    return any(k == pref or k.startswith(pref) for k in keys)


issues: dict[str, dict[str, list[str]]] = defaultdict(dict)
print(f"Checking {len(sample)} sampled folders...")

for folder in sample:
    keys = list_folder_contents(folder)
    missing = [e for e in expected if not has_key(keys, folder, e)]

    if missing:
        issues[folder]["missing"] = missing

print(
    json.dumps(
        {
            "sampled_folders": sample,
            "issues": issues,
            "total_folders_found_in_scan": len(folders),
        },
        indent=2,
    )
)
