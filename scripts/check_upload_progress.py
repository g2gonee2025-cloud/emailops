#!/usr/bin/env python3
"""
Non-intrusive upload progress check for DigitalOcean Spaces.
- Counts local files under C:/Users/ASUS/Desktop/Outlook
- Counts remote objects under s3://<bucket>/raw/outlook/
- Outputs a simple progress summary without touching the running upload process.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import boto3
from botocore.config import Config

sys.path.append(str(Path("backend/src").resolve()))
from urllib.parse import urlparse

from cortex.config.loader import get_config  # type: ignore

SOURCE_DIR = Path(r"C:\Users\ASUS\Desktop\Outlook")
PREFIX = "raw/outlook/"


def iter_local_files(root: Path) -> Iterator[Path]:
    for r, _d, files in os.walk(root):
        for f in files:
            yield Path(r) / f


def count_remote_objects(s3, bucket: str, prefix: str) -> int:
    total = 0
    token = None
    while True:
        kwargs = {
            "Bucket": bucket,
            "Prefix": prefix,
            "MaxKeys": 1000,
        }
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        total += resp.get("KeyCount", 0)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return total


def main() -> None:
    cfg = get_config()
    bucket = cfg.storage.bucket_raw
    endpoint = cfg.storage.endpoint_url
    region = cfg.storage.region
    access_key = cfg.storage.access_key
    secret_key = cfg.storage.secret_key

    # Use region root endpoint for list operations (avoid bucket-host NoSuchKey)
    endpoint_parsed = urlparse(endpoint)
    endpoint_host = endpoint_parsed.netloc
    region_root = None
    if endpoint_host.endswith(".digitaloceanspaces.com"):
        parts = endpoint_host.split(".")
        # expect: <bucket>.<region>.digitaloceanspaces.com
        if len(parts) >= 3:
            region_part = parts[-3]  # e.g., tor1
            region_root = f"https://{region_part}.digitaloceanspaces.com"

    list_endpoint = region_root or endpoint

    s3 = boto3.client(
        "s3",
        endpoint_url=list_endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(
            signature_version="s3v4", retries={"max_attempts": 3, "mode": "adaptive"}
        ),
    )

    local_total = (
        sum(1 for _ in iter_local_files(SOURCE_DIR)) if SOURCE_DIR.exists() else 0
    )
    try:
        remote_total = count_remote_objects(s3, bucket, PREFIX)
    except Exception as e:
        print(f"Warning: list failed at {list_endpoint}: {e}")
        # Fallback: try original endpoint
        s3_fallback = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )
        remote_total = count_remote_objects(s3_fallback, bucket, PREFIX)

    pct = (remote_total / local_total * 100.0) if local_total > 0 else 0.0

    print("Progress summary:")
    print(f"  Local files:  {local_total}")
    print(f"  Remote objs:  {remote_total} (prefix={PREFIX})")
    print(f"  Estimated %:  {pct:.2f}%")
    if local_total > 0 and remote_total > local_total:
        print("  Note: remote count exceeds local; duplicates or prior runs may exist.")


if __name__ == "__main__":
    main()
