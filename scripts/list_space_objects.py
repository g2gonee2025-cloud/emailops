#!/usr/bin/env python3
"""
List existing objects in DigitalOcean Space using repo config.
Prints a summary and sample keys without touching any running uploads.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config

# Load app config
sys.path.append(str(Path("backend/src").resolve()))
from cortex.config.loader import get_config  # type: ignore

SAMPLE_LIMIT = 100
PREFIXES_TO_CHECK = [
    "",
    "raw/",
    "raw/outlook/",
    # Observed keys appear to include bucket name as part of key path
    "emailops-storage-tor1/raw/outlook/",
]


def make_client(endpoint_url: str, region: str, access_key: str, secret_key: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(
            signature_version="s3v4", retries={"max_attempts": 3, "mode": "adaptive"}
        ),
    )


def region_root_endpoint(space_endpoint: str) -> str:
    parsed = urlparse(space_endpoint)
    host = parsed.netloc
    if host.endswith(".digitaloceanspaces.com"):
        parts = host.split(".")
        if len(parts) >= 3:
            region = parts[-3]
            return f"https://{region}.digitaloceanspaces.com"
    return space_endpoint


def iter_keys(s3, bucket: str, prefix: str, limit: int) -> tuple[int, list[str]]:
    keys: list[str] = []
    total = 0
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        total += resp.get("KeyCount", 0)
        for c in resp.get("Contents", [])[: max(0, limit - len(keys))]:
            keys.append(c["Key"])
            if len(keys) >= limit:
                break
        if len(keys) >= limit:
            break
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return total, keys


def main() -> None:
    cfg = get_config()
    bucket = cfg.storage.bucket_raw
    endpoint = cfg.storage.endpoint_url
    region = cfg.storage.region
    access_key = cfg.storage.access_key
    secret_key = cfg.storage.secret_key

    root_endpoint = region_root_endpoint(endpoint)

    s3 = make_client(root_endpoint, region, access_key, secret_key)

    print(f"Bucket: {bucket}")
    print(f"Endpoint: {root_endpoint}")

    for pfx in PREFIXES_TO_CHECK:
        try:
            total, sample = iter_keys(s3, bucket, pfx, SAMPLE_LIMIT)
        except Exception as e:
            print(f"Prefix '{pfx}': ERROR: {e}")
            continue
        print(f"Prefix '{pfx or '/'}': total={total}, sample={len(sample)}")
        for k in sample[:10]:
            print(f"  - {k}")


if __name__ == "__main__":
    main()
