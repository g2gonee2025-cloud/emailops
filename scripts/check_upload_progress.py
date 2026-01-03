#!/usr/bin/env python3
"""
Non-intrusive upload progress check for DigitalOcean Spaces.
- Counts local files under a specified directory.
- Counts remote objects under a specified S3 prefix.
- Outputs a simple progress summary without touching the running upload process.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.src.cortex.config.loader import get_config

# Constants for S3 client
S3_RETRY_ATTEMPTS = 3


def iter_local_files(root: Path) -> Iterator[Path]:
    """Iterate over all files in a directory."""
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            yield Path(dirpath) / filename


def count_remote_objects(s3, bucket: str, prefix: str) -> int:
    """Count objects in an S3-compatible service using a paginator."""
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    return sum(page.get("KeyCount", 0) for page in pages)


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Check upload progress for DigitalOcean Spaces.")
    parser.add_argument(
        "source_dir",
        type=Path,
        help="The local source directory to scan for files.",
    )
    parser.add_argument(
        "--prefix",
        default="raw/outlook/",
        help="The S3 prefix to scan for remote objects.",
    )
    args = parser.parse_args()

    cfg = get_config()
    if not all(
        [
            cfg,
            cfg.storage,
            cfg.storage.bucket_raw,
            cfg.storage.endpoint_url,
            cfg.storage.region,
            cfg.storage.access_key,
            cfg.storage.secret_key,
        ]
    ):
        print("Error: Storage configuration is missing or incomplete.", file=sys.stderr)
        sys.exit(1)

    bucket = cfg.storage.bucket_raw
    endpoint = cfg.storage.endpoint_url
    region = cfg.storage.region
    access_key = cfg.storage.access_key
    secret_key = cfg.storage.secret_key

    # Prepend a default scheme if missing.
    if "://" not in endpoint:
        endpoint = f"https://{endpoint}"

    # Use region root endpoint for list operations (avoid bucket-host NoSuchKey)
    endpoint_parsed = urlparse(endpoint)
    hostname = endpoint_parsed.hostname
    region_root = None
    if hostname and hostname.endswith(".digitaloceanspaces.com"):
        parts = hostname.split(".")
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
            signature_version="s3v4",
            retries={"max_attempts": S3_RETRY_ATTEMPTS, "mode": "adaptive"},
            s3={"addressing_style": "path"},
        ),
    )

    local_total = (
        sum(1 for _ in iter_local_files(args.source_dir))
        if args.source_dir.exists()
        else 0
    )
    remote_total = 0
    try:
        remote_total = count_remote_objects(s3, bucket, args.prefix)
    except ClientError as e:
        print(f"Warning: list failed at {list_endpoint}: {e}", file=sys.stderr)
        # Fallback: try original endpoint
        s3_fallback = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": S3_RETRY_ATTEMPTS, "mode": "adaptive"},
                s3={"addressing_style": "path"},
            ),
        )
        try:
            remote_total = count_remote_objects(s3_fallback, bucket, args.prefix)
        except ClientError as e_fallback:
            print(
                f"Error: Fallback list failed at {endpoint}: {e_fallback}",
                file=sys.stderr,
            )
            sys.exit(1)

    if local_total == 0:
        pct = float("inf") if remote_total > 0 else 0.0
    else:
        pct = (remote_total / local_total) * 100.0

    print("Progress summary:")
    print(f"  Local files:  {local_total}")
    print(f"  Remote objs:  {remote_total} (prefix={args.prefix})")
    print(f"  Estimated %:  {pct:.2f}%")
    if local_total > 0 and remote_total > local_total:
        print("  Note: remote count exceeds local; duplicates or prior runs may exist.")


if __name__ == "__main__":
    main()
