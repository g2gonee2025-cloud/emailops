#!/usr/bin/env python3
"""
S3 Ingestion Utility CLI.
Consolidates scanning, real-data fetching, and deep dive inspection.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import boto3
from botocore.client import Config

# Add backend/src to path (depth 2: scripts/ingestion/s3_cli.py)
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir / "backend" / "src"))

from cortex.config.loader import get_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("s3_cli")


def get_s3_client(config):
    s3_config = config.storage
    return boto3.client(
        "s3",
        region_name=s3_config.region,
        endpoint_url=s3_config.endpoint_url,
        aws_access_key_id=s3_config.access_key,
        aws_secret_access_key=s3_config.secret_key,
        config=Config(signature_version="s3v4"),
    )


def list_roots(_args):
    """List root folders in the bucket."""
    config = get_config()
    storage = getattr(config, "storage", None)
    if storage is None:
        raise ValueError("Storage configuration is missing")
    for _attr in ("region", "endpoint_url", "access_key", "secret_key"):
        if not hasattr(storage, _attr):
            raise ValueError(f"Storage configuration missing '{_attr}'")
    client = get_s3_client(config)
    bucket = storage.bucket_raw

    print(f"Listing roots in '{bucket}'...")
    paginator = client.get_paginator("list_objects_v2")
    # Using delimiter to simulate folders
    for page in paginator.paginate(Bucket=bucket, Delimiter="/"):
        prefixes = page.get("CommonPrefixes", [])
        for p in prefixes:
            print(f"- {p['Prefix']}")


def fetch_sample(args):
    """Find a real multi-message manifest and dump it."""
    config = get_config()
    client = get_s3_client(config)
    bucket = config.storage.bucket_raw

    print(f"Scanning '{bucket}' for manifest with >1 messages...")
    paginator = client.get_paginator("list_objects_v2")

    scanned = 0
    max_scan = args.max_scan

    # Ensure the max-scan limit applies to total objects examined, not just manifests
    pagination_kwargs = {"Bucket": bucket, "PaginationConfig": {"PageSize": 50}}
    if max_scan:
        pagination_kwargs["PaginationConfig"]["MaxItems"] = max_scan
    for page in paginator.paginate(**pagination_kwargs):
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith("manifest.json"):
                scanned += 1
                try:
                    resp = client.get_object(Bucket=bucket, Key=key)
                    data = json.loads(resp["Body"].read().decode("utf-8"))

                    msgs = data.get("messages", [])
                    is_multi = isinstance(msgs, list) and len(msgs) > 1

                    if is_multi or args.any:
                        print(f"\n✅ FOUND REAL DATA: {key}")
                        print("=" * 60)
                        print(json.dumps(data, indent=2))
                        print("=" * 60)
                        return
                    else:
                        sys.stdout.write(".")
                        sys.stdout.flush()

                except Exception as e:
                    logger.warning(f"Failed to read {key}: {e}")

            if scanned >= max_scan:
                print(f"\nStopped after {scanned} manifests.")
                return


def main():
    parser = argparse.ArgumentParser(description="S3 Ingestion Utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # roots
    subparsers.add_parser("roots", help="List root folders")

    # sample
    p_sample = subparsers.add_parser("sample", help="Fetch a sample manifest")
    p_sample.add_argument(
        "--max-scan", type=int, default=100, help="Max manifests to scan"
    )
    p_sample.add_argument(
        "--any",
        action="store_true",
        help="Return first manifest found (ignore multi-message req)",
    )

    args = parser.parse_args()

    if args.command == "roots":
        list_roots(args)
    elif args.command == "sample":
        fetch_sample(args)


if __name__ == "__main__":
    main()
