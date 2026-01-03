#!/usr/bin/env python3
"""
S3 Ingestion Utility CLI.
Consolidates scanning, real-data fetching, and deep dive inspection.
"""

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


def find_project_root(marker="pyproject.toml"):
    """Find the project root by searching for a marker file."""
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Project root marker '{marker}' not found.")


# Add project root to path
try:
    project_root = find_project_root()
    sys.path.append(str(project_root / "backend" / "src"))
    from cortex.config.loader import get_config
    from cortex.ingestion.pii import PIIEngine
except (FileNotFoundError, ImportError) as e:
    print(f"Error: Failed to set up path and imports. {e}", file=sys.stderr)
    sys.exit(1)


logger = logging.getLogger(__name__)


def redact_manifest_pii(data, pii_engine: PIIEngine):
    """
    Recursively redact sensitive data in a dictionary or list.

    Operates on a deep copy to avoid mutating the original data.
    """
    data = deepcopy(data)

    def _redact_recursive(item):
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, str):
                    item[key] = pii_engine.redact(value)
                else:
                    _redact_recursive(value)
        elif isinstance(item, list):
            for i, value in enumerate(item):
                if isinstance(value, str):
                    item[i] = pii_engine.redact(value)
                else:
                    _redact_recursive(value)
        return item

    return _redact_recursive(data)


def get_s3_client(config):
    """Initialize S3 client using secure credentials."""
    s3_config = getattr(config, "storage", None)
    if not s3_config:
        raise ValueError("Storage configuration is missing.")

    client_args = {
        "region_name": s3_config.region,
        "endpoint_url": s3_config.endpoint_url,
        "config": Config(signature_version="s3v4"),
    }

    # Conditionally add credentials to support IAM roles
    if hasattr(s3_config, "access_key") and hasattr(s3_config, "secret_key"):
        client_args["aws_access_key_id"] = s3_config.access_key
        client_args["aws_secret_access_key"] = s3_config.secret_key

    return boto3.client("s3", **client_args)


def list_roots(args):
    """List root folders in the bucket."""
    config = get_config()
    storage = getattr(config, "storage", None)
    if not storage or not hasattr(storage, "bucket_raw"):
        raise ValueError("Storage configuration 'bucket_raw' is missing.")

    for attr in ("region", "endpoint_url"):
        if not hasattr(storage, attr):
            raise ValueError(f"Storage configuration missing '{attr}'")

    client = get_s3_client(config)
    bucket = storage.bucket_raw

    print(f"Listing roots in '{bucket}'...")
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Delimiter="/"):
            prefixes = page.get("CommonPrefixes", [])
            if not prefixes:
                print("No root folders found.")
                continue
            for p in prefixes:
                print(f"- {p['Prefix']}")
    except ClientError as e:
        logger.error(f"Failed to list roots in '{bucket}': {e}")


def fetch_sample(args):
    """Find a real multi-message manifest and dump it."""
    config = get_config()
    client = get_s3_client(config)
    storage = getattr(config, "storage", None)
    if not storage or not hasattr(storage, "bucket_raw"):
        raise ValueError("Storage configuration 'bucket_raw' is missing.")
    bucket = storage.bucket_raw

    print(f"Scanning '{bucket}' for manifest with >1 messages...")
    paginator = client.get_paginator("list_objects_v2")

    max_scan = args.max_scan

    pagination_kwargs = {
        "Bucket": bucket,
        "PaginationConfig": {"PageSize": 50},
    }
    if args.prefix:
        pagination_kwargs["Prefix"] = args.prefix
        print(f"Scanning with prefix: {args.prefix}...")

    # A max_scan of 0 means unlimited scan
    if max_scan > 0:
        pagination_kwargs["PaginationConfig"]["MaxItems"] = max_scan

    found_manifest = False
    for page in paginator.paginate(**pagination_kwargs):
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith("manifest.json"):
                try:
                    resp = client.get_object(Bucket=bucket, Key=key)
                    with resp["Body"] as stream:
                        data = json.loads(stream.read().decode("utf-8"))

                    msgs = data.get("messages", [])
                    is_multi = isinstance(msgs, list) and len(msgs) > 1

                    if is_multi or args.any:
                        pii_engine = PIIEngine(strict=False)
                        print(f"\n✅ FOUND REAL DATA: {key}")
                        print("=" * 60)
                        redacted_data = redact_manifest_pii(data, pii_engine)
                        print(json.dumps(redacted_data, indent=2))
                        print("=" * 60)
                        found_manifest = True
                        return
                    else:
                        sys.stdout.write(".")
                        sys.stdout.flush()

                except ClientError as e:
                    logger.warning(f"S3 client error for '{key}': {e}")
                except json.JSONDecodeError:
                    logger.warning(f"Corrupt manifest JSON in '{key}'")
                except Exception as e:
                    logger.error(f"Unexpected error processing '{key}': {e}")

    if not found_manifest:
        print("\nNo matching manifest found within the scan limit.")


def main():
    parser = argparse.ArgumentParser(description="S3 Ingestion Utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # roots
    subparsers.add_parser("roots", help="List root folders")

    # sample
    p_sample = subparsers.add_parser("sample", help="Fetch a sample manifest")
    p_sample.add_argument(
        "--max-scan",
        type=int,
        default=1000,
        help="Max S3 objects to scan (0 for unlimited)",
    )
    p_sample.add_argument(
        "--any",
        action="store_true",
        help="Return first manifest found (ignore multi-message req)",
    )
    p_sample.add_argument(
        "--prefix", type=str, default="", help="S3 prefix to narrow the scan"
    )

    args = parser.parse_args()

    # Associate commands with functions
    commands = {
        "roots": list_roots,
        "sample": fetch_sample,
    }
    # Execute the command
    commands[args.command](args)


if __name__ == "__main__":
    main()
