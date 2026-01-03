#!/usr/bin/env python3
"""
List existing objects in DigitalOcean Space using repo config.
Prints a summary and sample keys without touching any running uploads.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from cortex.common.exceptions import ConfigurationError
from cortex.config.loader import get_config

SAMPLE_LIMIT = 100


def make_client(
    endpoint_url: str, region: str, access_key: str | None, secret_key: str | None
):
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


def get_keys_sample(s3, bucket: str, prefix: str, limit: int) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        remaining = limit - len(keys)
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": min(remaining, 1000)}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for c in resp.get("Contents", [])[: max(0, limit - len(keys))]:
            keys.append(c["Key"])
            if len(keys) >= limit:
                break
        if len(keys) >= limit:
            break
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def redact_key(key: str, head: int = 30, tail: int = 30) -> str:
    """Redact the middle of a key for safe display."""
    if len(key) <= head + tail:
        return key
    return f"{key[:head]}...{key[-tail:]}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List objects in a DigitalOcean Space."
    )
    parser.add_argument(
        "prefixes",
        nargs="*",
        default=["", "raw/", "raw/outlook/"],
        help="Optional list of prefixes to check.",
    )
    args = parser.parse_args()

    cfg = get_config()
    bucket = cfg.storage.bucket_raw
    endpoint = str(cfg.storage.endpoint_url) if cfg.storage.endpoint_url else None
    region = cfg.storage.region

    if not all([bucket, endpoint, region]):
        raise ConfigurationError(
            "Missing required storage configuration: bucket, endpoint, or region."
        )

    access_key_secret = cfg.storage.access_key
    secret_key_secret = cfg.storage.secret_key
    access_key = access_key_secret.get_secret_value() if access_key_secret else None
    secret_key = secret_key_secret.get_secret_value() if secret_key_secret else None

    s3 = make_client(endpoint, region, access_key, secret_key)

    print(f"Bucket: {bucket}")
    print(f"Endpoint: {endpoint}")

    for pfx in args.prefixes:
        try:
            sample = get_keys_sample(s3, bucket, pfx, SAMPLE_LIMIT)
        except ClientError as e:
            print(f"Prefix '{pfx}': ERROR: {e}")
            continue
        print(f"Prefix '{pfx or '/'}': sample={len(sample)}")
        for k in sample[:10]:
            print(f"  - {redact_key(k)}")


if __name__ == "__main__":
    main()
