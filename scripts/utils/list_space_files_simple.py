import argparse
import os
from collections.abc import Iterable

import boto3
from botocore.client import Config
from dotenv import load_dotenv


def _ensure_env(keys: Iterable[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    missing = []
    for key in keys:
        value = os.getenv(key)
        if value:
            value = value.strip()
        if not value:
            missing.append(key)
        else:
            values[key] = value
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List objects in the configured DigitalOcean Space"
    )
    parser.add_argument(
        "--prefix", default="", help="Prefix to filter objects (e.g. raw/outlook/)"
    )
    parser.add_argument(
        "--limit", type=int, default=200, help="Maximum number of objects to print"
    )
    args = parser.parse_args()

    # Load .env into the environment (override to pick up latest values).
    load_dotenv(override=True)

    env = _ensure_env(
        ["S3_ENDPOINT", "S3_REGION", "S3_BUCKET_RAW", "S3_ACCESS_KEY", "S3_SECRET_KEY"]
    )

    print(f"Connecting to {env['S3_BUCKET_RAW']} @ {env['S3_ENDPOINT']}")
    print(f"Access key length: {len(env['S3_ACCESS_KEY'])}")
    print(f"Secret key length: {len(env['S3_SECRET_KEY'])}")

    client = boto3.client(
        "s3",
        endpoint_url=env["S3_ENDPOINT"],
        region_name=env["S3_REGION"],
        aws_access_key_id=env["S3_ACCESS_KEY"],
        aws_secret_access_key=env["S3_SECRET_KEY"],
        config=Config(signature_version="s3v4"),
    )

    try:
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=env["S3_BUCKET_RAW"], Prefix=args.prefix)
        count = 0
        printed = 0
        for page in pages:
            for obj in page.get("Contents", []):
                count += 1
                if printed < args.limit:
                    print(
                        f"- {obj['Key']} (size={obj['Size']} last_modified={obj['LastModified']})"
                    )
                    printed += 1
        if count == 0:
            print("No files found.")
        else:
            print(f"Displayed {printed} of {count} object(s) (use --limit to change)")
    except Exception as exc:  # pragma: no cover - simple utility script
        print(f"Error listing files: {exc}")


if __name__ == "__main__":
    main()
