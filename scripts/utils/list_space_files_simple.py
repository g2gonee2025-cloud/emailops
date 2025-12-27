"""
Standalone utility to list objects in a DigitalOcean Space.

Similar to `aws s3 ls`, but uses local .env file for auth.
"""
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
    """Main function to list objects in a DigitalOcean Space."""
    parser = argparse.ArgumentParser(
        description="List objects in a configured S3-compatible bucket (e.g., DigitalOcean Space)."
    )
    parser.add_argument(
        "prefix",
        nargs="?",
        default="",
        help="Prefix to filter objects (e.g., 'raw/outlook/'). Optional.",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=200,
        help="Maximum number of objects to display.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print connection details and other debug info.",
    )
    args = parser.parse_args()

    # Load .env into the environment (override to pick up latest values).
    load_dotenv(override=True)

    env = _ensure_env(
        ["S3_ENDPOINT", "S3_REGION", "S3_BUCKET_RAW", "S3_ACCESS_KEY", "S3_SECRET_KEY"]
    )

    if args.verbose:
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

        # Use rich for table output if available
        try:
            from rich import box
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(
                title=f"Objects in {env['S3_BUCKET_RAW']}/{args.prefix}",
                box=box.SIMPLE,
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Key")
            table.add_column("Size", justify="right")
            table.add_column("Last Modified")

            RICH_AVAILABLE = True
        except ImportError:
            RICH_AVAILABLE = False

        count = 0
        for page in pages:
            for obj in page.get("Contents", []):
                if count < args.limit:
                    if RICH_AVAILABLE:
                        table.add_row(
                            obj["Key"],
                            str(obj["Size"]),
                            str(obj["LastModified"]),
                        )
                    else:
                        print(
                            f"- {obj['Key']} (size={obj['Size']} "
                            f"last_modified={obj['LastModified']})"
                        )
                count += 1

        if count == 0:
            print("No objects found.")
        else:
            if RICH_AVAILABLE:
                console.print(table)

            # Summary line
            displayed = min(count, args.limit)
            print(
                f"\nDisplayed {displayed} of {count} object(s). "
                f"(use --limit to change)"
            )

    except Exception as exc:  # pragma: no cover - simple utility script
        print(f"Error listing objects: {exc}")


if __name__ == "__main__":
    main()
