import argparse
import json
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError
from cortex.ingestion.conv_manifest.validation import scan_and_refresh
from dotenv import load_dotenv


def download_object(
    client, bucket: str, key: str, folder_prefix: str, dest: Path
) -> None:
    """Worker function to download a single S3 object."""
    rel = key[len(folder_prefix) :]
    if not rel:
        return

    target = (dest / rel).resolve()
    if not str(target).startswith(str(dest.resolve())):
        print(f"Skipping potentially malicious path: {rel!r}", file=sys.stderr)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(target))


def download_folder(client, bucket: str, folder_prefix: str, dest_root: Path) -> Path:
    """List S3 objects and download them concurrently."""
    dest = dest_root / folder_prefix.rstrip("/").split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)

    paginator = client.get_paginator("list_objects_v2")
    objects_to_download = []
    for page in paginator.paginate(Bucket=bucket, Prefix=folder_prefix):
        objects_to_download.extend(page.get("Contents", []))

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                download_object, client, bucket, obj["Key"], folder_prefix, dest
            )
            for obj in objects_to_download
        ]
        for future in futures:
            future.result()  # Raise exceptions if any occurred

    return dest


def load_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8-sig")
    except Exception as e:
        # EXCEPTION_HANDLING: Log exceptions before returning None
        logging.warning(f"Could not read {path} with utf-8-sig, trying utf-8: {e}")
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as e2:
            logging.error(f"Failed to read {path} as text: {e2}")
            return None


def redact_manifest(text: str | None) -> str:
    """Redact sensitive fields from manifest JSON for safe logging."""
    if text is None:
        return "(missing or unreadable)"
    try:
        data = json.loads(text)
        for key in ["subject_label", "participants", "last_from", "last_to"]:
            if key in data:
                data[key] = "<redacted>"
        return json.dumps(data, indent=2)
    except json.JSONDecodeError:
        return "(invalid JSON)"


def main() -> None:
    # STYLE: Use argparse for command-line arguments
    parser = argparse.ArgumentParser(
        description="Download a folder from S3, validate its manifest, and show the diff."
    )
    parser.add_argument(
        "folder_prefix",
        type=str,
        help="The S3 folder prefix to download and process.",
    )
    args = parser.parse_args()
    folder_prefix = args.folder_prefix

    # SECURITY: Do not override existing environment variables from .env file.
    load_dotenv()

    # NULL_SAFETY: Validate required environment variables
    required_vars = {
        "S3_ENDPOINT",
        "S3_REGION",
        "S3_BUCKET_RAW",
        "S3_ACCESS_KEY",
        "S3_SECRET_KEY",
    }
    missing_vars = [v for v in required_vars if not os.getenv(v)]
    if missing_vars:
        print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
        exit(1)

    endpoint = os.getenv("S3_ENDPOINT")
    region = os.getenv("S3_REGION")
    bucket = os.getenv("S3_BUCKET_RAW")
    key = os.getenv("S3_ACCESS_KEY")
    secret = os.getenv("S3_SECRET_KEY")

    # EXCEPTION_HANDLING: Catch errors during S3 client creation
    try:
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            config=Config(signature_version="s3v4"),
        )
    except BotoCoreError as e:
        print(f"Error creating S3 client: {e}", file=sys.stderr)
        sys.exit(1)

    # EXCEPTION_HANDLING: Use TemporaryDirectory for automatic cleanup
    with tempfile.TemporaryDirectory(prefix="manifest_diff_") as temp_dir:
        root = Path(temp_dir)
        print(f"Temp root: {root}")

        try:
            # TYPE_ERRORS: The check above ensures `bucket` is not None.
            conv_dir = download_folder(client, bucket, folder_prefix, root)
            manifest_path = conv_dir / "manifest.json"

            before = load_text(manifest_path)
            print("\n--- BEFORE manifest.json ---")
            print(redact_manifest(before))

            report = scan_and_refresh(root)
            after = load_text(manifest_path)

            print("\n--- AFTER manifest.json ---")
            print(redact_manifest(after))

            print(
                f"\nValidation summary: folders_scanned={report.folders_scanned} "
                f"created={report.manifests_created} updated={report.manifests_updated} "
                f"problems={len(report.problems)}"
            )
        except (BotoCoreError, OSError) as e:
            print(f"An error occurred during processing: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
