#!/usr/bin/env python3
"""
Upload files from C:/Users/ASUS/Desktop/Outlook to DigitalOcean Spaces.
Uses boto3 with direct credential configuration and robust error handling.
"""

import argparse
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Configuration
S3_ENDPOINT = "https://tor1.digitaloceanspaces.com"
S3_REGION = "tor1"
S3_BUCKET = "emailops-storage-tor1"
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
if not S3_ACCESS_KEY or not S3_SECRET_KEY:
    raise ValueError(
        "S3_ACCESS_KEY and S3_SECRET_KEY environment variables must be set."
    )

MAX_WORKERS = 4  # Conservative number


def get_s3_client():
    """Create S3 client for DigitalOcean Spaces."""
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=10,
            read_timeout=30,
        ),
    )


def get_content_type(file_path: Path) -> str:
    content_type, _ = mimetypes.guess_type(str(file_path))
    return content_type or "application/octet-stream"


def upload_file(s3_client, local_path: Path, s3_key: str, overwrite: bool = False) -> str:
    """
    Upload a single file atomically using a conditional PutObject operation.
    This avoids a TOCTOU (Time-of-check to time-of-use) race condition.

    Returns status string ('uploaded', 'skipped', 'failed').
    """
    filename = local_path.name
    try:
        content_type = get_content_type(local_path)
        print(f"Uploading: {filename}...", flush=True)

        put_args = {
            "Bucket": S3_BUCKET,
            "Key": s3_key,
            "ContentType": content_type,
        }
        if not overwrite:
            # IfNoneMatch: '*' makes the upload conditional on the object not existing.
            # If it exists, S3 returns a 412 PreconditionFailed error.
            put_args["IfNoneMatch"] = "*"

        with open(local_path, "rb") as f:
            put_args["Body"] = f
            s3_client.put_object(**put_args)

        print(f"Done: {filename}", flush=True)
        return "uploaded"
    except ClientError as e:
        if e.response["Error"]["Code"] == "PreconditionFailed":
            print(f"Skipped (already exists): {filename}", flush=True)
            return "skipped"
        else:
            # Handle other S3-related errors
            print(f"Error (S3) {filename}: {e}", flush=True)
            return f"failed: {e}"
    except (IOError, OSError) as e:
        # Handle file I/O errors specifically
        print(f"Error (I/O) {filename}: {e}", flush=True)
        return f"failed: {e}"
    except Exception as e:
        # Catch any other unexpected errors and provide a traceback
        import traceback
        print(f"An unexpected error occurred with {filename}:", flush=True)
        traceback.print_exc()
        return f"failed: {e}"


def discover_files(source_dir: Path, bucket_prefix: str):
    """
    Generator function to discover files in the source directory.
    Yields tuples of (local_path, s3_key).
    """
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist.")
        return
    for root, _, files in os.walk(source_dir):
        for file in files:
            local_path = Path(root) / file
            relative_path = local_path.relative_to(source_dir)
            s3_key = bucket_prefix + str(relative_path).replace("\\", "/")
            yield local_path, s3_key


def main():
    parser = argparse.ArgumentParser(
        description="Upload files to DigitalOcean Spaces."
    )
    parser.add_argument(
        "source_dir", type=Path, help="Local directory to upload from."
    )
    parser.add_argument(
        "bucket_prefix", type=str, help="Prefix to use for S3 keys."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files if they already exist in the bucket.",
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    bucket_prefix = args.bucket_prefix
    overwrite_files = args.overwrite

    s3_client = get_s3_client()
    file_generator = discover_files(source_dir, bucket_prefix)

    print(f"Starting upload with {MAX_WORKERS} workers...", flush=True)

    stats = {"uploaded": 0, "skipped": 0, "failed": 0}
    processed_files = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all files from the generator to the executor
        futures = {
            executor.submit(upload_file, s3_client, path, key, overwrite_files): key
            for path, key in file_generator
        }

        if not futures:
            print("No files found to upload.")
            return

        for i, future in enumerate(as_completed(futures)):
            processed_files += 1
            result = future.result()
            if result == "uploaded":
                stats["uploaded"] += 1
            elif result == "skipped":
                stats["skipped"] += 1
            else:
                stats["failed"] += 1

            if processed_files % 5 == 0:
                print(
                    f"Progress: {processed_files} processed | Up: {stats['uploaded']} | Skip: {stats['skipped']} | Fail: {stats['failed']}",
                    flush=True,
                )

    print("\nComplete!", flush=True)
    print(f"Total files processed: {processed_files}")
    print(stats)


if __name__ == "__main__":
    main()
