#!/usr/bin/env python3
"""
Upload files from C:/Users/ASUS/Desktop/Outlook to DigitalOcean Spaces.
Uses boto3 with direct credential configuration and robust error handling.
"""

import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Configuration
S3_ENDPOINT = "https://emailops-storage-tor1.tor1.digitaloceanspaces.com"
S3_REGION = "tor1"
S3_BUCKET = "emailops-storage-tor1"
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")

SOURCE_DIR = Path(r"C:\Users\ASUS\Desktop\Outlook")
BUCKET_PREFIX = "raw/outlook/"
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


def file_exists(s3_client, s3_key: str) -> bool:
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        return True
    except ClientError:
        return False


def upload_file(s3_client, local_path: Path, s3_key: str) -> str:
    """Upload a single file. Returns status string."""
    filename = local_path.name
    try:
        # Check existence first
        if file_exists(s3_client, s3_key):
            return "skipped"

        content_type = get_content_type(local_path)
        print(f"Uploading: {filename}...", flush=True)

        s3_client.upload_file(
            str(local_path),
            S3_BUCKET,
            s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        print(f"Done: {filename}", flush=True)
        return "uploaded"
    except Exception as e:
        print(f"Error {filename}: {e}", flush=True)
        return f"failed: {e}"


def main():
    print(f"Scanning files in {SOURCE_DIR}...", flush=True)

    files_to_upload = []
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")
        return

    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            local_path = Path(root) / file
            relative_path = local_path.relative_to(SOURCE_DIR)
            s3_key = BUCKET_PREFIX + str(relative_path).replace("\\", "/")
            files_to_upload.append((local_path, s3_key))

    total_files = len(files_to_upload)
    print(f"Found {total_files} files.", flush=True)

    if total_files == 0:
        return

    s3_client = get_s3_client()

    print(f"Starting upload with {MAX_WORKERS} workers...", flush=True)

    stats = {"uploaded": 0, "skipped": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(upload_file, s3_client, path, key): key
            for path, key in files_to_upload
        }

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result == "uploaded":
                stats["uploaded"] += 1
            elif result == "skipped":
                stats["skipped"] += 1
            else:
                stats["failed"] += 1

            if (i + 1) % 5 == 0:
                print(
                    f"Progress: {i + 1}/{total_files} | Up: {stats['uploaded']} | Skip: {stats['skipped']} | Fail: {stats['failed']}",
                    flush=True,
                )

    print("\nComplete!", flush=True)
    print(stats)


if __name__ == "__main__":
    main()
