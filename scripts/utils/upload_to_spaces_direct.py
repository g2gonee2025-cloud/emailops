#!/usr/bin/env python3
"""
Upload files from Desktop/Outlook to DigitalOcean Spaces.

Uses boto3 with direct credential configuration and robust error handling.
"""

import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config

# Configuration
S3_ENDPOINT = "https://emailops-storage-tor1.tor1.digitaloceanspaces.com"
S3_REGION = "tor1"
S3_BUCKET = "emailops-storage-tor1"
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "DO00XHY7KTQGELEBRGYV")
S3_SECRET_KEY = os.getenv(
    "S3_SECRET_KEY", "UjuF90LBXLVdJFH6mMvxL9+mkE7peuyP5RL1oVYiyNs"
)

SOURCE_DIR = Path(r"C:\Users\ASUS\Desktop\Outlook")
BUCKET_PREFIX = "raw/outlook/"
MAX_WORKERS = 5  # Reduced from 10 to avoid rate limiting


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
            connect_timeout=30,
            read_timeout=60,
        ),
    )


def get_content_type(file_path: Path) -> str:
    """Determine content type based on file extension."""
    content_type, _ = mimetypes.guess_type(str(file_path))
    return content_type or "application/octet-stream"


def upload_file(s3_client, local_path: Path, s3_key: str) -> tuple[bool, str]:
    """Upload a single file to S3. Returns (success, message)."""
    try:
        content_type = get_content_type(local_path)
        file_size = local_path.stat().st_size
        s3_client.upload_file(
            str(local_path),
            S3_BUCKET,
            s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        return True, f"{s3_key} ({file_size} bytes)"
    except Exception as e:
        return False, f"{s3_key}: {e!s}"


def main():
    """Main upload function."""
    print(f"Scanning files in {SOURCE_DIR}...")

    # Collect all files
    files_to_upload = []
    for root, _dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            local_path = Path(root) / file
            relative_path = local_path.relative_to(SOURCE_DIR)
            s3_key = BUCKET_PREFIX + str(relative_path).replace("\\", "/")
            files_to_upload.append((local_path, s3_key))

    total_files = len(files_to_upload)
    print(f"Found {total_files} files to upload")

    if total_files == 0:
        print("No files to upload!")
        return

    # Create S3 client
    s3_client = get_s3_client()

    # Skip bucket check - proceed directly to upload
    print(f"Uploading to {S3_BUCKET} with prefix '{BUCKET_PREFIX}'")
    print(f"Using {MAX_WORKERS} concurrent workers...")
    print("-" * 60)

    # Upload files with progress
    start_time = time.time()
    uploaded = 0
    failed = 0
    failed_files = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(upload_file, s3_client, local_path, s3_key): s3_key
            for local_path, s3_key in files_to_upload
        }

        for future in as_completed(futures):
            success, result = future.result()
            if success:
                uploaded += 1
            else:
                failed += 1
                failed_files.append(result)

            # Progress update every 50 files
            total_processed = uploaded + failed
            if total_processed % 50 == 0 or total_processed == total_files:
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                eta = (total_files - total_processed) / rate if rate > 0 else 0
                print(
                    f"Progress: {total_processed}/{total_files} ({100*total_processed/total_files:.1f}%) | "
                    f"Uploaded: {uploaded} | Failed: {failed} | "
                    f"Rate: {rate:.1f}/s | ETA: {eta:.0f}s"
                )

    # Summary
    elapsed = time.time() - start_time
    print("-" * 60)
    print("\n✓ Upload complete!")
    print(f"  Total files:    {total_files}")
    print(f"  Uploaded:       {uploaded}")
    print(f"  Failed:         {failed}")
    print(f"  Time elapsed:   {elapsed:.1f} seconds")
    if total_files > 0:
        print(f"  Average rate:   {total_files/elapsed:.1f} files/sec")

    if failed_files:
        print("\nFailed files (first 20):")
        for f in failed_files[:20]:
            print(f"  - {f}")
        if len(failed_files) > 20:
            print(f"  ... and {len(failed_files) - 20} more")


if __name__ == "__main__":
    main()
