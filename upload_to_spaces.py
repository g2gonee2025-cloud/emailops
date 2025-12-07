#!/usr/bin/env python3
"""
Upload extracted Outlook conversations to DigitalOcean Spaces.
Uses boto3 with concurrent uploads for efficiency.
"""

import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# DigitalOcean Spaces configuration
S3_ENDPOINT = "https://sgp1.digitaloceanspaces.com"
S3_REGION = "sgp1"
S3_BUCKET = "emailops-storage-sgp1"
S3_ACCESS_KEY = "DO00Z8YMD9NLWRAD78FE"
S3_SECRET_KEY = "AXwqbJfprO69xy3hWpxVvvJDCace0r8UIgHP5rCQ3Fw"

# Source directory
SOURCE_DIR = Path(r"C:\Users\ASUS\Desktop\Outlook")

# Prefix in the bucket
BUCKET_PREFIX = "raw/outlook/"

# Max concurrent uploads
MAX_WORKERS = 10


def get_s3_client():
    """Create S3 client for DigitalOcean Spaces."""
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(
            signature_version="s3v4", retries={"max_attempts": 3, "mode": "adaptive"}
        ),
    )


def get_content_type(file_path: Path) -> str:
    """Determine content type based on file extension."""
    content_type, _ = mimetypes.guess_type(str(file_path))
    return content_type or "application/octet-stream"


def check_file_exists(s3_client, s3_key: str) -> bool:
    """Check if a file already exists in S3."""
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        return True
    except ClientError:
        return False


def upload_file(
    s3_client, local_path: Path, s3_key: str, skip_existing: bool = True
) -> tuple[bool, str, bool]:
    """Upload a single file to S3. Returns (success, message, skipped)."""
    try:
        # Check if file already exists
        if skip_existing and check_file_exists(s3_client, s3_key):
            return True, s3_key, True  # Skipped

        content_type = get_content_type(local_path)
        s3_client.upload_file(
            str(local_path), S3_BUCKET, s3_key, ExtraArgs={"ContentType": content_type}
        )
        return True, s3_key, False  # Uploaded
    except Exception as e:
        return False, f"{s3_key}: {e}", False


def main():
    """Main upload function."""
    print(f"Scanning files in {SOURCE_DIR}...")

    # Collect all files
    files_to_upload = []
    for root, _dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            local_path = Path(root) / file
            # Create relative path for S3 key
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

    # Test connection
    print("Testing connection to DigitalOcean Spaces...")
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
        print(f"✓ Connected to bucket: {S3_BUCKET}")
    except Exception as e:
        print(f"✗ Failed to connect to bucket: {e}")
        return

    # Upload files with progress
    start_time = time.time()
    uploaded = 0
    skipped = 0
    failed = 0
    failed_files = []

    print(f"\nUploading {total_files} files with {MAX_WORKERS} concurrent workers...")
    print("(Skipping files that already exist in bucket)")
    print("-" * 60)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(upload_file, s3_client, local_path, s3_key, True): s3_key
            for local_path, s3_key in files_to_upload
        }

        for future in as_completed(futures):
            success, result, was_skipped = future.result()
            if success:
                if was_skipped:
                    skipped += 1
                else:
                    uploaded += 1
            else:
                failed += 1
                failed_files.append(result)

            # Progress update every 100 files
            total_processed = uploaded + skipped + failed
            if total_processed % 100 == 0 or total_processed == total_files:
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                eta = (total_files - total_processed) / rate if rate > 0 else 0
                print(
                    f"Progress: {total_processed}/{total_files} ({100*total_processed/total_files:.1f}%) | "
                    f"New: {uploaded} | Skip: {skipped} | Rate: {rate:.1f}/s | ETA: {eta:.0f}s"
                )

    # Summary
    elapsed = time.time() - start_time
    print("-" * 60)
    print("\n✓ Upload complete!")
    print(f"  Total files:    {total_files}")
    print(f"  Uploaded (new): {uploaded}")
    print(f"  Skipped (exist):{skipped}")
    print(f"  Failed:         {failed}")
    print(f"  Time elapsed:   {elapsed:.1f} seconds")
    print(f"  Average rate:   {total_files/elapsed:.1f} files/sec")

    if failed_files:
        print("\nFailed files:")
        for f in failed_files[:10]:
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")


if __name__ == "__main__":
    main()
