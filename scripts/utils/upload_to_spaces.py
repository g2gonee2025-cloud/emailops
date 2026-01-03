#!/usr/bin/env python3
"""
Upload extracted Outlook conversations to DigitalOcean Spaces.
Uses boto3 with concurrent uploads for efficiency.
"""

import argparse
import mimetypes
import os
import sys
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError


# --- Path setup ---
def find_project_root(marker: str = "pyproject.toml") -> Path:
    """Find the project root by searching upwards for a marker file."""
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(
        f"Could not find project root marker '{marker}' starting from {current_path}"
    )


def setup_sys_path():
    """Add the project's 'backend/src' directory to the system path."""
    try:
        project_root = find_project_root()
        src_path = project_root / "backend" / "src"
        if src_path.is_dir():
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
        else:
            print(
                f"Warning: '{src_path}' not found. Imports may fail.", file=sys.stderr
            )
    except RuntimeError as e:
        print(f"Warning: {e}. Imports may fail.", file=sys.stderr)


setup_sys_path()

# cortex.config is now available
from cortex.config.loader import AppConfig, get_config

# --- Constants ---

# Prefix in the bucket
BUCKET_PREFIX = "raw/outlook/"

# Max concurrent uploads
MAX_WORKERS = 10

# Max failed files to store in memory
MAX_FAILED_FILES_TO_STORE = 1000


def get_s3_client(config: AppConfig):
    """Create S3 client for DigitalOcean Spaces."""
    try:
        return boto3.client(
            "s3",
            endpoint_url=config.storage.endpoint_url,
            region_name=config.storage.region,
            aws_access_key_id=config.storage.access_key,
            aws_secret_access_key=config.storage.secret_key,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )
    except (NoCredentialsError, PartialCredentialsError) as e:
        print(f"✗ AWS credentials not found or incomplete: {e}", file=sys.stderr)
        sys.exit(1)


def get_content_type(file_path: Path) -> str:
    """Determine content type based on file extension."""
    content_type, _ = mimetypes.guess_type(str(file_path))
    return content_type or "application/octet-stream"


def upload_file(
    s3_client, local_path: Path, s3_key: str, s3_bucket: str, skip_existing: bool = True
) -> tuple[bool, str, bool]:
    """
    Upload a single file to S3. Returns (success, message, skipped).

    Handles 'file already exists' check atomically.
    """
    try:
        content_type = get_content_type(local_path)
        extra_args = {"ContentType": content_type}

        if skip_existing:
            try:
                s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
                return True, s3_key, True  # Skipped
            except ClientError as e:
                if e.response["Error"]["Code"] != "404":
                    raise  # Re-raise other client errors (permissions, etc.)

        s3_client.upload_file(str(local_path), s3_bucket, s3_key, ExtraArgs=extra_args)
        return True, s3_key, False  # Uploaded
    except ClientError as e:
        # Boto3 can raise ClientError for various reasons (e.g., permissions)
        return False, f"{s3_key}: {e}", False
    except FileNotFoundError:
        return False, f"Local file not found: {local_path}", False
    except Exception as e:
        # Catching other unexpected errors
        return False, f"Unexpected error for {s3_key}: {e}", False


def scan_files(source_dir: Path) -> Generator[tuple[Path, str], None, None]:
    """Scan and yield files to upload, avoiding loading all into memory."""
    for root, _, files in os.walk(source_dir):
        for file in files:
            local_path = Path(root) / file
            relative_path = local_path.relative_to(source_dir)
            s3_key = BUCKET_PREFIX + str(relative_path).replace("\\", "/")
            yield local_path, s3_key


def main(source_dir_str: str, skip_existing: bool):
    """Main upload function."""
    source_dir = Path(source_dir_str)
    if not source_dir.is_dir():
        print(f"✗ Source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    print("Loading configuration...")
    try:
        config = get_config()
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)

    s3_bucket = config.storage.bucket_raw
    if not s3_bucket:
        print("✗ S3 bucket not configured in .env file.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning files in {source_dir}...")
    s3_client = get_s3_client(config)

    print("Testing connection to DigitalOcean Spaces...")
    try:
        s3_client.head_bucket(Bucket=s3_bucket)
        print(f"✓ Connected to bucket: {s3_bucket}")
    except ClientError as e:
        print(f"✗ Failed to connect to bucket '{s3_bucket}': {e}", file=sys.stderr)
        sys.exit(1)

    start_time = time.time()
    uploaded, skipped, failed = 0, 0, 0
    failed_files = []
    total_files = 0  # We will count as we go

    print(f"\nUploading files with {MAX_WORKERS} concurrent workers...")
    if skip_existing:
        print("(Skipping files that already exist in bucket)")
    print("-" * 60)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Use the generator to submit tasks
        file_generator = scan_files(source_dir)
        futures = {
            executor.submit(
                upload_file, s3_client, local_path, s3_key, s3_bucket, skip_existing
            ): s3_key
            for local_path, s3_key in file_generator
        }
        total_files = len(futures)

        if total_files == 0:
            print("No files found to upload!")
            return

        for future in as_completed(futures):
            try:
                success, result, was_skipped = future.result()
                if success:
                    if was_skipped:
                        skipped += 1
                    else:
                        uploaded += 1
                else:
                    failed += 1
                    if len(failed_files) < MAX_FAILED_FILES_TO_STORE:
                        failed_files.append(result)

                # Progress update
                total_processed = uploaded + skipped + failed
                if total_processed % 100 == 0 or total_processed == total_files:
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed if elapsed > 0.001 else 0
                    eta_seconds = (
                        (total_files - total_processed) / rate if rate > 0.001 else 0
                    )
                    print(
                        f"Progress: {total_processed}/{total_files} ({100 * total_processed / total_files:.1f}%) | "
                        f"New: {uploaded} | Skip: {skipped} | Fail: {failed} | "
                        f"Rate: {rate:.1f}/s | ETA: {eta_seconds:.0f}s"
                    )
            except Exception as e:
                failed += 1
                s3_key_for_future = futures.get(future, "unknown")
                err_msg = f"Worker for '{s3_key_for_future}' raised an exception: {e}"
                if len(failed_files) < MAX_FAILED_FILES_TO_STORE:
                    failed_files.append(err_msg)

    elapsed = time.time() - start_time
    print("-" * 60)
    print("\n✓ Upload complete!")
    print(f"  Total files:    {total_files}")
    print(f"  Uploaded (new): {uploaded}")
    print(f"  Skipped (exist):{skipped}")
    print(f"  Failed:         {failed}")
    print(f"  Time elapsed:   {elapsed:.1f} seconds")
    if elapsed > 0.001:
        avg_rate = total_files / elapsed
        print(f"  Average rate:   {avg_rate:.1f} files/sec")

    if failed_files:
        print("\n--- Failed files ---")
        for f in failed_files[:10]:
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload extracted Outlook conversations to DigitalOcean Spaces."
    )
    parser.add_argument(
        "source_dir",
        type=str,
        help="The local source directory containing files to upload.",
    )
    parser.add_argument(
        "--no-skip",
        action="store_false",
        dest="skip_existing",
        help="Overwrite files in the bucket even if they already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.source_dir, args.skip_existing)
