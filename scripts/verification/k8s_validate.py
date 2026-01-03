import os
import sys
import tempfile
import traceback
from pathlib import Path

import boto3
from cortex.ingestion.conv_manifest.validation import scan_and_refresh
from pydantic import VERSION as PYDANTIC_VERSION


def main():
    try:
        # Validate required environment variables.
        required_vars = [
            "S3_ENDPOINT",
            "S3_ACCESS_KEY",
            "S3_SECRET_KEY",
            "S3_REGION",
            "S3_BUCKET_RAW",
        ]
        if any(not os.environ.get(var) for var in required_vars):
            missing = [var for var in required_vars if not os.environ.get(var)]
            print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
            sys.exit(1)

        s3_endpoint = os.environ.get("S3_ENDPOINT")
        s3_key = os.environ.get("S3_ACCESS_KEY")
        s3_secret = os.environ.get("S3_SECRET_KEY")
        s3_region = os.environ.get("S3_REGION")
        bucket = os.environ.get("S3_BUCKET_RAW")

        # Fix: If endpoint contains bucket name (virtual host style), strip it or reset it
        # DigitalOcean Spaces endpoint should be https://region.digitaloceanspaces.com
        if s3_region and s3_endpoint and "digitaloceanspaces.com" in s3_endpoint:
            s3_endpoint = f"https://{s3_region}.digitaloceanspaces.com"

        print(f"Connecting to S3: {s3_endpoint} (Bucket: {bucket})")

        s3 = boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id=s3_key,
            aws_secret_access_key=s3_secret,
            region_name=s3_region,
        )

        # --- Configuration ---
        s3_prefix = "Outlook/"
        max_folders_to_process = 20
        max_files_per_folder = 10
        # ---------------------

        print(f"Listing folders in {bucket}/{s3_prefix}...")

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix, Delimiter="/")

        folders = []
        for page in pages:
            if "CommonPrefixes" in page:
                for cp in page["CommonPrefixes"]:
                    folders.append(cp["Prefix"])
                    if len(folders) >= max_folders_to_process:
                        break
            if len(folders) >= max_folders_to_process:
                break

        print(f"Found {len(folders)} folders. Downloading sample...")
        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = Path(temp_dir)
            print(f"Using temporary directory: {target_dir}")
            for i, folder in enumerate(folders):
                print(f"[{i + 1}/{len(folders)}] Processing {folder}")

                # Use a paginator to handle folders with more than 1000 objects.
                object_paginator = s3.get_paginator("list_objects_v2")
                object_pages = object_paginator.paginate(Bucket=bucket, Prefix=folder)

                files_downloaded = 0
                for page in object_pages:
                    if "Contents" not in page:
                        continue

                    for obj in page["Contents"]:
                        if files_downloaded >= max_files_per_folder:
                            break
                        key = obj["Key"]
                        # key e.g. "Outlook/Folder/file.txt"
                        # we want local: "/tmp/validation_sample/Folder/file.txt"

                        # strip "Outlook/"
                        if key.startswith(s3_prefix):
                            rel = key[len(s3_prefix):]
                            dest_path = target_dir / rel

                            # Prevent path traversal: resolve the path and ensure it's within the target directory.
                            resolved_dest = dest_path.resolve()
                            if not resolved_dest.is_relative_to(target_dir.resolve()):
                                print(f"  [!] WARNING: Skipping potentially malicious path: {key}")
                                continue

                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            s3.download_file(bucket, key, str(dest_path))
                            files_downloaded += 1
                    if files_downloaded >= max_files_per_folder:
                        print(f"  [i] Reached file limit for folder: {max_files_per_folder}")
                        break

            print("\n\n" + "=" * 50)
            print("RUNNING EXPORT VALIDATION (In-Cluster)")
            print("=" * 50)

            report = scan_and_refresh(target_dir)

            print("\nValidation Results:")
            # Handle Pydantic v1/v2 compatibility for serialization.
            if PYDANTIC_VERSION.startswith("1."):
                print(report.json(indent=2))
            else:
                print(report.model_dump_json(indent=2))

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
