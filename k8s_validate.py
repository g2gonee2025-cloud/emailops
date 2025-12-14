import os
import sys
from pathlib import Path

import boto3
from cortex.ingestion.conv_manifest.validation import scan_and_refresh


def main():
    try:
        s3_endpoint = os.environ.get("S3_ENDPOINT")
        s3_key = os.environ.get("S3_ACCESS_KEY")
        s3_secret = os.environ.get("S3_SECRET_KEY")
        s3_region = os.environ.get("S3_REGION")
        bucket = os.environ.get("S3_BUCKET_RAW")

        # Fix: If endpoint contains bucket name (virtual host style), strip it or reset it
        # DigitalOcean Spaces endpoint should be https://region.digitaloceanspaces.com
        if s3_region and "digitaloceanspaces.com" in s3_endpoint:
            s3_endpoint = f"https://{s3_region}.digitaloceanspaces.com"

        print(f"Connecting to S3: {s3_endpoint} (Bucket: {bucket})")

        s3 = boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id=s3_key,
            aws_secret_access_key=s3_secret,
            region_name=s3_region,
        )

        prefix = "Outlook/"
        print(f"Listing folders in {bucket}/{prefix}...")

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/")

        folders = []
        for page in pages:
            if "CommonPrefixes" in page:
                for cp in page["CommonPrefixes"]:
                    folders.append(cp["Prefix"])
                    if len(folders) >= 20:
                        break
            if len(folders) >= 20:
                break

        print(f"Found {len(folders)} folders. Downloading sample...")
        target_dir = Path("/tmp/validation_sample")

        # Clean up if exists
        import shutil

        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        for i, folder in enumerate(folders):
            print(f"[{i+1}/{len(folders)}] Processing {folder}")

            # List objects in folder
            # We don't want to use paginator for subfolder if it's small, but to be safe
            objs = s3.list_objects_v2(Bucket=bucket, Prefix=folder)

            if "Contents" not in objs:
                continue

            for obj in objs["Contents"]:
                key = obj["Key"]
                # key e.g. "Outlook/Folder/file.txt"
                # we want local: "/tmp/validation_sample/Folder/file.txt"

                # strip "Outlook/"
                if key.startswith(prefix):
                    rel = key[len(prefix) :]
                    dest_path = target_dir / rel
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(bucket, key, str(dest_path))

        print("\n\n" + "=" * 50)
        print("RUNNING EXPORT VALIDATION (In-Cluster)")
        print("=" * 50)

        report = scan_and_refresh(target_dir)

        print("\nValidation Results:")
        print(report.model_dump_json(indent=2))

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
