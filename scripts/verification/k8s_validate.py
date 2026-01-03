import os
import sys
import tempfile
import traceback
from pathlib import Path

import boto3
from cortex.ingestion.conv_manifest.validation import scan_and_refresh
from pydantic import VERSION as PYDANTIC_VERSION


def _validate_environment():
    """Validate and return required environment variables."""
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
    return {var.lower(): os.environ.get(var) for var in required_vars}


def _initialize_s3_client(
    endpoint_url, aws_access_key_id, aws_secret_access_key, region_name
):
    """Initialize and return a boto3 S3 client."""
    if region_name and "digitaloceanspaces.com" in endpoint_url:
        endpoint_url = f"https://{region_name}.digitaloceanspaces.com"
    print(f"Connecting to S3: {endpoint_url}")
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )
    sts_client = boto3.client(
        "sts",
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )
    account_id = sts_client.get_caller_identity()["Account"]
    return s3_client, account_id


def _list_s3_folders(s3_client, bucket, prefix, max_folders):
    """List folders in an S3 bucket with a given prefix."""
    print(f"Listing folders in {bucket}/{prefix}...")
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/")
    folders = []
    for page in pages:
        if "CommonPrefixes" in page:
            for cp in page["CommonPrefixes"]:
                folders.append(cp["Prefix"])
                if len(folders) >= max_folders:
                    return folders
    return folders


def _download_sample_files(
    s3_client,
    bucket,
    s3_prefix,
    folders,
    max_files_per_folder,
    target_dir,
    account_id,
):
    """Download sample files from a list of S3 folders."""
    print(f"Found {len(folders)} folders. Downloading sample...")
    for i, folder in enumerate(folders):
        print(f"[{i + 1}/{len(folders)}] Processing {folder}")
        object_paginator = s3_client.get_paginator("list_objects_v2")
        object_pages = object_paginator.paginate(Bucket=bucket, Prefix=folder)
        files_downloaded = 0
        for page in object_pages:
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                if files_downloaded >= max_files_per_folder:
                    break
                key = obj["Key"]
                if key.startswith(s3_prefix):
                    rel = key[len(s3_prefix) :]
                    dest_path = target_dir / rel
                    resolved_dest = dest_path.resolve()
                    if not resolved_dest.is_relative_to(target_dir.resolve()):
                        print(
                            f"  [!] WARNING: Skipping potentially malicious path: {key}"
                        )
                        continue
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    s3_client.download_file(
                        bucket,
                        key,
                        str(dest_path),
                        ExtraArgs={"ExpectedBucketOwner": account_id},
                    )
                    files_downloaded += 1
            if files_downloaded >= max_files_per_folder:
                print(f"  [i] Reached file limit for folder: {max_files_per_folder}")
                break


def main():
    try:
        env_vars = _validate_environment()
        s3, account_id = _initialize_s3_client(
            env_vars["s3_endpoint"],
            env_vars["s3_access_key"],
            env_vars["s3_secret_key"],
            env_vars["s3_region"],
        )
        s3_prefix = "Outlook/"
        max_folders_to_process = 20
        max_files_per_folder = 10
        folders = _list_s3_folders(
            s3, env_vars["s3_bucket_raw"], s3_prefix, max_folders_to_process
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = Path(temp_dir)
            print(f"Using temporary directory: {target_dir}")
            _download_sample_files(
                s3,
                env_vars["s3_bucket_raw"],
                s3_prefix,
                folders,
                max_files_per_folder,
                target_dir,
                account_id,
            )
            print("\n\n" + "=" * 50)
            print("RUNNING EXPORT VALIDATION (In-Cluster)")
            print("=" * 50)
            report = scan_and_refresh(target_dir)
            print("\nValidation Results:")
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
