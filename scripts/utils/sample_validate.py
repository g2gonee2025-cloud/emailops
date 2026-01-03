import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from cortex.ingestion.conv_manifest.validation import scan_and_refresh
from dotenv import load_dotenv


# STYLE: Use a named constant instead of a magic number
MAX_FOLDERS_TO_SAMPLE = 10


def main() -> None:
    load_dotenv(override=True)

    # NULL_SAFETY: Validate environment variables
    required_vars = ["S3_ENDPOINT", "S3_REGION", "S3_BUCKET_RAW", "S3_ACCESS_KEY", "S3_SECRET_KEY"]
    if any(not os.getenv(var) for var in required_vars):
        raise ValueError(f"Missing one or more required environment variables: {', '.join(required_vars)}")

    endpoint = os.getenv("S3_ENDPOINT")

    # SECURITY: endpoint_url is accepted from environment without enforcing HTTPS
    if endpoint and urlparse(endpoint).scheme != "https":
        raise ValueError("S3_ENDPOINT must use https://")

    region = os.getenv("S3_REGION")
    bucket = os.getenv("S3_BUCKET_RAW")
    key = os.getenv("S3_ACCESS_KEY")
    secret = os.getenv("S3_SECRET_KEY")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
    )

    prefix = "raw/outlook/"
    paginator = client.get_paginator("list_objects_v2")
    folders: list[str] = []
    if not bucket:
        raise ValueError("S3 bucket is not specified; cannot paginate objects.")
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            folders.extend(cp["Prefix"] for cp in page.get("CommonPrefixes", []))
            if len(folders) >= MAX_FOLDERS_TO_SAMPLE:
                break
    except ClientError as e:
        print(f"Error listing S3 objects: {e}")
        return
    folders = folders[:MAX_FOLDERS_TO_SAMPLE]
    print(f"Selected folders: {len(folders)}")

    # EXCEPTION_HANDLING: Ensure temporary directory is always cleaned up
    with tempfile.TemporaryDirectory(prefix="conv_sample_") as temp_dir:
        root = Path(temp_dir)
        print(f"Downloading to {root}")

        def download_folder(folder_prefix: str) -> Path:
            # LOGIC_ERRORS: Destination folder name collision
            dest_folder_name = folder_prefix.rstrip("/").replace("/", "_")
            dest = root / dest_folder_name

            created_dirs = set()

            for page in paginator.paginate(Bucket=bucket, Prefix=folder_prefix):
                for obj in page.get("Contents", []):
                    rel = obj["Key"][len(folder_prefix) :]
                    if not rel:
                        continue
                    target = (dest / rel).resolve()

                    # SECURITY: Path traversal risk
                    if not str(target).startswith(str(dest.resolve())):
                        print(f"Skipping potentially malicious path: {obj['Key']}")
                        continue

                    # PERFORMANCE: Avoid redundant mkdir calls
                    parent_dir = target.parent
                    if parent_dir not in created_dirs:
                        parent_dir.mkdir(parents=True, exist_ok=True)
                        created_dirs.add(parent_dir)

                    try:
                        client.download_file(bucket, obj["Key"], str(target))
                    except ClientError as e:
                        # Do not abort the entire batch on a single object failure
                        print(f"Failed to download {obj['Key']}: {e}")
            return dest

        local_folders = [download_folder(f) for f in folders]
        print(f"Downloaded folders: {len(local_folders)}")
        for p in local_folders:
            print(p.name, "Conversation.txt exists?", (p / "Conversation.txt").exists())

        # EXCEPTION_HANDLING: Catch exceptions during validation
        try:
            report = scan_and_refresh(root)
            print("Validation report:")
            print(f"  folders_scanned: {report.folders_scanned}")
            print(f"  manifests_created: {report.manifests_created}")
            print(f"  manifests_updated: {report.manifests_updated}")
            print(f"  problems: {len(report.problems)}")
            for prob in report.problems[:20]:
                print(f"   - {prob.folder}: {prob.issue}")
        except Exception as e:
            print(f"An error occurred during validation: {e}")


if __name__ == "__main__":
    main()
