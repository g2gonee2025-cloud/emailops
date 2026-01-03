import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.client import BaseClient, Config
from botocore.exceptions import ClientError
from cortex.ingestion.conv_manifest.validation import ValidationReport, scan_and_refresh
from dotenv import load_dotenv

MAX_FOLDERS_TO_SAMPLE = 10


def _setup_environment() -> dict[str, str]:
    """Load and validate required environment variables."""
    load_dotenv(override=True)
    required_vars = [
        "S3_ENDPOINT",
        "S3_REGION",
        "S3_BUCKET_RAW",
        "S3_ACCESS_KEY",
        "S3_SECRET_KEY",
    ]
    optional_vars = ["S3_BUCKET_OWNER"]
    env_vars = {var: os.getenv(var) for var in required_vars + optional_vars}

    missing_required = [key for key in required_vars if not env_vars[key]]
    if missing_required:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_required)}"
        )

    endpoint = env_vars["S3_ENDPOINT"]
    if endpoint and urlparse(endpoint).scheme != "https":
        raise ValueError("S3_ENDPOINT must use https://")

    return {k: v for k, v in env_vars.items() if v is not None}


def _create_s3_client(endpoint: str, region: str, key: str, secret: str) -> BaseClient:
    """Create and return a boto3 S3 client."""
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
    )


def _list_s3_folders(
    client: BaseClient, bucket: str, prefix: str, bucket_owner: str | None
) -> list[str]:
    """List folders in an S3 bucket with a given prefix."""
    paginator = client.get_paginator("list_objects_v2")
    folders = []
    paginate_kwargs = {"Bucket": bucket, "Prefix": prefix, "Delimiter": "/"}
    if bucket_owner:
        paginate_kwargs["ExpectedBucketOwner"] = bucket_owner
    try:
        for page in paginator.paginate(**paginate_kwargs):
            folders.extend(
                cp.get("Prefix", "") for cp in page.get("CommonPrefixes", [])
            )
            if len(folders) >= MAX_FOLDERS_TO_SAMPLE:
                break
    except ClientError as e:
        print(f"Error listing S3 objects: {e}")
    return folders[:MAX_FOLDERS_TO_SAMPLE]


def _download_folder(
    client: BaseClient,
    bucket: str,
    folder_prefix: str,
    root: Path,
    bucket_owner: str | None,
) -> Path:
    """Download a folder from S3 to a local directory."""
    dest_folder_name = folder_prefix.rstrip("/").replace("/", "_")
    dest = root / dest_folder_name
    created_dirs = set()
    paginator = client.get_paginator("list_objects_v2")

    paginate_kwargs = {"Bucket": bucket, "Prefix": folder_prefix}
    if bucket_owner:
        paginate_kwargs["ExpectedBucketOwner"] = bucket_owner

    download_extra_args = {}
    if bucket_owner:
        download_extra_args["ExpectedBucketOwner"] = bucket_owner

    for page in paginator.paginate(**paginate_kwargs):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            rel_path = key[len(folder_prefix) :]
            if not rel_path:
                continue
            target = (dest / rel_path).resolve()
            if not str(target).startswith(str(dest.resolve())):
                print(f"Skipping potentially malicious path: {key}")
                continue
            parent_dir = target.parent
            if parent_dir not in created_dirs:
                parent_dir.mkdir(parents=True, exist_ok=True)
                created_dirs.add(parent_dir)
            try:
                client.download_file(
                    bucket, key, str(target), ExtraArgs=download_extra_args
                )
            except ClientError as e:
                print(f"Failed to download {key}: {e}")
    return dest


def _run_validation(root: Path) -> None:
    """Run validation on the downloaded folders and print a report."""
    try:
        report: ValidationReport = scan_and_refresh(root)
        print("Validation report:")
        print(f"  folders_scanned: {report.folders_scanned}")
        print(f"  manifests_created: {report.manifests_created}")
        print(f"  manifests_updated: {report.manifests_updated}")
        print(f"  problems: {len(report.problems)}")
        for prob in report.problems[:20]:
            print(f"   - {prob.folder}: {prob.issue}")
    except Exception as e:
        print(f"An error occurred during validation: {e}")


def main() -> None:
    """Main function to orchestrate the S3 folder sampling and validation."""
    try:
        env = _setup_environment()
        client = _create_s3_client(
            env["S3_ENDPOINT"],
            env["S3_REGION"],
            env["S3_ACCESS_KEY"],
            env["S3_SECRET_KEY"],
        )
        bucket_owner = env.get("S3_BUCKET_OWNER")

        folders = _list_s3_folders(
            client, env["S3_BUCKET_RAW"], "raw/outlook/", bucket_owner
        )
        print(f"Selected folders: {len(folders)}")

        with tempfile.TemporaryDirectory(prefix="conv_sample_") as temp_dir:
            root = Path(temp_dir)
            print(f"Downloading to {root}")
            local_folders = [
                _download_folder(client, env["S3_BUCKET_RAW"], f, root, bucket_owner)
                for f in folders
            ]
            print(f"Downloaded folders: {len(local_folders)}")
            for p in local_folders:
                print(
                    p.name,
                    "Conversation.txt exists?",
                    (p / "Conversation.txt").exists(),
                )
            _run_validation(root)
    except (ValueError, ClientError) as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
