"""
Verification script to download a sample of conversation folders from S3,
run validation to generate participant information, and print a sample manifest.

This script is intended to be run as a module from the repository root:
python -m scripts.verification.verify_participants --output-dir ./temp_verification --limit 10
"""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, BotoCoreError

# Ensure the project root is on the path to allow execution from any directory.
try:
    from cortex.config.loader import get_config
    from cortex.ingestion.conv_manifest.validation import scan_and_refresh
except ImportError:
    # If cortex is not in the path, assume the script is run from the repo root
    # and add the source directory to the path.
    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / "backend" / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.append(str(src_path))
    from cortex.config.loader import get_config
    from cortex.ingestion.conv_manifest.validation import scan_and_refresh


def setup_logging():
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify participant manifest generation for S3 conversation data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="The directory to use for temporary downloads and artifacts.",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="Outlook/",
        help="The S3 prefix to scan for conversation folders.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="The maximum number of conversation folders to download.",
    )
    return parser.parse_args()


def create_s3_client(s3_config):
    """
    Create and configure the S3 client, validating config first.

    Raises:
        ValueError: If required S3 configuration is missing.
    """
    required_keys = ["region", "endpoint_url", "access_key", "secret_key"]
    for key in required_keys:
        if not getattr(s3_config, key, None):
            raise ValueError(f"Missing required S3 configuration: storage.{key}")

    session = boto3.session.Session()
    return session.client(
        "s3",
        region_name=s3_config.region,
        endpoint_url=s3_config.endpoint_url,
        aws_access_key_id=s3_config.access_key,
        aws_secret_access_key=s3_config.secret_key,
        config=Config(signature_version="s3v4"),
    )


def setup_directories(output_dir: Path):
    """
    Safely create temporary directories for downloads and artifacts.

    Raises:
        ValueError: If the output directory is not within the project root.
        IOError: If directory creation fails.
    """
    project_root = Path(__file__).resolve().parents[2]
    try:
        output_dir.resolve().relative_to(project_root.resolve())
    except ValueError:
        raise ValueError(
            "For safety, the output directory must be within the project repository."
        )

    temp_dir = output_dir / "downloads"
    artifacts_root = output_dir / "artifacts"

    try:
        # Cleanup previous runs
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temp_dir.mkdir(parents=True, exist_ok=False)
        artifacts_root.mkdir(parents=True, exist_ok=False)
    except (IOError, OSError) as e:
        logging.error(f"Failed to create directories: {e}", exc_info=True)
        raise

    logging.info(f"Created temporary directory: {temp_dir.resolve()}")
    return temp_dir, artifacts_root


def download_folders(s3_client, bucket, prefix, limit, download_dir):
    """
    Download specified conversation files from S3 using paginators.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    folder_pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/")

    target_prefixes = []
    try:
        for page in folder_pages:
            for p in page.get("CommonPrefixes", []):
                if len(target_prefixes) >= limit:
                    break
                target_prefixes.append(p["Prefix"])
            if len(target_prefixes) >= limit:
                break
    except ClientError as e:
        logging.error(f"Failed to list S3 prefixes: {e}", exc_info=True)
        return

    logging.info(
        f"Found {len(target_prefixes)} folders, downloading first {limit}..."
    )

    for p_prefix in target_prefixes:
        folder_name = p_prefix.rstrip("/").split("/")[-1]
        local_folder = download_dir / folder_name
        local_folder.mkdir(exist_ok=True)  # exist_ok=True handles reruns gracefully

        try:
            object_pages = paginator.paginate(Bucket=bucket, Prefix=p_prefix)
            for page in object_pages:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    filename = Path(key).name
                    if filename in ["Conversation.txt", "manifest.json"]:
                        dest_path = local_folder / filename
                        logging.info(f"Downloading {key} to {dest_path}")
                        s3_client.download_file(bucket, key, str(dest_path))
        except ClientError as e:
            logging.warning(
                f"Could not download objects from {p_prefix}: {e}", exc_info=True
            )
            continue  # Continue to the next folder

    logging.info("Download phase complete.")


def run_and_report_validation(temp_dir: Path):
    """Run the validation logic and print a summary report."""
    logging.info("Running validation...")
    report = scan_and_refresh(temp_dir)

    print("\n--- Report ---")
    print(f"Folders Scanned: {report.folders_scanned}")
    print(f"Manifests Updated: {report.manifests_updated}")
    print(f"Manifests Created: {report.manifests_created}")

    print("\n--- Inspection of First Updated Manifest ---")
    for folder in temp_dir.iterdir():
        if not folder.is_dir():
            continue
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open() as f:
                data = json.load(f)
                print(f"Folder: {data.get('folder')}")
                participants = json.dumps(data.get("participants"), indent=2)
                print(f"Participants: {participants}")
            break


def main():
    """Main execution flow."""
    setup_logging()
    args = parse_args()
    output_dir = args.output_dir

    try:
        temp_dir, _ = setup_directories(output_dir)
        config = get_config()
        s3_client = create_s3_client(config.storage)

        download_folders(
            s3_client,
            config.storage.bucket_raw,
            args.s3_prefix,
            args.limit,
            temp_dir,
        )

        run_and_report_validation(temp_dir)

    except (ValueError, BotoCoreError) as e:
        logging.error(f"Configuration or operational error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if output_dir.exists():
            logging.info(f"Cleaning up temporary directory: {output_dir.resolve()}")
            try:
                shutil.rmtree(output_dir)
            except (IOError, OSError) as e:
                logging.error(f"Failed to clean up directory {output_dir}: {e}")
                sys.exit(1)


if __name__ == "__main__":
    main()
