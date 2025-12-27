import json
import os
import shutil
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_SRC = (REPO_ROOT / "backend" / "src").resolve()
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from cortex.ingestion.conv_manifest.validation import scan_and_refresh

# Configuration
LIMIT = 200
S3_PREFIX = "Outlook/"  # Set to "" to scan the entire bucket
TEMP_DIR = Path("temp_s3_validation")
MANIFEST_FILENAME = "manifest.json"


def setup_s3_client():
    """Initialize and return an S3 client and bucket name."""
    session = boto3.Session(
        aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
        region_name=os.getenv("S3_REGION", "nyc3"),
    )
    s3 = session.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT", "https://nyc3.digitaloceanspaces.com"),
    )
    bucket = os.getenv("S3_BUCKET_RAW", "emailops-bucket")
    print(f"Connecting to bucket: {bucket}")
    return s3, bucket


def discover_s3_folders(s3, bucket):
    """Discover conversation folders in the S3 bucket."""
    found_folders = set()
    print("Scanning S3 for conversations...")
    continuation_token = None
    while True:
        args = {"Bucket": bucket, "Prefix": S3_PREFIX, "Delimiter": "/"}
        if continuation_token:
            args["ContinuationToken"] = continuation_token
        response = s3.list_objects_v2(**args)
        if "CommonPrefixes" not in response:
            break
        for prefix_info in response.get("CommonPrefixes", []):
            folder_prefix = prefix_info["Prefix"]
            if folder_prefix.endswith("/"):
                found_folders.add(folder_prefix[:-1])
        if not response.get("IsTruncated"):
            break
        continuation_token = response.get("NextContinuationToken")
    return found_folders


def download_manifests(s3, bucket, found_folders):
    """Download manifest and conversation files."""
    downloads_count = 0
    for folder_prefix in found_folders:
        if downloads_count >= LIMIT:
            break
        local_folder = TEMP_DIR / folder_prefix
        local_folder.mkdir(parents=True, exist_ok=True)
        (local_folder / "attachments").mkdir(exist_ok=True)
        try:
            manifest_key = f"{folder_prefix}/{MANIFEST_FILENAME}"
            s3.download_file(
                bucket, manifest_key, str(local_folder / MANIFEST_FILENAME)
            )
            conv_key = f"{folder_prefix}/Conversation.txt"
            try:
                s3.download_file(
                    bucket, conv_key, str(local_folder / "Conversation.txt")
                )
            except Exception:
                (local_folder / "Conversation.txt").write_text("Stub content")
            downloads_count += 1
            if downloads_count % 10 == 0:
                print(f"Downloaded {downloads_count} folders...", end="\r")
        except Exception as e:
            print(f"\nError downloading {folder_prefix}: {e}")
    return downloads_count


def run_validation():
    """Run the scan_and_refresh validation."""
    print("Running scan_and_refresh...")
    report = scan_and_refresh(TEMP_DIR)
    print("\n=== Validation Results ===")
    print(f"Manifests Updated: {report.manifests_updated}")
    print(f"Manifests Created: {report.manifests_created}")


def check_data_preservation():
    """Check for data loss in the manifests."""
    print("\n=== Data Preservation Check ===")
    preserved = 0
    checked = 0
    for folder_path in TEMP_DIR.rglob(MANIFEST_FILENAME):
        checked += 1
        try:
            data = json.loads(folder_path.read_text())
            if "messages" not in data:
                print(f"FAILURE: {folder_path} lost 'messages' key!")
                continue
            messages = data["messages"]
            if not isinstance(messages, list):
                print(f"FAILURE: {folder_path} 'messages' is not a list!")
                continue
            preserved += 1
        except Exception as e:
            print(f"Error checking {folder_path}: {e}")
    print(f"Preserved 'messages' field: {preserved}/{checked}")
    if preserved == checked and checked > 0:
        print("\nSUCCESS: All manifests retained their rich data.")
    else:
        print("\nFAILURE: Data loss detected.")


def main():
    """Run the S3 manifest verification process."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True)

    # Load environment variables from .env file
    dotenv_path = REPO_ROOT / ".env"
    if not dotenv_path.exists():
        print(f"ERROR: .env file not found at {dotenv_path}")
        print("Please create one with S3 credentials (see README).")
        sys.exit(1)
    load_dotenv(dotenv_path=dotenv_path)

    s3, bucket = setup_s3_client()

    found_folders = discover_s3_folders(s3, bucket)
    downloads_count = download_manifests(s3, bucket, found_folders)

    print(f"\nDownloaded {downloads_count} validation candidates.")

    if downloads_count == 0:
        print("No manifests found in S3. Please check bucket/permissions.")
        return

    run_validation()
    check_data_preservation()


if __name__ == "__main__":
    main()
