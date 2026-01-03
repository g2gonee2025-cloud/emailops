import json
import os
import shutil
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_SRC = (REPO_ROOT / "backend" / "src").resolve()
# STYLE: The following sys.path manipulation is a workaround to allow this standalone
# script to import modules from the main 'cortex' project. This is necessary because
# the script is not part of an installed package. In a production environment, it's
# preferable to have a proper package structure, but for a verification script,
# this approach is a pragmatic compromise.
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
    required_vars = ["S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_BUCKET_RAW"]
    if any(not os.getenv(var) for var in required_vars):
        missing = [var for var in required_vars if not os.getenv(var)]
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    try:
        session = boto3.Session(
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
            region_name=os.getenv("S3_REGION", "nyc3"),
        )
        s3 = session.client(
            "s3",
            endpoint_url=os.getenv(
                "S3_ENDPOINT", "https://nyc3.digitaloceanspaces.com"
            ),
        )
        bucket = os.getenv("S3_BUCKET_RAW")
        s3.head_bucket(Bucket=bucket)  # Verify connection and bucket access
        print(f"Connecting to bucket: {bucket}")
        return s3, bucket
    except (ClientError, NoCredentialsError) as e:
        print(f"S3 connection failed: {e}. Check credentials and bucket name.")
        sys.exit(1)


def discover_s3_folders(s3, bucket):
    """Discover conversation folders in the S3 bucket."""
    found_folders = set()
    print("Scanning S3 for conversations...")
    paginator = s3.get_paginator("list_objects_v2")
    try:
        pages = paginator.paginate(Bucket=bucket, Prefix=S3_PREFIX, Delimiter="/")
        for page in pages:
            # Logic Error Fix: Discover manifests in subfolders (CommonPrefixes)
            for prefix_info in page.get("CommonPrefixes", []):
                folder_prefix = prefix_info["Prefix"]
                if folder_prefix.endswith("/"):
                    found_folders.add(folder_prefix[:-1])
                if len(found_folders) >= LIMIT:
                    break
            if len(found_folders) >= LIMIT:
                break
            # Logic Error Fix: Discover manifests at the root of the prefix
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(MANIFEST_FILENAME):
                    folder = Path(key).parent
                    if str(folder) != ".":
                        found_folders.add(str(folder))
                if len(found_folders) >= LIMIT:
                    break
            if len(found_folders) >= LIMIT:
                break
    except ClientError as e:
        print(f"\nS3 discovery failed: {e}")
        return set()  # Return empty set to halt gracefully
    return found_folders


def download_manifests(s3, bucket, found_folders):
    """Download manifest and conversation files securely."""
    downloads_count = 0
    sorted_folders = sorted(list(found_folders))  # Deterministic order
    temp_dir_abs = os.path.abspath(TEMP_DIR)

    for folder_prefix in sorted_folders:
        if downloads_count >= LIMIT:
            print(f"\nReached download limit of {LIMIT}.")
            break

        # SECURITY: Prevent path traversal.
        # Construct the full path and ensure it's within the temp directory.
        local_folder_path = os.path.abspath(os.path.join(TEMP_DIR, folder_prefix))
        if not local_folder_path.startswith(temp_dir_abs):
            print(
                f"\nSKIPPING insecure S3 prefix (path traversal attempt): {folder_prefix}"
            )
            continue

        local_folder = Path(local_folder_path)
        local_folder.mkdir(parents=True, exist_ok=True)
        # PERFORMANCE: No longer creating 'attachments' subdir.

        try:
            # Download manifest
            manifest_key = f"{folder_prefix}/{MANIFEST_FILENAME}"
            manifest_dest = local_folder / MANIFEST_FILENAME
            s3.download_file(bucket, manifest_key, str(manifest_dest))

            # Download conversation file (if it exists)
            conv_key = f"{folder_prefix}/Conversation.txt"
            conv_dest = local_folder / "Conversation.txt"
            try:
                s3.download_file(bucket, conv_key, str(conv_dest))
            except ClientError as e:
                # If Conversation.txt is missing (404), that's acceptable.
                if e.response["Error"]["Code"] == "404":
                    conv_dest.write_text("Stub content (original not found)")
                else:
                    # Any other S3 error during conversation download is a real problem.
                    raise

            downloads_count += 1
            if downloads_count % 10 == 0:
                print(f"Downloaded {downloads_count} folders...", end="\r")

        except ClientError as e:
            # EXCEPTION_HANDLING: Specific error for manifest download.
            # If the manifest download fails, it's a critical error for this folder.
            print(
                f"\nError downloading for '{folder_prefix}': {e.response['Error']['Message']}"
            )
            # Clean up the directory we created for this failed download.
            shutil.rmtree(local_folder)
        except Exception as e:
            # Fallback for other unexpected errors during the process.
            print(f"\nAn unexpected error occurred for '{folder_prefix}': {e}")
            shutil.rmtree(local_folder)

    return downloads_count


def run_validation():
    """Run the scan_and_refresh validation."""
    print("\nRunning scan_and_refresh...")
    try:
        report = scan_and_refresh(TEMP_DIR)
        print("\n=== Validation Results ===")
        # TYPE_ERROR Fix: Check if the report object and its attributes exist
        if hasattr(report, "manifests_updated") and hasattr(
            report, "manifests_created"
        ):
            print(f"Manifests Updated: {report.manifests_updated}")
            print(f"Manifests Created: {report.manifests_created}")
        else:
            print(
                "Validation report is not in the expected format. Full report object:"
            )
            print(report)
    except Exception as e:
        # Catch errors from the validation logic itself
        print(f"\nAn error occurred during 'scan_and_refresh': {e}")


def check_data_preservation():
    """Check for data loss in the manifests."""
    print("\n=== Data Preservation Check ===")
    preserved = 0
    checked = 0
    for folder_path in TEMP_DIR.rglob(f"**/{MANIFEST_FILENAME}"):
        checked += 1
        try:
            data = json.loads(folder_path.read_text(encoding="utf-8"))
            if "messages" not in data:
                print(f"FAILURE: {folder_path.relative_to(TEMP_DIR)} lost 'messages' key!")
                continue
            messages = data["messages"]
            if not isinstance(messages, list):
                print(
                    f"FAILURE: {folder_path.relative_to(TEMP_DIR)} 'messages' is not a list!"
                )
                continue
            preserved += 1
        except json.JSONDecodeError as e:
            # EXCEPTION_HANDLING: Specific error for malformed JSON
            print(
                f"ERROR decoding JSON for {folder_path.relative_to(TEMP_DIR)}: {e}"
            )
        except IOError as e:
            # EXCEPTION_HANDLING: Specific error for file read issues
            print(f"ERROR reading file {folder_path.relative_to(TEMP_DIR)}: {e}")
        except Exception as e:
            # Fallback for other unexpected errors.
            print(
                f"An unexpected error occurred checking {folder_path.relative_to(TEMP_DIR)}: {e}"
            )

    print(f"\nPreserved 'messages' field in {preserved}/{checked} manifests.")
    if preserved == checked and checked > 0:
        print("SUCCESS: All manifests retained their rich data.")
    else:
        print("FAILURE: Data loss detected in one or more manifests.")


def main():
    """Run the S3 manifest verification process."""
    # Load environment variables from .env file first
    dotenv_path = REPO_ROOT / ".env"
    if not dotenv_path.exists():
        print(f"ERROR: .env file not found at {dotenv_path}")
        print("Please create one with S3 credentials (see README).")
        sys.exit(1)
    load_dotenv(dotenv_path=dotenv_path)

    # EXCEPTION_HANDLING: Ensure cleanup happens even if the script fails.
    try:
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
        TEMP_DIR.mkdir(parents=True)

        s3, bucket = setup_s3_client()

        found_folders = discover_s3_folders(s3, bucket)
        downloads_count = download_manifests(s3, bucket, found_folders)

        print(f"\n\nDownloaded {downloads_count} validation candidates.")

        if downloads_count == 0:
            print("No manifests found in S3. Please check bucket/permissions.")
            return

        run_validation()
        check_data_preservation()

    except Exception as e:
        print(f"\nAn unhandled error occurred in main execution: {e}")
        sys.exit(1)
    finally:
        # EXCEPTION_HANDLING: Robust cleanup of the temp directory.
        if TEMP_DIR.exists():
            try:
                shutil.rmtree(TEMP_DIR)
                print(f"\nCleaned up temporary directory: {TEMP_DIR}")
            except OSError as e:
                print(f"Error removing temporary directory {TEMP_DIR}: {e}")


if __name__ == "__main__":
    main()
