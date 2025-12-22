import json
import shutil
import sys
from pathlib import Path

import boto3

REPO_ROOT = Path(__file__).resolve().parent
BACKEND_SRC = (REPO_ROOT / "backend" / "src").resolve()
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from cortex.ingestion.conv_manifest.validation import scan_and_refresh  # noqa: E402

# Configuration
LIMIT = 200
TEMP_DIR = Path("temp_s3_validation")
MANIFEST_FILENAME = "manifest.json"


def load_env_vars():
    """Load S3 credentials from .env manually."""
    env_vars = {}
    try:
        with Path(".env").open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    env_vars[key.strip()] = val.strip()
    except Exception as e:
        print(f"Error loading .env: {e}")
    return env_vars


def main():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True)

    env = load_env_vars()

    # Setup S3 client
    session = boto3.Session(
        aws_access_key_id=env.get("S3_ACCESS_KEY"),
        aws_secret_access_key=env.get("S3_SECRET_KEY"),
        region_name=env.get("S3_REGION", "nyc3"),
    )

    s3 = session.client(
        "s3", endpoint_url=env.get("S3_ENDPOINT", "https://nyc3.digitaloceanspaces.com")
    )
    bucket = env.get("S3_BUCKET_RAW", "emailops-bucket")

    print(f"Connecting to bucket: {bucket}")

    # List objects to find conversation folders
    # Assuming structure is flat or has a common prefix like "Outlook/"
    paginator = s3.get_paginator("list_objects_v2")

    found_folders = set()
    downloads_count = 0

    print("Scanning S3 for conversations...")

    # We look for "manifest.json" directly to identify valid conversation roots
    for page in paginator.paginate(Bucket=bucket):
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith(MANIFEST_FILENAME):
                folder_prefix = str(Path(key).parent)

                if folder_prefix in found_folders:
                    continue

                found_folders.add(folder_prefix)

                # Download manifest and Conversation.txt
                local_folder = TEMP_DIR / folder_prefix
                local_folder.mkdir(parents=True, exist_ok=True)
                (local_folder / "attachments").mkdir(
                    exist_ok=True
                )  # Fake attachments dir

                try:
                    # Download manifest
                    s3.download_file(bucket, key, str(local_folder / MANIFEST_FILENAME))

                    # Download conversation text (needed for valid refresh)
                    conv_key = f"{folder_prefix}/Conversation.txt"
                    try:
                        s3.download_file(
                            bucket, conv_key, str(local_folder / "Conversation.txt")
                        )
                    except Exception:
                        # Stub if missing (unlikely if valid)
                        (local_folder / "Conversation.txt").write_text("Stub content")

                    downloads_count += 1
                    if downloads_count % 10 == 0:
                        print(f"Downloaded {downloads_count} folders...", end="\r")

                except Exception as e:
                    print(f"\nError downloading {folder_prefix}: {e}")

                if downloads_count >= LIMIT:
                    break
        if downloads_count >= LIMIT:
            break

    print(f"\nDownloaded {downloads_count} validation candidates.")

    if downloads_count == 0:
        print("No manifests found in S3. Please check bucket/permissions.")
        return

    # Run Validation
    print("Running scan_and_refresh...")
    report = scan_and_refresh(TEMP_DIR)

    print("\n=== Validation Results ===")
    print(f"Manifests Updated: {report.manifests_updated}")
    print(f"Manifests Created: {report.manifests_created}")

    # Check Preservation
    print("\n=== Data Preservation Check ===")
    preserved = 0
    checked = 0

    # Iterate specifically over the folders we downloaded
    # (TEMP_DIR structure might be nested depending on prefix)
    for folder_path in TEMP_DIR.rglob(MANIFEST_FILENAME):
        checked += 1
        try:
            data = json.loads(folder_path.read_text())
            # Check for critical fields that naive validation deletes
            if "messages" not in data:
                print(f"FAILURE: {folder_path} lost 'messages' key!")
                continue

            messages = data["messages"]
            if not isinstance(messages, list):
                print(f"FAILURE: {folder_path} 'messages' is not a list!")
                continue

            # If we reached here, it's good
            preserved += 1
        except Exception as e:
            print(f"Error checking {folder_path}: {e}")

    print(f"Preserved 'messages' field: {preserved}/{checked}")

    if preserved == checked and checked > 0:
        print("\nSUCCESS: All manifests retained their rich data.")
    else:
        print("\nFAILURE: Data loss detected.")


if __name__ == "__main__":
    main()
