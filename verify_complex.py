
import sys
import os
import shutil
import boto3
from pathlib import Path
from botocore.client import Config
import json
import re

# Add backend/src to path
sys.path.append(os.path.join(os.getcwd(), 'backend', 'src'))
from cortex.config.loader import get_config
from cortex.ingestion.conv_manifest.validation import scan_and_refresh

def run_test():
    # 1. Setup Temp Dir
    temp_dir = Path("temp_complex_validation")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    artifacts_root = Path("artifacts")
    if artifacts_root.exists(): # Cleanup artifacts too
        shutil.rmtree(artifacts_root)
    artifacts_root.mkdir()

    print(f"Created temp dir: {temp_dir.absolute()}")

    # 2. Search for Complex Folders
    try:
        config = get_config()
        s3_config = config.storage

        session = boto3.session.Session()
        client = session.client('s3',
            region_name=s3_config.region,
            endpoint_url=s3_config.endpoint_url,
            aws_access_key_id=s3_config.access_key,
            aws_secret_access_key=s3_config.secret_key,
            config=Config(signature_version='s3v4')
        )

        print("Scanning S3 for complex conversations (multiple participants)...")
        # List "Outlook/" prefixes
        paginator = client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=s3_config.bucket_raw, Prefix="Outlook/", Delimiter="/")

        candidates_found = 0
        folders_scanned = 0

        for page in page_iterator:
            prefixes = [p['Prefix'] for p in page.get('CommonPrefixes', [])]

            for prefix in prefixes:
                folders_scanned += 1
                if folders_scanned > 200: # Scan limit
                    break

                # Detailed check of this folder
                sub_resp = client.list_objects_v2(Bucket=s3_config.bucket_raw, Prefix=prefix)
                content_map = {obj['Key'].split('/')[-1]: obj for obj in sub_resp.get('Contents', [])}

                if "Conversation.txt" in content_map:
                    # Heuristic: Check size. If > 5KB, might be interesting.
                    # Or simply download and regex check "From:" count.
                    # We'll download and check to be sure.

                    obj_key = prefix + "Conversation.txt"

                    # Read first 10KB to check for multiple Froms
                    try:
                        # Range get to save bw? No, just get it.
                        resp = client.get_object(Bucket=s3_config.bucket_raw, Key=obj_key)
                        text = resp['Body'].read().decode('utf-8', errors='ignore')

                        from_count = len(re.findall(r"From:", text, re.IGNORECASE))

                        if from_count > 1:
                            print(f"\nFound COMPLEX Candidate: {prefix}")
                            print(f"  - Size: {len(text)} chars")
                            print(f"  - 'From:' occurrences: {from_count}")

                            # Save it for validation
                            folder_name = prefix.rstrip('/').split('/')[-1]
                            local_folder = temp_dir / folder_name
                            local_folder.mkdir()

                            # Save Conversation.txt
                            (local_folder / "Conversation.txt").write_text(text, encoding='utf-8')

                            # Download manifest too
                            if "manifest.json" in content_map:
                                client.download_file(s3_config.bucket_raw, prefix + "manifest.json", str(local_folder / "manifest.json"))

                            candidates_found += 1
                            if candidates_found >= 3: # Just need a few good examples
                                break
                    except Exception as e:
                        print(f"Skip {prefix}: {e}")

            if candidates_found >= 3 or folders_scanned > 200:
                break

        if candidates_found == 0:
            print("No complex conversations found in search range.")
            return

        print(f"\nDownloading complete ({candidates_found} folders). Running validation...")

        # 3. Run Validation
        report = scan_and_refresh(temp_dir)

        print("\n--- Validation Report ---")
        print(f"Folders Scanned: {report.folders_scanned}")
        print(f"Manifests Updated: {report.manifests_updated}")

        # 4. Inspect Results
        print("\n--- Inspection of Results ---")
        for folder in temp_dir.iterdir():
            manifest_path = folder / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    data = json.load(f)
                    print(f"\nFolder: {data.get('folder')}")
                    print("Participants Found:")
                    print(json.dumps(data.get('participants'), indent=2))

                    # Verify against expectation
                    if len(data.get('participants', [])) > 1:
                         print("✅ SUCCESS: Detected multiple participants.")
                    else:
                         print("⚠️ NOTE: Only 1 participant detected.")

    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    run_test()
