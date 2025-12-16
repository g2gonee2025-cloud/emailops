
import sys
import os
import shutil
import boto3
from pathlib import Path
from botocore.client import Config
import json

# Add backend/src to path
sys.path.append(os.path.join(os.getcwd(), 'backend', 'src'))
from cortex.config.loader import get_config
from cortex.ingestion.conv_manifest.validation import scan_and_refresh

def run_test():
    # 1. Setup Temp Dir (using gitignored name)
    temp_dir = Path("temp_validation_s3_20")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    # Fake "artifacts" sibling dir needed by validation logic
    artifacts_root = Path("artifacts")
    if artifacts_root.exists():
        shutil.rmtree(artifacts_root)
    artifacts_root.mkdir()

    print(f"Created temp dir: {temp_dir.absolute()}")

    # 2. Download 20 Folders from S3
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

        print("Listing S3 folders...")
        resp = client.list_objects_v2(Bucket=s3_config.bucket_raw, Prefix="Outlook/", Delimiter="/")
        prefixes = [p['Prefix'] for p in resp.get('CommonPrefixes', [])]

        target_prefixes = prefixes[:20]
        print(f"Found {len(prefixes)} folders, selecting first {len(target_prefixes)}...")

        for prefix in target_prefixes:
            folder_name = prefix.rstrip('/').split('/')[-1]
            local_folder = temp_dir / folder_name
            local_folder.mkdir()

            sub_resp = client.list_objects_v2(Bucket=s3_config.bucket_raw, Prefix=prefix)
            if 'Contents' not in sub_resp:
                continue

            for obj in sub_resp['Contents']:
                key = obj['Key']
                filename = key.split('/')[-1]
                if filename in ["Conversation.txt", "manifest.json"]:
                    client.download_file(s3_config.bucket_raw, key, str(local_folder / filename))

        print("Download complete. Running validation...")

        # 3. Run Validation
        report = scan_and_refresh(temp_dir)

        print("\n--- Report ---")
        print(f"Folders Scanned: {report.folders_scanned}")
        print(f"Manifests Updated: {report.manifests_updated}")
        print(f"Manifests Created: {report.manifests_created}")

        # 4. Inspect Results
        print("\n--- Inspection of First Updated Manifest ---")
        for folder in temp_dir.iterdir():
            manifest_path = folder / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    data = json.load(f)
                    print(f"Folder: {data.get('folder')}")
                    print(f"Participants: {json.dumps(data.get('participants'), indent=2)}")
                break

    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    run_test()
