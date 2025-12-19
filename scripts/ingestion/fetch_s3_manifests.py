import json
import sys
from pathlib import Path

import boto3
from botocore.client import Config

# Add backend/src to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "backend" / "src"))
from cortex.config.loader import get_config


def fetch_multi_message_manifests():
    try:
        config = get_config()
        s3_config = config.storage

        print(f"Connecting to S3: {s3_config.endpoint_url}")
        print(f"Bucket: {s3_config.bucket_raw}")

        session = boto3.session.Session()
        client = session.client(
            "s3",
            region_name=s3_config.region,
            endpoint_url=s3_config.endpoint_url,
            aws_access_key_id=s3_config.access_key,
            aws_secret_access_key=s3_config.secret_key,
            config=Config(
                signature_version="s3v4"
            ),  # DO usage often needs this or standard
        )

        paginator = client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=s3_config.bucket_raw)

        found_count = 0
        target = 2

        print("\nScanning for multi-message manifests...")

        for page in page_iterator:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith("manifest.json"):
                    # Download and inspect
                    try:
                        response = client.get_object(
                            Bucket=s3_config.bucket_raw, Key=key
                        )
                        content = response["Body"].read().decode("utf-8")
                        manifest = json.loads(content)

                        msgs = manifest.get("messages", [])
                        if isinstance(msgs, list) and len(msgs) > 1:
                            print(f"\n✅ Found Multi-Message Thread: {key}")
                            print(f"Subject: {manifest.get('subject')}")
                            print(f"Message Count: {len(msgs)}")
                            print("-" * 40)
                            print(json.dumps(manifest, indent=2))
                            print("=" * 40)

                            found_count += 1
                            if found_count >= target:
                                return
                    except Exception as e:
                        print(f"Error reading {key}: {e}")
                        continue

        if found_count == 0:
            print("No multi-message manifests found.")

    except Exception as e:
        print(f"Fatal Error: {e}")


if __name__ == "__main__":
    fetch_multi_message_manifests()
