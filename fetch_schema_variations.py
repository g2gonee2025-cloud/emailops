
import sys
import os
import json
import boto3
from botocore.client import Config
import random

# Add backend/src to path
sys.path.append(os.path.join(os.getcwd(), 'backend', 'src'))
from cortex.config.loader import get_config

def fetch_random_manifest():
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

        paginator = client.get_paginator('list_objects_v2')
        # We'll try to get a list frame from somewhere later in the bucket
        # "Outlook/EML-2025" might not exist yet if data is old.
        # Let's just scan linearly but print ONE example of EACH version we find.

        page_iterator = paginator.paginate(Bucket=s3_config.bucket_raw)

        print("Scanning for schema variations...")

        seen_versions = set()

        count = 0
        for page in page_iterator:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('manifest.json'):
                    count += 1
                    try:
                        resp = client.get_object(Bucket=s3_config.bucket_raw, Key=key)
                        data = json.loads(resp['Body'].read().decode('utf-8'))

                        ver = data.get('manifest_version', 'unknown')
                        has_msgs = 'messages' in data

                        sig = f"ver={ver}|has_msgs={has_msgs}"

                        if sig not in seen_versions:
                            print(f"\n✅ FOUND NEW SCHEMA VARIATION: {sig}")
                            print(f"Key: {key}")
                            print("=" * 60)
                            print(json.dumps(data, indent=2))
                            print("=" * 60)
                            seen_versions.add(sig)

                        if len(seen_versions) > 2:
                            return

                    except:
                        pass

            if count > 300:
                break

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_random_manifest()
