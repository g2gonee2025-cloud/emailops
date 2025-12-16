
import sys
import os
import json
import boto3
from botocore.client import Config

# Add backend/src to path
sys.path.append(os.path.join(os.getcwd(), 'backend', 'src'))
from cortex.config.loader import get_config

def fetch_real_manifest():
    try:
        config = get_config()
        s3_config = config.storage

        print(f"Connecting to S3: {s3_config.endpoint_url}")

        session = boto3.session.Session()
        client = session.client('s3',
            region_name=s3_config.region,
            endpoint_url=s3_config.endpoint_url,
            aws_access_key_id=s3_config.access_key,
            aws_secret_access_key=s3_config.secret_key,
            config=Config(signature_version='s3v4')
        )

        # Use MaxKeys to avoid hanging
        paginator = client.get_paginator('list_objects_v2')
        # We just want a few chunks to find ONE example
        page_iterator = paginator.paginate(Bucket=s3_config.bucket_raw, MaxKeys=50)

        print("Scanning S3 for a REAL multi-message manifest...")

        scanned = 0
        for page in page_iterator:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('manifest.json'):
                    scanned += 1
                    try:
                        # Fetch the object
                        resp = client.get_object(Bucket=s3_config.bucket_raw, Key=key)
                        data = json.loads(resp['Body'].read().decode('utf-8'))

                        msgs = data.get('messages', [])
                        if isinstance(msgs, list) and len(msgs) > 1:
                            print(f"\n✅ FOUND REAL DATA: {key}")
                            print("=" * 60)
                            print(json.dumps(data, indent=2))
                            print("=" * 60)
                            return
                        else:
                            sys.stdout.write(".")
                            sys.stdout.flush()
                    except:
                        pass

            # Safety break after 100 checked files to not run forever
            if scanned > 100:
                print("\nchecked 100 manifests, no multi-message threads found yet.")
                print("Showing the last single-message one as a consolation real example:")
                # print last one found? (omitted for brevity)
                return

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_real_manifest()
