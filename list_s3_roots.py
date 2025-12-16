
import sys
import os
import boto3
from botocore.client import Config

# Add backend/src to path
sys.path.append(os.path.join(os.getcwd(), 'backend', 'src'))
from cortex.config.loader import get_config

def list_s3_roots():
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

        print(f"Listing root folders in bucket: {s3_config.bucket_raw}")
        resp = client.list_objects_v2(Bucket=s3_config.bucket_raw, Delimiter='/')

        print("\nprefixes:")
        for p in resp.get('CommonPrefixes', []):
            print(f"- {p['Prefix']}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_s3_roots()
