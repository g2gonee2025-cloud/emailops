import sys
from pathlib import Path

import boto3

sys.path.append(str(Path("backend/src").resolve()))
from botocore.config import Config
from cortex.config.loader import get_config


def check_s3():
    config = get_config()
    print("Connecting to S3...")
    print(f"Endpoint: {config.storage.endpoint_url}")
    print(f"Region: {config.storage.region}")
    print(f"Bucket: {config.storage.bucket_raw}")

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=config.storage.endpoint_url,
            region_name=config.storage.region,
            aws_access_key_id=config.storage.access_key,
            aws_secret_access_key=config.storage.secret_key,
            config=Config(signature_version="s3v4"),
        )

        # List first 1 object to verify access
        response = s3.list_objects_v2(Bucket=config.storage.bucket_raw, MaxKeys=1)
        print("Connectivity Check: SUCCESS")
        if "Contents" in response:
            print(f"Found object: {response['Contents'][0]['Key']}")
        else:
            print("Bucket is empty or empty prefix")

    except Exception as e:
        print(f"Connectivity Check: FAILED\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    check_s3()
