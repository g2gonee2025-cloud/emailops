import logging

import boto3
from botocore.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_bucket():
    # Credentials from what I saw in .env
    access_key = "DO00XHY7KTQGELEBRGYV"
    secret_key = "UjuF90LBXLVdJFH6mMvxL9+mkE7peuyP5RL1oVYiyNs"
    region = "tor1"
    bucket = "emailops-storage-tor1"
    # Endpoint should likely be regional for boto3, not virtual-hosted
    endpoint = "https://tor1.digitaloceanspaces.com"

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
    )

    prefix = ""

    print(f"Listing contents of s3://{bucket}/{prefix} ...")

    try:
        response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=20)

        if "Contents" not in response:
            print("No objects found.")
            return

        print(f"Found {len(response['Contents'])} objects (showing top 20):")
        for obj in response["Contents"]:
            print(f" - {obj['Key']} ({obj['Size']} bytes)")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    inspect_bucket()
