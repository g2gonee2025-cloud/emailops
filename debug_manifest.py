import json

import boto3
from botocore.config import Config

S3_ENDPOINT = "https://sgp1.digitaloceanspaces.com"
S3_REGION = "sgp1"
S3_BUCKET = "emailops-storage-sgp1"
S3_ACCESS_KEY = "${S3_ACCESS_KEY}"
S3_SECRET_KEY = "${SE_SECRET_KEY}"

client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    region_name=S3_REGION,
    access_key=S3_ACCESS_KEY,
    secret_access_key=S3_SECRET_KEY,
    config=Config(signature_version="s3v4"),
)

key = "raw/outlook/EML-2024-10-30_2A173A - Lunch in the Canteen(November 4-8)/manifest.json"

try:
    response = client.get_object(Bucket=S3_BUCKET, Key=key)
    content = response["Body"].read()
    print(f"Content length: {len(content)}")

    try:
        decoded = content.decode("utf-8-sig")
        print("Decoded successfully")

        # Attempt fix
        fixed = decoded.replace(':"""', ':""')
        # Also handle potential spaces
        fixed = fixed.replace(': """', ': ""')

        parsed = json.loads(fixed)
        print("Parsed successfully after fix")
        # print(parsed)
    except Exception as e:
        print(f"Error parsing: {e}")

except Exception as e:
    print(f"Error fetching: {e}")
