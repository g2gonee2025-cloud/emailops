import json
import sys
from pathlib import Path

import boto3
from botocore.config import Config

sys.path.append(str(Path("backend/src").resolve()))
from cortex.config.loader import get_config

config = get_config()

client = boto3.client(
    "s3",
    endpoint_url=config.storage.endpoint_url,
    region_name=config.storage.region,
    aws_access_key_id=config.storage.access_key,
    aws_secret_access_key=config.storage.secret_key,
    config=Config(signature_version="s3v4"),
)

key = "raw/outlook/EML-2024-10-30_2A173A - Lunch in the Canteen(November 4-8)/manifest.json"

try:
    response = client.get_object(Bucket=config.storage.bucket_raw, Key=key)
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
