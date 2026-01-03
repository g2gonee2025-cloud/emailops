import os
import sys

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

DEFAULT_ENDPOINT = "https://tor1.digitaloceanspaces.com"
DEFAULT_REGION = "tor1"
S3_PREFIX = "Outlook/"
S3_DELIMITER = "/"
PROGRESS_PAGE_INTERVAL = 5


def main() -> None:
    endpoint = os.getenv("OUTLOOKCORTEX_S3_ENDPOINT", DEFAULT_ENDPOINT)
    key = os.getenv("OUTLOOKCORTEX_S3_ACCESS_KEY")
    secret = os.getenv("OUTLOOKCORTEX_S3_SECRET_KEY")
    bucket = os.getenv("OUTLOOKCORTEX_S3_BUCKET")

    if not all([key, secret, bucket]):
        print(
            "Error: Missing required environment variables. "
            "Please set OUTLOOKCORTEX_S3_ACCESS_KEY, OUTLOOKCORTEX_S3_SECRET_KEY, and OUTLOOKCORTEX_S3_BUCKET."
        )
        return

    print(f"Connecting to {endpoint} bucket {bucket}...")
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            region_name=DEFAULT_REGION,
        )
        paginator = s3.get_paginator("list_objects_v2")
        folder_count = 0
        for page_num, page in enumerate(
            paginator.paginate(Bucket=bucket, Prefix=S3_PREFIX, Delimiter=S3_DELIMITER),
            start=1,
        ):
            common_prefixes = page.get("CommonPrefixes", [])
            prefix_count = len(common_prefixes)
            folder_count += prefix_count
            if page_num % PROGRESS_PAGE_INTERVAL == 0:
                print(
                    f"Page {page_num}: found {prefix_count} folders (Total so far: {folder_count})"
                )
        print(f"FINAL COUNT: {folder_count}")
    except (ClientError, NoCredentialsError) as e:
        print(f"Error: Could not connect to S3. {e}", file=sys.stderr)
        return


if __name__ == "__main__":
    main()
