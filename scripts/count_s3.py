import os

import boto3


def count() -> None:
    endpoint = os.getenv(
        "OUTLOOKCORTEX_S3_ENDPOINT", "https://tor1.digitaloceanspaces.com"
    )
    key = os.getenv("OUTLOOKCORTEX_S3_ACCESS_KEY")
    secret = os.getenv("OUTLOOKCORTEX_S3_SECRET_KEY")
    bucket = os.getenv("OUTLOOKCORTEX_S3_BUCKET")

    print(f"Connecting to {endpoint} bucket {bucket}...")
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        region_name="tor1",
    )

    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page_num, page in enumerate(
        paginator.paginate(Bucket=bucket, Prefix="Outlook/", Delimiter="/"), start=1
    ):
        subs = page.get("CommonPrefixes", [])
        c = len(subs)
        count += c
        PROGRESS_PAGE_INTERVAL = 5
        if page_num % PROGRESS_PAGE_INTERVAL == 0:
            print(f"Page {page_num}: found {c} folders (Total so far: {count})")

    print(f"FINAL COUNT: {count}")


if __name__ == "__main__":
    count()
