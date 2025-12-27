import os

import boto3


def run():
    """
    Entry point for the list_blocker script.

    Creates a boto3 S3 client using the OUTLOOKCORTEX_S3_* environment variables.
    The function encapsulates the script's main logic, which may perform network
    I/O against S3 and produce console output as side effects.

    Parameters:
        None

    Returns:
        None
    """
    endpoint = os.getenv("OUTLOOKCORTEX_S3_ENDPOINT")
    access_key = os.getenv("OUTLOOKCORTEX_S3_ACCESS_KEY")
    secret_key = os.getenv("OUTLOOKCORTEX_S3_SECRET_KEY")
    missing = [
        name
        for name, val in [
            ("OUTLOOKCORTEX_S3_ENDPOINT", endpoint),
            ("OUTLOOKCORTEX_S3_ACCESS_KEY", access_key),
            ("OUTLOOKCORTEX_S3_SECRET_KEY", secret_key),
        ]
        if not val
    ]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("OUTLOOKCORTEX_S3_ENDPOINT"),
        aws_access_key_id=os.getenv("OUTLOOKCORTEX_S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("OUTLOOKCORTEX_S3_SECRET_KEY", ""),
        region_name="tor1",
    )
    prefix = "Outlook/TML-2024-11-19_3CA775 - H&C - Imports & Exports - Actuals/"
    print(f"Listing {prefix}...")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket="emailops-storage-tor1", Prefix=prefix):
        for c in page.get("Contents", []):
            print(f"{c['Key']} ({c['Size']} bytes)")


if __name__ == "__main__":
    run()
