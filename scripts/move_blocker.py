import os

import boto3


def run():
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("OUTLOOKCORTEX_S3_ENDPOINT"),
        aws_access_key_id=os.getenv("OUTLOOKCORTEX_S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("OUTLOOKCORTEX_S3_SECRET_KEY"),
        region_name="tor1",
    )
    bucket = os.getenv("OUTLOOKCORTEX_S3_BUCKET_RAW", "emailops-storage-tor1")
    src = "Outlook/TML-2024-11-19_3CA775 - H&C - Imports & Exports - Actuals/attachments/FINAL OCTOBER 2023- SEPT 17 2024.xlsx"
    dest = "Outlook/skipped_files/FINAL OCTOBER 2023- SEPT 17 2024.xlsx"

    print(f"Moving {src} to {dest}...")
    try:
        s3.copy_object(
            Bucket=bucket, CopySource={"Bucket": bucket, "Key": src}, Key=dest
        )
        s3.delete_object(Bucket=bucket, Key=src)
        print("Done.")
    except Exception as e:
        print(f"Failed to move object: {e}")


if __name__ == "__main__":
    run()
