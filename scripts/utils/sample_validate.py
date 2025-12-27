import os
import tempfile
from pathlib import Path

import boto3
from botocore.client import Config
from cortex.ingestion.conv_manifest.validation import scan_and_refresh
from dotenv import load_dotenv


def main() -> None:
    load_dotenv(override=True)
    endpoint = os.getenv("S3_ENDPOINT")
    region = os.getenv("S3_REGION")
    bucket = os.getenv("S3_BUCKET_RAW")
    key = os.getenv("S3_ACCESS_KEY")
    secret = os.getenv("S3_SECRET_KEY")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
    )

    prefix = "raw/outlook/"
    paginator = client.get_paginator("list_objects_v2")
    folders: list[str] = []
    if not bucket:
        raise ValueError("S3 bucket is not specified; cannot paginate objects.")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        folders.extend(cp["Prefix"] for cp in page.get("CommonPrefixes", []))
        if len(folders) >= 10:
            break
    folders = folders[:10]
    print(f"Selected folders: {len(folders)}")

    root = Path(tempfile.mkdtemp(prefix="conv_sample_"))
    print(f"Downloading to {root}")

    def download_folder(folder_prefix: str) -> Path:
        dest = root / folder_prefix.rstrip("/").split("/")[-1]
        dest.mkdir(parents=True, exist_ok=True)
        for page in paginator.paginate(Bucket=bucket, Prefix=folder_prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(folder_prefix) :]
                if not rel:
                    continue
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    client.download_file(bucket, obj["Key"], str(target))
                except Exception as e:
                    # Do not abort the entire batch on a single object failure
                    print(f"Failed to download {obj['Key']}: {e}")
        return dest

    local_folders = [download_folder(f) for f in folders]
    print(f"Downloaded folders: {len(local_folders)}")
    for p in local_folders:
        print(p.name, "Conversation.txt exists?", (p / "Conversation.txt").exists())

    report = scan_and_refresh(root)
    print("Validation report:")
    print(f"  folders_scanned: {report.folders_scanned}")
    print(f"  manifests_created: {report.manifests_created}")
    print(f"  manifests_updated: {report.manifests_updated}")
    print(f"  problems: {len(report.problems)}")
    for prob in report.problems[:20]:
        print(f"   - {prob.folder}: {prob.issue}")


if __name__ == "__main__":
    main()
