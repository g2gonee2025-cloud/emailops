import os
import tempfile
from pathlib import Path

import boto3
from botocore.client import Config
from cortex.ingestion.conv_manifest.validation import scan_and_refresh
from dotenv import load_dotenv

FOLDER_PREFIX = "raw/outlook/EML-2025-03-26_3A04C2 - Lunch in the Canteen(March 31-4)/"


def download_folder(client, bucket: str, folder_prefix: str, dest_root: Path) -> Path:
    paginator = client.get_paginator("list_objects_v2")
    dest = dest_root / folder_prefix.rstrip("/").split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    for page in paginator.paginate(Bucket=bucket, Prefix=folder_prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(folder_prefix) :]
            if not rel:
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, obj["Key"], str(target))
    return dest


def load_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8-sig")
    except Exception:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None


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

    root = Path(tempfile.mkdtemp(prefix="manifest_diff_"))
    print(f"Temp root: {root}")

    conv_dir = download_folder(client, bucket, FOLDER_PREFIX, root)
    manifest_path = conv_dir / "manifest.json"

    before = load_text(manifest_path)
    print("\n--- BEFORE manifest.json ---")
    if before is None:
        print("(missing or unreadable)")
    else:
        print(before)

    report = scan_and_refresh(root)
    after = load_text(manifest_path)

    print("\n--- AFTER manifest.json ---")
    if after is None:
        print("(missing or unreadable)")
    else:
        print(after)

    print(
        f"\nValidation summary: folders_scanned={report.folders_scanned} "
        f"created={report.manifests_created} updated={report.manifests_updated} "
        f"problems={len(report.problems)}"
    )


if __name__ == "__main__":
    main()
