import json
import os
import tempfile
from pathlib import Path

import boto3
from botocore.client import Config
from cortex.ingestion.conv_manifest.validation import scan_and_refresh
from dotenv import load_dotenv

BASE_PREFIX = "raw/outlook/"
MAX_FOLDERS = 20


def list_conversation_prefixes(
    client, bucket: str, base_prefix: str, limit: int
) -> list[str]:
    prefixes: list[str] = []
    seen: set[str] = set()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=base_prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key") or ""
            if not key.endswith("Conversation.txt"):
                continue
            folder_prefix = key.rsplit("/", 1)[0] + "/"
            if folder_prefix in seen:
                continue
            seen.add(folder_prefix)
            prefixes.append(folder_prefix)
            if len(prefixes) >= limit:
                return prefixes
    return prefixes


def download_folder(client, bucket: str, folder_prefix: str, dest_root: Path) -> Path:
    paginator = client.get_paginator("list_objects_v2")
    dest = dest_root / folder_prefix.rstrip("/").split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    for page in paginator.paginate(Bucket=bucket, Prefix=folder_prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key") or ""
            rel = key[len(folder_prefix) :]
            if not rel:
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
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


def summarize_manifest(text: str | None) -> dict[str, str]:
    if not text:
        return {"status": "missing_or_unreadable"}
    try:
        data = json.loads(text)
    except Exception:
        return {"status": "corrupt_json"}

    def _get(name: str) -> str:
        val = data.get(name)
        if val is None:
            return ""  # keep empty string for missing
        return str(val)

    return {
        "status": "ok",
        "subject_label": _get("subject_label"),
        "started_at_utc": _get("started_at_utc"),
        "ended_at_utc": _get("ended_at_utc"),
        "message_count": _get("message_count"),
        "attachment_count": _get("attachment_count"),
        "sha256_conversation": _get("sha256_conversation"),
        "conv_id": _get("conv_id"),
        "conv_key_type": _get("conv_key_type"),
    }


def main() -> None:
    load_dotenv(override=True)
    endpoint = os.getenv("S3_ENDPOINT")
    region = os.getenv("S3_REGION")
    bucket = os.getenv("S3_BUCKET_RAW")
    key = os.getenv("S3_ACCESS_KEY")
    secret = os.getenv("S3_SECRET_KEY")

    if not all([endpoint, region, bucket, key, secret]):
        raise SystemExit("Missing S3_* env vars")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
    )

    prefixes = list_conversation_prefixes(client, bucket, BASE_PREFIX, MAX_FOLDERS)
    if not prefixes:
        raise SystemExit("No conversation folders found under prefix")

    root = Path(tempfile.mkdtemp(prefix="manifest_batch_"))
    print(f"Temp root: {root}")
    before: dict[str, dict[str, str]] = {}

    for prefix in prefixes:
        conv_dir = download_folder(client, bucket, prefix, root)
        manifest_path = conv_dir / "manifest.json"
        before[prefix] = summarize_manifest(load_text(manifest_path))

    report = scan_and_refresh(root)
    after: dict[str, dict[str, str]] = {}
    for prefix in prefixes:
        conv_dir = root / prefix.rstrip("/").split("/")[-1]
        manifest_path = conv_dir / "manifest.json"
        after[prefix] = summarize_manifest(load_text(manifest_path))

    print(
        f"\nReport: folders_scanned={report.folders_scanned} "
        f"created={report.manifests_created} updated={report.manifests_updated} "
        f"problems={len(report.problems)}"
    )

    print("\n--- started_at_utc / ended_at_utc (before -> after) ---")
    for prefix in prefixes:
        b = before[prefix]
        a = after[prefix]
        print(f"\n{prefix}")
        print(
            "before:",
            b.get("started_at_utc"),
            b.get("ended_at_utc"),
        )
        print(
            "after: ",
            a.get("started_at_utc"),
            a.get("ended_at_utc"),
        )


if __name__ == "__main__":
    main()
