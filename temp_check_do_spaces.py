from __future__ import annotations

import json
from itertools import islice

from cortex.config.loader import get_config
from cortex.ingestion.s3_source import S3SourceHandler


def main() -> None:
    config = get_config()
    handler = S3SourceHandler()

    print("DigitalOcean Spaces configuration:")
    print(f"  Endpoint: {config.s3.endpoint_url}")
    print(f"  Region:   {config.s3.region}")
    print(f"  Bucket:   {config.s3.bucket_raw}")

    print("\nListing up to 10 conversation folders...")
    folders = list(islice(handler.list_conversation_folders(limit=10), 10))

    if not folders:
        print("No folders found under raw/outlook/.")
        return

    print(f"Found {len(folders)} folder(s) in the first page:")
    for idx, folder in enumerate(folders, start=1):
        print(f"  {idx:02d}. {folder.name} ({len(folder.files)} files)")

    sample = folders[0]
    print(f"\nSampling folder: {sample.name}")
    data = handler.stream_conversation_data(sample)

    conversation_txt = data.get("conversation_txt", "")
    manifest = data.get("manifest", {})
    summary = data.get("summary", {})
    attachments = data.get("attachments", [])

    print(
        "conversation.txt stats:",
        f"{len(conversation_txt)} characters,",
        f"{conversation_txt.count(chr(10))} lines",
    )
    print("manifest keys:", sorted(manifest.keys()))
    print("summary keys:", sorted(summary.keys()))
    print(f"Attachments queued: {len(attachments)}")


if __name__ == "__main__":
    main()
