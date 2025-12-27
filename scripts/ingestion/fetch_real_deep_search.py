import json
import logging
import sys
from pathlib import Path

import boto3
from botocore.client import Config

# Add backend/src to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "backend" / "src"))
from cortex.config.loader import get_config


def fetch_deep_real_manifest() -> None:
    try:
        config = get_config()
        s3_config = getattr(config, "storage", None)
        if s3_config is None:
            raise ValueError(
                "Missing 'storage' section in configuration; cannot fetch Deep Search manifest."
            )

        session = boto3.session.Session()
        client = session.client(
            "s3",
            region_name=s3_config.region,
            endpoint_url=s3_config.endpoint_url,
            aws_access_key_id=s3_config.access_key,
            aws_secret_access_key=s3_config.secret_key,
            config=Config(signature_version="s3v4"),
        )

        paginator = client.get_paginator("list_objects_v2")
        # Scan up to 20 pages (approx 20,000 objects if 1000 keys/page)
        # We need to find one with 'messages'
        page_iterator = paginator.paginate(Bucket=s3_config.bucket_raw)

        print("Deep scanning S3 for a manifest WITH 'messages' list...")

        count = 0
        for page in page_iterator:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith("manifest.json"):
                    count += 1
                    if count % 10 == 0:
                        sys.stdout.write(f"\rChecked {count} manifests...")
                        sys.stdout.flush()

                    try:
                        resp = client.get_object(Bucket=s3_config.bucket_raw, Key=key)
                        data = json.loads(resp["Body"].read().decode("utf-8"))

                        msgs = data.get("messages")
                        # Check strictly for list and non-empty
                        if isinstance(msgs, list) and len(msgs) > 0:
                            print(
                                f"\n\n✅ FOUND VALID MANIFEST IN BUCKET: {s3_config.bucket_raw}"
                            )
                            print("=" * 60)
                            print(
                                "Matching manifest found. Key omitted to prevent sensitive data exposure."
                            )
                            print("=" * 60)
                            return
                    except Exception as e:
                        logging.warning(
                            f"Could not process manifest in bucket {s3_config.bucket_raw}: {e}"
                        )
                        # Ignore individual read errors

            if count > 500:
                print("\nStopped after checking 500 manifests.")
                break

        print("\nNo manifest with 'messages' list found in search range.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    try:
        fetch_deep_real_manifest()
    except Exception:
        import logging

        logging.exception("Unhandled exception in fetch_real_deep_search")
