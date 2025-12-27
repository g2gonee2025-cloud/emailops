import argparse
import json
import sys
from pathlib import Path

import boto3
from botocore.client import Config

# Add backend/src to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "backend" / "src"))
from cortex.config.loader import get_config


def fetch_random_manifest(
    prefix: str = "", max_scan: int = 300, max_variations: int = 3
):
    """
    Scans an S3 bucket for manifest.json files and prints schema variations.

    Args:
        prefix: The S3 prefix to start scanning from.
        max_scan: The maximum number of manifests to scan.
        max_variations: The maximum number of schema variations to find.
    """
    try:
        config = get_config()
        s3_config = config.storage

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
        pagination_args = {"Bucket": s3_config.bucket_raw}
        if prefix:
            print(f"Scanning with prefix: {prefix}")
            pagination_args["Prefix"] = prefix
        page_iterator = paginator.paginate(**pagination_args)

        print("Scanning for schema variations...")

        seen_versions = set()

        count = 0
        for page in page_iterator:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith("manifest.json"):
                    count += 1
                    try:
                        resp = client.get_object(Bucket=s3_config.bucket_raw, Key=key)
                        data = json.loads(resp["Body"].read().decode("utf-8"))

                        ver = data.get("manifest_version", "unknown")
                        has_msgs = "messages" in data

                        sig = f"ver={ver}|has_msgs={has_msgs}"

                        if sig not in seen_versions:
                            print(f"\n✅ FOUND NEW SCHEMA VARIATION: {sig}")
                            print(f"Key: {key}")
                            print("=" * 60)
                            # To prevent PII leakage, just print keys
                            print(json.dumps(list(data.keys()), indent=2))
                            print("=" * 60)
                            seen_versions.add(sig)

                        if len(seen_versions) >= max_variations:
                            print(f"Found {max_variations} variations, stopping.")
                            return

                    except Exception as e:
                        print(
                            f"Warning: Failed to process manifest {key}: {e}",
                            file=sys.stderr,
                        )

            if count > max_scan:
                print(f"Scanned {max_scan} manifests, stopping.")
                break

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scan S3 for manifest schema variations."
    )
    parser.add_argument(
        "--prefix", type=str, default="", help="The S3 prefix to start scanning from."
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=300,
        help="The maximum number of manifests to scan.",
    )
    parser.add_argument(
        "--max-variations",
        type=int,
        default=3,
        help="The maximum number of schema variations to find.",
    )
    args = parser.parse_args()

    fetch_random_manifest(
        prefix=args.prefix,
        max_scan=args.max_scan,
        max_variations=args.max_variations,
    )
