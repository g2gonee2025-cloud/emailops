import argparse
import json
import sys
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

# Add backend/src to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "backend" / "src"))

from cortex.config.loader import get_config


def main(target: int, prefix: str):
    """
    Scans an S3 bucket for multi-message manifests and prints information
    about them.
    """
    try:
        config = get_config()
        s3_config = getattr(config, "storage", None)
        if s3_config is None:
            raise RuntimeError(
                "Storage configuration is missing; cannot connect to S3."
            )

        print(f"Connecting to S3: {s3_config.endpoint_url}")
        print(f"Bucket: {s3_config.bucket_raw}")
        if prefix:
            print(f"Scanning with prefix: {prefix}")

        session = boto3.session.Session()
        client = session.client(
            "s3",
            region_name=s3_config.region,
            endpoint_url=s3_config.endpoint_url,
            config=Config(signature_version="s3v4"),
        )

        paginator = client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=s3_config.bucket_raw, Prefix=prefix)

        found_count = 0

        print("\nScanning for multi-message manifests...")

        for page in page_iterator:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith("manifest.json"):
                    # Download and inspect
                    try:
                        response = client.get_object(
                            Bucket=s3_config.bucket_raw, Key=key
                        )
                        content = response["Body"].read().decode("utf-8")
                        manifest = json.loads(content)

                        msgs = manifest.get("messages", [])
                        if isinstance(msgs, list) and len(msgs) > 1:
                            print(f"\n✅ Found Multi-Message Thread: {key}")
                            print(f"  - Message Count: {len(msgs)}")

                            found_count += 1
                            if found_count >= target:
                                return
                    except json.JSONDecodeError:
                        print(f"  - Invalid JSON in manifest: {key}")
                    except client.exceptions.ClientError as e:
                        # More specific S3-related error
                        print(
                            f"  - AWS Client Error for key {key}: {e.response['Error']['Code']}"
                        )
                    except Exception as e:
                        # General unexpected errors
                        print(f"  - Unexpected error for key {key}: {e}")

        if found_count == 0:
            print("\nNo multi-message manifests found.")

    except client.exceptions.ClientError as e:
        print(f"Fatal S3 Error: {e.response['Error']['Code']}")
    except Exception as e:
        print(f"Fatal Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan S3 for multi-message manifests.")
    parser.add_argument(
        "--target",
        type=int,
        default=2,
        help="Number of multi-message manifests to find before stopping.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="S3 prefix to scan within the bucket.",
    )
    args = parser.parse_args()

    main(target=args.target, prefix=args.prefix)
