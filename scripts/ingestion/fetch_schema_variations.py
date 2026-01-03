import argparse
import json
import sys
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

# Keywords to detect potentially sensitive information in JSON keys for redaction.
SENSITIVE_KEY_SUBSTRINGS = [
    "email",
    "user",
    "pass",
    "secret",
    "token",
    "key",
    "auth",
    "name",
    "address",
    "phone",
    "account",
    "customer",
    "session",
]


def find_project_root(marker: str = "pyproject.toml") -> Path:
    """Find the project root by searching for a marker file."""
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root marker: {marker}")


# Configure path for project imports
try:
    project_root = find_project_root()
    sys.path.append(str(project_root / "backend" / "src"))
    from cortex.config.loader import get_config
except (FileNotFoundError, ImportError) as e:
    print(
        f"Error: Could not import project modules. ({e})\n"
        "Ensure this script is run from within the project and that "
        "dependencies are installed.",
        file=sys.stderr,
    )
    sys.exit(1)


def redact_s3_key(key: str) -> str:
    """Redacts the middle part of an S3 key to prevent PII leakage."""
    parts = key.split("/")
    if len(parts) > 2:
        return f"{parts[0]}/.../{parts[-1]}"
    return key


def redact_json_keys(data: dict) -> list[str]:
    """Redacts keys from a dictionary that may contain sensitive info."""
    redacted_keys = []
    for key in data.keys():
        is_sensitive = any(sub in key.lower() for sub in SENSITIVE_KEY_SUBSTRINGS)
        if is_sensitive:
            # Hash the key to provide a consistent but anonymous identifier
            redacted_keys.append(f"<REDACTED:{hash(key)}>")
        else:
            redacted_keys.append(key)
    return redacted_keys


def fetch_schema_variations(
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

        # Validate that required S3 configuration is present
        required_s3_fields = [
            "region",
            "endpoint_url",
            "access_key",
            "secret_key",
            "bucket_raw",
        ]
        for field in required_s3_fields:
            if not getattr(s3_config, field, None):
                raise ValueError(f"Missing required S3 config: '{field}'")

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
        total_processed = 0
        for page in page_iterator:
            if "Contents" not in page:
                continue

            # Stop if we've already scanned enough manifests in previous pages
            if count >= max_scan:
                print(f"Scanned {count} manifests, stopping.")
                break

            for obj in page["Contents"]:
                total_processed += 1
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
                            print(f"Key: {redact_s3_key(key)}")
                            print("=" * 60)
                            # To prevent PII leakage, just print keys
                            print(json.dumps(redact_json_keys(data), indent=2))
                            print("=" * 60)
                            seen_versions.add(sig)

                        if len(seen_versions) >= max_variations:
                            print(f"Found {max_variations} variations, stopping.")
                            return

                    except (json.JSONDecodeError, ClientError) as e:
                        print(
                            f"Warning: Failed to process manifest {redact_s3_key(key)}: {type(e).__name__}",
                            file=sys.stderr,
                        )

            if count >= max_scan:
                print(
                    f"Scanned {count} manifests across {total_processed} total S3 objects."
                )
                break

    except (ValueError, ClientError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def positive_int(value: str) -> int:
    """Argparse type checker for positive integers."""
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid integer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scan S3 for manifest schema variations."
    )
    parser.add_argument(
        "--prefix", type=str, default="", help="The S3 prefix to start scanning from."
    )
    parser.add_argument(
        "--max-scan",
        type=positive_int,
        default=300,
        help="The maximum number of manifests to scan.",
    )
    parser.add_argument(
        "--max-variations",
        type=positive_int,
        default=3,
        help="The maximum number of schema variations to find.",
    )
    args = parser.parse_args()

    fetch_schema_variations(
        prefix=args.prefix,
        max_scan=args.max_scan,
        max_variations=args.max_variations,
    )
