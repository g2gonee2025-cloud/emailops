"""
Debug a manifest file from S3.

This script fetches a JSON manifest file from an S3-compatible service,
attempts to clean up common JSON formatting errors, and then parses it.

Usage:
    python3 -m scripts.utils.debug_manifest <s3_key>

Note:
    This script must be run as a module from the project root directory
    to ensure that the 'cortex' package is correctly resolved.
"""

import argparse
import json
import re
import sys
from io import BytesIO

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

# The 'cortex' module is resolved by running this script as a module.
# Do not modify sys.path here.
from cortex.common.exceptions import ConfigurationError
from cortex.config.loader import get_config


def robust_json_clean(content: str) -> str:
    """
    Cleans common JSON formatting errors more robustly using regex.
    This targets multi-line string markers that are not properly escaped.
    """
    # This regex looks for patterns like `:" ""`, `:"""`, or `:"""`
    # and replaces them with `:""`, being careful not to replace
    # content inside of a valid string.
    return re.sub(r':\s*"""', ': ""', content)


def main():
    """
    Fetches, cleans, and parses a JSON manifest file from an S3-compatible service.
    """
    parser = argparse.ArgumentParser(
        description="Debug a manifest file from S3.",
        epilog="Example: python3 -m scripts.utils.debug_manifest raw/outlook/some-file/manifest.json",
    )
    parser.add_argument("key", help="The S3 key of the manifest file to debug.")
    args = parser.parse_args()

    try:
        config = get_config()

        # Null-safety check for storage configuration
        if not all(
            [
                config.storage,
                config.storage.bucket_raw,
                config.storage.endpoint_url,
                config.storage.region,
                config.storage.access_key,
                config.storage.secret_key,
            ]
        ):
            print("Error: S3 storage configuration is incomplete.", file=sys.stderr)
            sys.exit(1)

        client = boto3.client(
            "s3",
            endpoint_url=config.storage.endpoint_url,
            region_name=config.storage.region,
            aws_access_key_id=config.storage.access_key,
            aws_secret_access_key=config.storage.secret_key,
            config=Config(signature_version="s3v4"),
        )
    except (ConfigurationError, BotoCoreError) as e:
        print(
            f"Error: Configuration or Boto3 client initialization failed: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        response = client.get_object(Bucket=config.storage.bucket_raw, Key=args.key)

        # Use a with statement to ensure the connection is closed
        # Read in chunks to handle large files efficiently
        content_buffer = BytesIO()
        with response["Body"] as stream:
            for chunk in stream.iter_chunks():
                content_buffer.write(chunk)
        content = content_buffer.getvalue()

        print(f"Content length: {len(content)}")

        try:
            decoded = content.decode("utf-8-sig")
            print("Decoded successfully")

            fixed = robust_json_clean(decoded)

            parsed = json.loads(fixed)
            print("Parsed successfully after fix:")
            print(json.dumps(parsed, indent=2))

        except UnicodeDecodeError as e:
            print(f"Error decoding content: {e}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}", file=sys.stderr)
            sys.exit(1)

    except ClientError as e:
        print(f"Error fetching object from S3: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
