import sys
import traceback

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from cortex.config.loader import get_config


def _get_aws_account_id(sts_client):
    """Get the AWS account ID."""
    try:
        # SECURITY: Get caller identity to retrieve the account ID.
        return sts_client.get_caller_identity()["Account"]
    except ClientError as e:
        # COMPATIBILITY: Handle cases where the provider is not AWS (e.g., DigitalOcean Spaces).
        print(f"Warning: Could not get AWS account ID. This may be expected if you are not using AWS S3. Error: {e}")
        return None


def main():
    """
    Checks the connectivity to the S3 bucket as defined in the project's configuration.

    This script is designed to be run as a module from the project root:
    `python -m scripts.check.check_s3`

    It will exit with status code 0 on success and 1 on failure.
    """
    try:
        config = get_config()
        # NULL_SAFETY: Validate config structure before accessing attributes.
        if not all(
            [
                config,
                config.storage,
                config.storage.endpoint_url,
                config.storage.region,
                config.storage.bucket_raw,
                config.storage.access_key,
                config.storage.secret_key,
            ]
        ):
            raise ValueError("S3 storage configuration is incomplete. Check your .env file.")

        # SECURITY: Do not print raw configuration details.
        print("Checking S3 connectivity...")
        print(f"Bucket: {config.storage.bucket_raw}")

        # PERFORMANCE: Set connect and read timeouts to avoid hanging.
        s3_config = Config(
            signature_version="s3v4",
            connect_timeout=5,
            read_timeout=5,
        )

        s3 = boto3.client(
            "s3",
            endpoint_url=config.storage.endpoint_url,
            region_name=config.storage.region,
            aws_access_key_id=config.storage.access_key,
            aws_secret_access_key=config.storage.secret_key,
            config=s3_config,
        )

        sts_client = boto3.client(
            "sts",
            endpoint_url=config.storage.endpoint_url,
            region_name=config.storage.region,
            aws_access_key_id=config.storage.access_key,
            aws_secret_access_key=config.storage.secret_key,
            config=s3_config,
        )

        account_id = _get_aws_account_id(sts_client)

        list_objects_kwargs = {
            "Bucket": config.storage.bucket_raw,
            "MaxKeys": 1,
        }
        if account_id:
            # SECURITY: Add ExpectedBucketOwner to verify S3 bucket ownership.
            list_objects_kwargs["ExpectedBucketOwner"] = account_id

        # List first 1 object to verify access
        response = s3.list_objects_v2(**list_objects_kwargs)  # NOSONAR

        print("Connectivity Check: SUCCESS")
        if "Contents" in response:
            print(f"Found object: {response['Contents'][0]['Key']}")
        else:
            # STYLE: Corrected misleading message.
            print("Bucket is empty.")

    # EXCEPTION_HANDLING: Catch specific exceptions and exit with non-zero status.
    except (BotoCoreError, ClientError) as e:
        print(f"Connectivity Check: FAILED\nS3 Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    except (ValueError, AttributeError) as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
