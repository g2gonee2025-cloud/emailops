import argparse
import logging
import os
import sys
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """
    Moves an object in an S3-compatible storage.
    """
    parser = argparse.ArgumentParser(description="Move an object in S3.")
    parser.add_argument("src", help="The source key of the object to move.")
    parser.add_argument("dest", help="The destination key for the object.")
    args = parser.parse_args()

    # --- Configuration and Validation ---
    endpoint_url = os.getenv("OUTLOOKCORTEX_S3_ENDPOINT")
    access_key = os.getenv("OUTLOOKCORTEX_S3_ACCESS_KEY")
    secret_key = os.getenv("OUTLOOKCORTEX_S3_SECRET_KEY")
    bucket_name = os.getenv("OUTLOOKCORTEX_S3_BUCKET_RAW")
    region_name = os.getenv("OUTLOOKCORTEX_S3_REGION")

    required_vars = {
        "OUTLOOKCORTEX_S3_ENDPOINT": endpoint_url,
        "OUTLOOKCORTEX_S3_ACCESS_KEY": access_key,
        "OUTLOOKCORTEX_S3_SECRET_KEY": secret_key,
        "OUTLOOKCORTEX_S3_BUCKET_RAW": bucket_name,
    }

    for var_name, value in required_vars.items():
        if not value:
            logging.error(f"Error: Environment variable {var_name} is not set.")
            sys.exit(1)

    # Security: Validate endpoint URL
    parsed_url = urlparse(endpoint_url)
    if parsed_url.scheme != "https":
        logging.error(
            f"Error: Insecure endpoint URL '{endpoint_url}'. "
            "Only https:// is supported."
        )
        sys.exit(1)

    try:
        # --- S3 Client Initialization ---
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name,
        )

        # --- S3 Operations ---
        copy_source = {"Bucket": bucket_name, "Key": args.src}

        logging.info(f"Attempting to move object in bucket '{bucket_name}'.")
        logging.warning(
            "This is a non-atomic operation (copy then delete). "
            "If the script fails, the object may be duplicated or remain in the source location."
        )

        s3.copy_object(Bucket=bucket_name, CopySource=copy_source, Key=args.dest)
        s3.delete_object(Bucket=bucket_name, Key=args.src)

        logging.info("Object moved successfully.")

    except ClientError as e:
        # Catch specific boto3 client errors
        logging.error(f"An S3 client error occurred: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        # Catch other potential errors (e.g., client instantiation)
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
