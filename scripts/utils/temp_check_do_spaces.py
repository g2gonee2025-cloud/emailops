from __future__ import annotations

import logging
import sys

from cortex.common.exceptions import ConfigurationError
from cortex.config.loader import get_config
from cortex.ingestion.s3_source import S3SourceHandler


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        config = get_config()
        handler = S3SourceHandler()
    except (ConfigurationError, ValueError) as e:
        logging.error(f"Initialization failed: {e}")
        logging.error(
            "Please ensure your .env file is configured correctly and has all required S3 variables."
        )
        sys.exit(1)

    logging.info("DigitalOcean Spaces configuration:")
    logging.debug(f"  Endpoint: {config.s3.endpoint_url}")
    logging.debug(f"  Region:   {config.s3.region}")
    logging.debug(f"  Bucket:   {config.s3.bucket_raw}")

    search_prefix = "Outlook/"
    limit = 10
    logging.info(f"\nListing up to {limit} conversation folders under '{search_prefix}'...")
    folders = list(handler.list_conversation_folders(prefix=search_prefix, limit=limit))

    if not folders:
        logging.warning(
            f"No conversation folders found in bucket '{handler.bucket}' with prefix '{search_prefix}'."
        )
        return

    logging.info(f"Found {len(folders)} folder(s):")
    for idx, folder in enumerate(folders, start=1):
        logging.info(f"  {idx:02d}. {folder.name}")

    sample = folders[0]
    try:
        num_files = len(sample.files)
    except TypeError:
        sample.files = list(sample.files)
        num_files = len(sample.files)
    logging.info(f"\nSampling folder: {sample.name} ({num_files} files)")

    data = handler.stream_conversation_data(sample)
    if not isinstance(data, dict):
        logging.error(f"Failed to stream data for folder '{sample.name}', received invalid data.")
        return

    conversation_txt = data.get("conversation_txt", "")
    manifest = data.get("manifest", {})
    summary = data.get("summary", {})
    attachments = data.get("attachments", [])

    # LOGIC_ERRORS/TYPE_ERRORS: Fix line counting and ensure type safety.
    if isinstance(conversation_txt, str):
        # Use splitlines() for accurate line counting across LF/CRLF.
        num_lines = len(conversation_txt.splitlines())
        logging.info(
            f"conversation.txt stats: {len(conversation_txt)} characters, {num_lines} lines"
        )
    else:
        # Handle cases where the value might be bytes or another type.
        logging.warning("'conversation_txt' is not a string, skipping stats.")

    # TYPE_ERRORS: Ensure manifest is a dictionary before accessing keys.
    if isinstance(manifest, dict):
        try:
            logging.info(f"manifest keys: {sorted(manifest.keys())}")
        except TypeError:
            logging.warning("manifest keys are not comparable, cannot sort.")
            logging.info(f"manifest keys (unsorted): {list(manifest.keys())}")
    else:
        logging.warning("'manifest' is not a dictionary, skipping keys.")

    # TYPE_ERRORS: Ensure summary is a dictionary before accessing keys.
    if isinstance(summary, dict):
        try:
            logging.info(f"summary keys: {sorted(summary.keys())}")
        except TypeError:
            logging.warning("summary keys are not comparable, cannot sort.")
            logging.info(f"summary keys (unsorted): {list(summary.keys())}")
    else:
        logging.warning("'summary' is not a dictionary, skipping keys.")

    if isinstance(attachments, list):
        logging.info(f"Attachments queued: {len(attachments)}")
    else:
        logging.warning("'attachments' is not a list, cannot count items.")


if __name__ == "__main__":
    main()
