import logging
import sys
from pathlib import Path

# Add backend/src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from cortex.ingestion.processor import IngestionProcessor
from cortex.ingestion.s3_source import S3SourceHandler


def run_batch_ingest():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("batch_ingest")

    # Prefix needs to be recursive because of the bucket name appearing in the key
    prefix = "emailops-storage-tor1/raw/outlook/"
    limit = 250

    # Endpoint must be regional
    correct_endpoint = "https://tor1.digitaloceanspaces.com"
    bucket_name = "emailops-storage-tor1"

    logger.info(
        f"Starting batch ingestion (REAL EXECUTION). Prefix='{prefix}', Limit={limit}"
    )

    # Initialize processor
    processor = IngestionProcessor()

    # Override handler
    processor.s3_handler = S3SourceHandler(
        endpoint_url=correct_endpoint, region="tor1", bucket=bucket_name
    )

    # Run ingestion
    summaries = processor.run_full_ingestion(prefix=prefix, limit=limit)

    logger.info(f"Batch completed. Processed {len(summaries)} folders.")

    # Print summary (stats)
    success = sum(1 for s in summaries if not s.aborted_reason and not s.problems)
    failed = len(summaries) - success

    print("\n--- Batch Report ---")
    print(f"Total Folders Scanned: {len(summaries)}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")

    if len(summaries) > 0:
        print(f"\nSample Job 0: {summaries[0].job_id}")
        print(f"  Status: {summaries[0].status}")
        print(f"  Messages: {summaries[0].messages_ingested}")


if __name__ == "__main__":
    run_batch_ingest()
