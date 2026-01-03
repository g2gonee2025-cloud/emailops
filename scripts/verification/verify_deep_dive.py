import argparse
import logging
import sys
from types import ModuleType
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import selectinload

# Configure Logging
logger = logging.getLogger("DeepDive")

def setup_logging():
    """Configure logging to file and stdout."""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("deep_dive.log"),
            ],
        )
    except (IOError, PermissionError) as e:
        print(f"Error setting up logging: {e}", file=sys.stderr)
        sys.exit(1)

# To run this script, ensure you are at the root of the repository and execute:
# PYTHONPATH=backend/src python3 scripts/verification/verify_deep_dive.py


# --- MOCK EMBEDDINGS ---
# We mock EmbeddingsClient to avoid torch dependency issues and speed up the test.
class MockEmbeddingsClient:
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Return dummy 1536-dim vectors
        return [[0.0] * 1536 for _ in texts]


# Patch it before importing mailroom/processor implies it
mock_module = ModuleType("cortex.embeddings.client")
mock_module.EmbeddingsClient = MockEmbeddingsClient
sys.modules["cortex.embeddings.client"] = mock_module


from cortex.db.models import Attachment, Conversation
from cortex.db.session import SessionLocal

# --- IMPORTS ---
from cortex.ingestion.processor import IngestionProcessor


def analyze_db_results(tenant_id: str):
    logger.info("--- DATABASE ANALYSIS ---")
    try:
        with SessionLocal() as session:
            # 1. Conversations
            conversations = (
                session.query(Conversation)
                .options(selectinload(Conversation.messages))
                .filter(Conversation.tenant_id == tenant_id)
                .all()
            )
            logger.info(f"Conversations created: {len(conversations)}")

            # 2. Messages
            total_messages = sum(len(c.messages) for c in conversations if c.messages)
            logger.info(f"Messages created: {total_messages}")

            # 3. Attachments
            atts = (
                session.query(Attachment)
                .join(Conversation)
                .filter(Conversation.tenant_id == tenant_id)
                .all()
            )
            logger.info(f"Attachments created: {len(atts)}")

            # Analysis: Enriched Metadata
            enriched_count = 0
            for conv in conversations:
                if isinstance(conv.extra_data, dict) and (
                    conv.extra_data.get("sender") or conv.extra_data.get("mail_subject")
                ):
                    enriched_count += 1

            logger.info(f"Conversations with Enriched Metadata (from CSV): {enriched_count}")

            if conversations:
                enrichment_rate = (enriched_count / len(conversations)) * 100
                logger.info(f"Enrichment Rate: {enrichment_rate:.1f}%")

                logger.info("Sample Conversation Metadata:")
                for conv in conversations[:5]:
                    logger.info(f" - {conv.subject}: {conv.extra_data}")
    except SQLAlchemyError as e:
        logger.error(f"Database query failed: {e}")


def run_deep_dive(num_conversations: int):
    logger.info(f"Starting Deep Dive Ingestion (Limit: {num_conversations})...")

    tenant_id = "deep-dive-tenant"
    processor = IngestionProcessor(tenant_id=tenant_id)

    try:
        summaries = processor.run_full_ingestion(prefix="Outlook/", limit=num_conversations)
    except Exception as e:
        logger.error(f"Ingestion process failed: {e}")
        summaries = None

    if not isinstance(summaries, list):
        logger.error("Ingestion did not return a valid list of summaries.")
        return

    logger.info("--- INGESTION SUMMARY ---")
    logger.info(f"Processed: {len(summaries)} folders")

    total_chunks = 0
    total_errors = 0

    for s in summaries:
        total_chunks += getattr(s, "chunks_created", 0) or 0
        problems = getattr(s, "problems", None)
        aborted_reason = getattr(s, "aborted_reason", None)
        if problems or aborted_reason:
            total_errors += 1
            job_id = getattr(s, "job_id", "Unknown")
            error_icon = "❌".encode("utf-8").decode("utf-8")
            logger.warning(f"{error_icon} Job {job_id} failed: {aborted_reason} / {problems}")

    logger.info(f"Total Chunks Generated: {total_chunks}")
    logger.info(f"Total Errors: {total_errors}")

    analyze_db_results(tenant_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a deep dive ingestion verification.")
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=100,
        help="Number of conversations to process.",
    )
    args = parser.parse_args()

    setup_logging()
    run_deep_dive(args.num_conversations)
