import logging
import sys
import unittest.mock
from pathlib import Path

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("deep_dive.log")],
)
logger = logging.getLogger("DeepDive")

# Add backend/src to path dynamically
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_SRC = (REPO_ROOT / "backend" / "src").resolve()
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))


# --- MOCK EMBEDDINGS ---
# We mock EmbeddingsClient to avoid torch dependency issues and speed up the test.
class MockEmbeddingsClient:
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Return dummy 1536-dim vectors (or whatever the config says, usually 768 or 1536)
        # Using 3840 based on one of the migrations seen earlier, but 128 is fewer bytes for test.
        # Let's check config? Safest is to just return a list of zeroes of length 1536.
        return [[0.0] * 3840 for _ in texts]


# Patch it before importing mailroom/processor implies it
sys.modules["cortex.embeddings.client"] = unittest.mock.Mock()
sys.modules["cortex.embeddings.client"].EmbeddingsClient = MockEmbeddingsClient

from cortex.db.models import Attachment, Conversation  # noqa: E402
from cortex.db.session import SessionLocal  # noqa: E402

# --- IMPORTS ---
from cortex.ingestion.processor import IngestionProcessor  # noqa: E402


def analyze_db_results(tenant_id: str):
    print("\n--- DATABASE ANALYSIS ---")
    with SessionLocal() as session:
        # 1. Conversations
        conversations = (
            session.query(Conversation)
            .filter(Conversation.tenant_id == tenant_id)
            .all()
        )
        print(f"Conversations created: {len(conversations)}")

        # 2. Messages are within Conversations, so we can't query them directly anymore.
        # We can count them by iterating through conversations.
        total_messages = sum(len(c.messages) for c in conversations if c.messages)
        print(f"Messages created: {total_messages}")

        # 3. Attachments
        atts = (
            session.query(Attachment)
            .join(Conversation)
            .filter(Conversation.tenant_id == tenant_id)
            .all()
        )
        print(f"Attachments created: {len(atts)}")

        # Analysis: Enriched Metadata from Conversation
        enriched_count = 0
        for conv in conversations:
            if conv.extra_data and (
                conv.extra_data.get("sender") or conv.extra_data.get("mail_subject")
            ):
                enriched_count += 1

        print(f"Conversations with Enriched Metadata (from CSV): {enriched_count}")

        if len(conversations) > 0:
            print(f"Enrichment Rate: {enriched_count / len(conversations) * 100:.1f}%")

            print("\nSample Conversation Metadata:")
            for conv in conversations[:5]:
                print(f" - {conv.subject}: {conv.extra_data}")


def run_deep_dive():
    print("Starting Deep Dive Ingestion (Limit: 100)...")

    tenant_id = "deep-dive-tenant"
    processor = IngestionProcessor(tenant_id=tenant_id)

    # Process 100 folders
    # Note: prefix="Outlook/" is the corrected default
    NUM_CONVERSATIONS = 100
    summaries = processor.run_full_ingestion(prefix="Outlook/", limit=NUM_CONVERSATIONS)

    print("\n--- INGESTION SUMMARY ---")
    print(f"Processed: {len(summaries)} folders")

    total_chunks = 0
    total_errors = 0

    for s in summaries:
        total_chunks += s.chunks_created
        if s.problems or s.aborted_reason:
            total_errors += 1
            print(f"❌ Job {s.job_id} failed: {s.aborted_reason} / {s.problems}")

    print(f"Total Chunks Generated: {total_chunks}")
    print(f"Total Errors: {total_errors}")

    analyze_db_results(tenant_id)


if __name__ == "__main__":
    run_deep_dive()
