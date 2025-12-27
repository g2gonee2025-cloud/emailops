#!/usr/bin/env python3
"""
Simple summary backfill - generates summaries only, no embedding.

For conversations without summary_text, this generates summaries using LLM
and stores them. Does NOT attempt to embed (use separate embedding job).
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add backend to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
if BACKEND_SRC.exists():
    sys.path.insert(0, str(BACKEND_SRC))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from cortex.db.models import Chunk, Conversation  # noqa: E402
from cortex.db.session import SessionLocal  # noqa: E402
from cortex.intelligence.summarizer import ConversationSummarizer  # noqa: E402
from sqlalchemy import select  # noqa: E402
from tqdm import tqdm  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("cortex.llm").setLevel(logging.WARNING)

logger = logging.getLogger("summary_backfill")


def get_missing_conversations(limit: int | None = None) -> list[tuple[uuid.UUID, str]]:
    """Get conversations without summaries."""
    with SessionLocal() as session:
        stmt = select(
            Conversation.conversation_id,
            Conversation.tenant_id,
        ).where(
            (Conversation.summary_text.is_(None)) | (Conversation.summary_text == "")
        )
        if limit:
            stmt = stmt.limit(limit)
        results = session.execute(stmt).all()
        return [(r[0], r[1]) for r in results]


def generate_summary(
    conversation_id: uuid.UUID, tenant_id: str, summarizer: ConversationSummarizer
) -> dict:
    """Generate summary for a single conversation."""
    result = {"conversation_id": str(conversation_id), "success": False, "error": None}

    try:
        with SessionLocal() as session:
            # Get conversation with chunks
            convo = session.execute(
                select(Conversation).where(
                    Conversation.conversation_id == conversation_id
                )
            ).scalar_one_or_none()

            if not convo:
                result["error"] = "Conversation not found"
                return result

            # Get chunks to build context
            chunks = (
                session.execute(
                    select(Chunk).where(
                        Chunk.conversation_id == conversation_id,
                        Chunk.is_summary.is_(False),
                    )
                )
                .scalars()
                .all()
            )

            if not chunks:
                result["error"] = "No chunks for conversation"
                return result

            # Build text from chunks
            text = "\n".join([c.text for c in chunks if c.text])
            if len(text) < 50:
                result["error"] = "Text too short"
                return result

            # Generate summary (no embedding)
            summary_text = summarizer.generate_summary(text)

            if not summary_text or len(summary_text.strip()) < 10:
                result["error"] = "Summary generation failed"
                return result

            # Update conversation
            convo.summary_text = summary_text
            session.commit()

            result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Simple Summary Backfill (no embedding)"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    convos = get_missing_conversations(args.limit)
    total = len(convos)
    logger.info(f"Found {total} conversations without summaries")

    if total == 0:
        logger.info("Nothing to do!")
        return

    summarizer = ConversationSummarizer()
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_summary, cid, tid, summarizer): cid
            for cid, tid in convos
        }

        with tqdm(total=total, desc="Generating summaries") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    success += 1
                else:
                    failed += 1
                pbar.update(1)
                pbar.set_postfix(ok=success, fail=failed)

    logger.info(f"Done: {success} success, {failed} failed")


if __name__ == "__main__":
    main()
