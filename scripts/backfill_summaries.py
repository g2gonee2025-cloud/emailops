import argparse
import concurrent.futures
import logging
import uuid
from logging.handlers import RotatingFileHandler

from cortex.db.models import Chunk, Conversation
from cortex.db.session import SessionLocal
from cortex.intelligence.summarizer import ConversationSummarizer
from sqlalchemy import select
from sqlalchemy.orm import Session
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            "backfill_summaries.log", maxBytes=10 * 1024 * 1024, backupCount=5
        ),
    ],
)
# Silence internal loggers for cleaner tqdm output
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("cortex.llm.runtime").setLevel(logging.WARNING)

logger = logging.getLogger("backfill")


def generate_stable_id(namespace: uuid.UUID, *args: str) -> uuid.UUID:
    """Generate a stable UUID-5 based on a namespace and a list of string arguments."""
    joined = ":".join(str(a) for a in args)
    return uuid.uuid5(namespace, joined)


def get_conversation_context(session: Session, conversation_id: uuid.UUID) -> str:
    """Reconstruct conversation text from chunks."""
    stmt = (
        select(Chunk)
        .where(Chunk.conversation_id == conversation_id)
        .where(Chunk.is_summary.is_(False))
        # Order by: attachments last, then by position
        .order_by(Chunk.is_attachment, Chunk.position)
    )
    chunks = session.execute(stmt).scalars().all()

    body_chunks = [c for c in chunks if not c.is_attachment]
    att_chunks = [c for c in chunks if c.is_attachment]

    parts = []
    if body_chunks:
        parts.append("--- Conversation Messages ---")
        for c in body_chunks:
            if c.text:
                parts.append(c.text)

    if att_chunks:
        parts.append("\n--- Attachments ---")
        for c in att_chunks:
            if c.text:
                parts.append(c.text)

    return "\n\n".join(parts)


def process_conversation_task(
    conversation_id: uuid.UUID, tenant_id: str, dry_run: bool
) -> bool:
    """
    Worker task: Process a single conversation in its own session.
    """
    # Create a fresh session for this thread
    with SessionLocal() as session:
        try:
            convo = session.get(Conversation, conversation_id)
            if not convo:
                logger.warning(f"Conversation {conversation_id} not found.")
                return False

            # 1. Reconstruct text
            context = get_conversation_context(session, conversation_id)
            if not context.strip():
                # logger.warning(f"No text content found for {conversation_id}. Skipping.")
                return False

            # Initialize summarizer (runtime is thread-safe)
            # We init here or reuse a global? LLMRuntime internal locks handle concurrency.
            # Creating new instance per task is safe but might be slightly overhead.
            # Ideally we pass a shared instance, but for simplicity we can init here or use a global.
            # Let's use a fresh one to be safe with unknown state.
            summarizer = ConversationSummarizer()

            # 2. Generate Summary
            summary_text = summarizer.generate_summary(context)
            if not summary_text:
                logger.warning(
                    f"Summary generation failed (empty) for {conversation_id}."
                )
                return False

            # 3. Embed Summary (with fallback for outages)
            # Explicitly catching connection errors here to ensure text persistence
            final_embedding = None
            try:
                summary_embedding = summarizer.embed_summary(summary_text)
                if summary_embedding and len(summary_embedding) > 0:
                    final_embedding = summary_embedding
            except Exception as e:
                logger.error(f"Embedding failed for {conversation_id}: {e}")
                # Continue with None embedding

            if dry_run:
                # logger.info(f"[DRY RUN] {conversation_id}: {summary_text[:30]}...")
                return True

            # 4. Update Conversation
            convo.summary_text = summary_text
            session.add(convo)

            # 5. Upsert Chunk
            tenant_ns = uuid.uuid5(uuid.NAMESPACE_DNS, f"tenant:{tenant_id}")
            chunk_id = generate_stable_id(
                tenant_ns, "chunk", str(conversation_id), "summary"
            )

            existing_chunk = session.execute(
                select(Chunk).where(Chunk.chunk_id == chunk_id)
            ).scalar_one_or_none()

            if existing_chunk:
                existing_chunk.text = summary_text
                # Only update embedding if we actually got a new one, otherwise keep old or set to None?
                # If API is down, we probably shouldn't wipe an OLD valid embedding if it existed.
                # But here we are generating NEW summary text, so old embedding is invalid anyway.
                existing_chunk.embedding = final_embedding
                session.add(existing_chunk)
            else:
                new_chunk = Chunk(
                    chunk_id=chunk_id,
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    is_summary=True,
                    is_attachment=False,
                    chunk_type="summary",
                    text=summary_text,
                    embedding=final_embedding,
                    position=-1,
                    char_start=0,
                    char_end=len(summary_text),
                    section_path="summary",
                    extra_data={"generated_by": "backfill_summaries"},
                )
                session.add(new_chunk)

            session.commit()
            return True

        except Exception as e:
            logger.error(f"Failed to process {conversation_id}: {e}")
            session.rollback()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Backfill conversation summaries (Parallel)."
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not commit changes.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of conversations."
    )
    parser.add_argument(
        "--force", action="store_true", help="Reprocess even if summary exists."
    )
    parser.add_argument(
        "--tenant", type=str, default="default", help="Tenant ID filter."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of concurrent workers."
    )
    args = parser.parse_args()

    # 1. Fetch IDs to process (Main Thread)
    with SessionLocal() as session:
        stmt = select(Conversation.conversation_id, Conversation.tenant_id)
        if args.tenant:
            stmt = stmt.where(Conversation.tenant_id == args.tenant)

        if not args.force:
            stmt = stmt.where(Conversation.summary_text.is_(None))

        if args.limit:
            stmt = stmt.limit(args.limit)

        # Execute and fetch all IDs into memory (list of tuples) - safe for normal dataset sizes
        # For MASSIVE datasets, we'd use yield_per, but fetching just UUIDs is lightweight.
        rows = session.execute(stmt).all()

    total_convs = len(rows)
    logger.info(f"Targeting {total_convs} conversations with {args.workers} workers.")

    if total_convs == 0:
        return

    # 2. Process in Parallel
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_conversation_task,
                row.conversation_id,
                row.tenant_id,
                args.dry_run,
            ): row.conversation_id
            for row in rows
        }

        # Progress bar
        with tqdm(total=total_convs, unit="conv") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    success_count += 1
                pbar.update(1)

    logger.info(
        f"Backfill complete. Successfully processed: {success_count}/{total_convs}"
    )


if __name__ == "__main__":
    main()
