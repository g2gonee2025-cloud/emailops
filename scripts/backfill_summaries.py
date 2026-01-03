import argparse
import concurrent.futures
import logging
import uuid
from logging.handlers import RotatingFileHandler

from cortex.db.models import Chunk, Conversation
from cortex.db.session import SessionLocal
from cortex.intelligence.summarizer import ConversationSummarizer
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
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

# Limit input context to summarizer to avoid excessive token usage.
# This is a safeguard, not a precise tokenizer limit.
MAX_CONTENT_LENGTH = 15000
# Position for a summary chunk, which doesn't have a linear position.
SUMMARY_CHUNK_POSITION = -1


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
    conversation_id: uuid.UUID,
    tenant_id: str,
    dry_run: bool,
    summarizer: ConversationSummarizer,
) -> bool:
    """
    Worker task: Process a single conversation in its own session.
    """
    # Create a fresh session for this thread
    try:
        with SessionLocal() as session:
            convo = session.get(Conversation, conversation_id)
            if not convo:
                logger.warning(f"Conversation {conversation_id} not found.")
                return False

            # 1. Reconstruct text
            context = get_conversation_context(session, conversation_id)
            if not context.strip():
                return False

            # Abort if in dry-run mode before expensive API calls
            if dry_run:
                logger.info(
                    f"[DRY RUN] Would process conversation {conversation_id}."
                )
                return True

            # 2. Generate Summary (with truncation)
            summary_text = summarizer.generate_summary(context[:MAX_CONTENT_LENGTH])
            if not summary_text:
                logger.warning(
                    f"Summary generation failed (empty) for {conversation_id}."
                )
                return False

            # 3. Embed Summary
            final_embedding = None
            try:
                summary_embedding = summarizer.embed_summary(summary_text)
                if summary_embedding and len(summary_embedding) > 0:
                    final_embedding = summary_embedding
            except Exception as e:
                logger.error(f"Embedding failed for {conversation_id}: {e}")

            # 4. Update Conversation
            convo.summary_text = summary_text
            session.add(convo)

            # 5. Upsert Chunk
            tenant_ns = uuid.uuid5(uuid.NAMESPACE_DNS, f"tenant:{tenant_id}")
            chunk_id = generate_stable_id(
                tenant_ns, "chunk", str(conversation_id), "summary"
            )

            # Upsert logic with retry for race conditions
            for attempt in range(3):
                try:
                    existing_chunk = session.get(Chunk, chunk_id)
                    if existing_chunk:
                        existing_chunk.text = summary_text
                        if final_embedding is not None:
                            existing_chunk.embedding = final_embedding
                        session.add(existing_chunk)
                    else:
                        if final_embedding is not None:
                            new_chunk = Chunk(
                                chunk_id=chunk_id,
                                tenant_id=tenant_id,
                                conversation_id=conversation_id,
                                is_summary=True,
                                is_attachment=False,
                                chunk_type="summary",
                                text=summary_text,
                                embedding=final_embedding,
                                position=SUMMARY_CHUNK_POSITION,
                                char_start=0,
                                char_end=len(summary_text),
                                section_path="summary",
                                extra_data={"generated_by": "backfill_summaries"},
                            )
                            session.add(new_chunk)
                        else:
                            logger.warning(
                                f"Skipping chunk creation for {conversation_id} due to embedding failure."
                            )

                    session.commit()
                    return True
                except IntegrityError:
                    session.rollback()
                    logger.warning(
                        f"Race condition on attempt {attempt + 1} for {conversation_id}, retrying."
                    )
                    if attempt == 2:
                        logger.error(
                            f"Failed to upsert chunk for {conversation_id} after 3 retries."
                        )
                        return False
                except Exception:
                    session.rollback()
                    raise  # Re-raise other exceptions to be caught by the outer block
            return False

    except Exception as e:
        logger.error(f"Failed to process {conversation_id}: {e}")
        # Session rollback is handled automatically by the `with` statement context exit
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
        "--tenant", type=str, default=None, help="Tenant ID filter. If not provided, processes all tenants."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of concurrent workers."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of conversations to fetch from DB at a time.",
    )
    args = parser.parse_args()

    # 1. Get total count for progress bar and batching calculation
    with SessionLocal() as session:
        count_stmt = select(func.count(Conversation.conversation_id))
        if args.tenant:
            count_stmt = count_stmt.where(Conversation.tenant_id == args.tenant)
        if not args.force:
            count_stmt = count_stmt.where(Conversation.summary_text.is_(None))
        if args.limit:
            total_convs = min(args.limit, session.execute(count_stmt).scalar())
        else:
            total_convs = session.execute(count_stmt).scalar()

    logger.info(f"Targeting {total_convs} conversations with {args.workers} workers.")
    if total_convs == 0:
        return

    # Initialize a single summarizer to be shared across all threads
    summarizer = ConversationSummarizer()
    success_count = 0

    # 2. Process in batches
    with tqdm(total=total_convs, unit="conv") as pbar:
        with SessionLocal() as session:
            offset = 0
            while offset < total_convs:
                batch_limit = min(args.batch_size, total_convs - offset)
                if args.limit:
                    batch_limit = min(batch_limit, args.limit - offset)

                stmt = select(Conversation.conversation_id, Conversation.tenant_id)
                if args.tenant:
                    stmt = stmt.where(Conversation.tenant_id == args.tenant)
                if not args.force:
                    stmt = stmt.where(Conversation.summary_text.is_(None))

                stmt = stmt.order_by(Conversation.created_at).offset(offset).limit(batch_limit)
                rows = session.execute(stmt).all()

                if not rows:
                    break  # No more rows to process

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=args.workers
                ) as executor:
                    futures = {
                        executor.submit(
                            process_conversation_task,
                            row[0],  # conversation_id
                            row[1],  # tenant_id
                            args.dry_run,
                            summarizer,
                        ): row[0]
                        for row in rows
                    }

                    for future in concurrent.futures.as_completed(futures):
                        try:
                            if future.result():
                                success_count += 1
                        except Exception as e:
                            logger.error(
                                f"A worker failed with an unhandled exception: {e}"
                            )
                        finally:
                            pbar.update(1)

                offset += len(rows)
                if args.limit and offset >= args.limit:
                    break

    logger.info(
        f"Backfill complete. Successfully processed: {success_count}/{total_convs}"
    )


if __name__ == "__main__":
    main()
