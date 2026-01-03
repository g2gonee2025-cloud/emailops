#!/usr/bin/env python3
"""
Backfill participants for existing conversations.

Parses Conversation.txt from S3 for each conversation and updates
the participants JSONB column with deduplicated From/To/Cc addresses.
"""

import json
import logging
import sys
from typing import Optional

from cortex.config.loader import get_config
from cortex.ingestion.conversation_parser import (
    extract_participants_from_conversation_txt,
)
from cortex.ingestion.s3_source import S3SourceHandler
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def backfill_participants(
    limit: int | None = None, dry_run: bool = False, batch_size: int = 100
):
    """
    Backfill participants for conversations with NULL participants.
    Args:
        limit: Optional limit on number of conversations to process.
        dry_run: If True, don't write to DB, just show what would be done.
        batch_size: Number of conversations to process per batch.
    """
    if limit is not None and limit < 0:
        raise ValueError("Limit must be a non-negative number.")

    config = get_config()
    engine = create_engine(config.database.url)
    handler = S3SourceHandler()

    with engine.connect() as conn:
        # Get conversations with NULL participants
        query = """
            SELECT conversation_id, folder_name
            FROM conversations
            WHERE participants IS NULL OR participants = '[]'::jsonb
            ORDER BY created_at
        """
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        # Fetch all conversations to process at once, removing streaming
        all_conversations = conn.execute(text(query)).fetchall()
        total_conversations = len(all_conversations)
        logger.info(f"Found {total_conversations} conversations to backfill.")

        updated = 0
        skipped = 0
        errors = 0

        # Process in batches
        for i in range(0, total_conversations, batch_size):
            batch = all_conversations[i : i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1}/{-(-total_conversations // batch_size)} (size: {len(batch)})"
            )
            updates_to_perform = []

            for row in batch:
                # Entire row processing is wrapped in a try/except to ensure
                # a single conversation failure doesn't stop the batch.
                try:
                    conv_id, folder_name = row

                    if not folder_name:
                        logger.warning(
                            f"Skipping conversation {conv_id} due to NULL folder_name."
                        )
                        skipped += 1
                        continue

                    # Build S3 prefix
                    prefix = f"Outlook/{folder_name}"
                    folders = list(
                        handler.list_conversation_folders(prefix=prefix, limit=1)
                    )

                    if not folders or not hasattr(folders[0], "files"):
                        logger.warning(
                            f"No S3 folder or files found for: {folder_name}"
                        )
                        skipped += 1
                        continue

                    # Find Conversation.txt
                    conv_txt_key = next(
                        (
                            f
                            for f in folders[0].files
                            if "conversation.txt" in f.lower()
                        ),
                        None,
                    )

                    if not conv_txt_key:
                        logger.warning(f"No Conversation.txt for: {folder_name}")
                        skipped += 1
                        continue

                    # Download and parse
                    content = handler.get_object_content(conv_txt_key)
                    text_content = content.decode("utf-8-sig", errors="replace")
                    participants = extract_participants_from_conversation_txt(
                        text_content
                    )

                    if not participants:
                        logger.debug(f"No participants found in: {folder_name}")
                        skipped += 1
                        continue

                    updates_to_perform.append(
                        {
                            "conv_id": str(conv_id),
                            "participants": json.dumps(participants),
                        }
                    )

                    if dry_run:
                        logger.info(
                            f"[DRY RUN] Would update {folder_name} with {len(participants)} participants"
                        )
                    else:
                        logger.info(
                            f"Queued update for {folder_name}: {len(participants)} participants"
                        )

                except Exception:
                    logger.exception(
                        f"Error processing conversation {conv_id} from folder {folder_name}"
                    )
                    errors += 1

            if updates_to_perform:
                if dry_run:
                    # In dry-run, we just count what would be updated.
                    updated += len(updates_to_perform)
                else:
                    # For actual runs, perform the database update.
                    transaction = conn.begin()
                    try:
                        update_stmt = text(
                            """
                            UPDATE conversations AS c
                            SET participants = v.participants::jsonb
                            FROM (VALUES :values) AS v (conversation_id, participants)
                            WHERE c.conversation_id = v.conversation_id::uuid
                            """
                        )

                        # Prepare data for VALUES clause
                        update_tuples = [
                            (d["conv_id"], d["participants"])
                            for d in updates_to_perform
                        ]

                        # SQLAlchemy 2.0 style parameter binding for execute
                        conn.execute(update_stmt, {"values": update_tuples})

                        transaction.commit()
                        updated_count = len(updates_to_perform)

                        if not dry_run:
                            updated += updated_count

                        logger.info(
                            f"Successfully updated {updated_count} conversations in batch."
                        )

                    except Exception:
                        logger.exception(
                            "Database update failed for batch. Rolling back."
                        )
                        if transaction:
                            transaction.rollback()
                        errors += len(updates_to_perform)

        logger.info(
            f"Backfill complete: {updated} updated, {skipped} skipped, {errors} errors"
        )
        return updated, skipped, errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Backfill participants for conversations"
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of conversations to process"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of conversations to process in a batch",
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    args = parser.parse_args()

    if args.batch_size <= 0:
        logger.error("Batch size must be a positive number")
        sys.exit(1)

    try:
        backfill_participants(
            limit=args.limit, dry_run=args.dry_run, batch_size=args.batch_size
        )
    except ValueError as e:
        logger.error(e)
        sys.exit(1)
