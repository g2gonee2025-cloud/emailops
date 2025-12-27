#!/usr/bin/env python3
"""
Backfill participants for existing conversations.

Parses Conversation.txt from S3 for each conversation and updates
the participants JSONB column with deduplicated From/To/Cc addresses.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

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


def backfill_participants(limit: int | None = None, dry_run: bool = False):
    """
    Backfill participants for conversations with NULL participants.

    Args:
        limit: Optional limit on number of conversations to process
        dry_run: If True, don't write to DB, just show what would be done
    """
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
            # Safe because limit is typed as int
            query += f" LIMIT {int(limit)}"

        result_proxy = conn.execution_options(stream_results=True).execute(text(query))

        updated = 0
        skipped = 0
        errors = 0

        import json

        for row in result_proxy:
            conv_id, folder_name = row
            try:
                # Build S3 prefix
                prefix = f"Outlook/{folder_name}/"
                folders = list(
                    handler.list_conversation_folders(
                        prefix=prefix.rstrip("/"), limit=1
                    )
                )

                if not folders:
                    logger.warning(f"No S3 folder found for: {folder_name}")
                    skipped += 1
                    continue

                # Find Conversation.txt
                conv_txt_key = None
                for f in folders[0].files:
                    if "conversation.txt" in f.lower():
                        conv_txt_key = f
                        break

                if not conv_txt_key:
                    logger.warning(f"No Conversation.txt for: {folder_name}")
                    skipped += 1
                    continue

                # Download and parse
                content = handler.get_object_content(conv_txt_key)
                text_content = content.decode("utf-8-sig", errors="replace")
                participants = extract_participants_from_conversation_txt(text_content)

                if not participants:
                    logger.debug(f"No participants found in: {folder_name}")
                    skipped += 1
                    continue

                if dry_run:
                    logger.info(
                        f"[DRY RUN] Would update {folder_name} with {len(participants)} participants"
                    )
                else:
                    # Update the database using raw psycopg2-style params
                    stmt = text(
                        """
                        UPDATE conversations
                        SET participants = CAST(:participants AS jsonb)
                        WHERE conversation_id = CAST(:conv_id AS uuid)
                    """
                    )
                    conn.execute(
                        stmt,
                        {
                            "participants": json.dumps(participants),
                            "conv_id": str(conv_id),
                        },
                    )
                    conn.commit()
                    logger.info(
                        f"Updated {folder_name}: {len(participants)} participants"
                    )

                updated += 1

            except Exception as e:
                logger.error(f"Error processing {folder_name}: {e}")
                errors += 1

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
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 0:
        print("Limit must be positive")
        sys.exit(1)  # Changed to sys.exit(1) for proper exit code

    backfill_participants(limit=args.limit, dry_run=args.dry_run)
