import argparse
import logging
from collections import Counter
from typing import Any

from cortex.db.models import Chunk
from cortex.db.session import get_db_session
from cortex.intelligence.graph import GraphExtractor
from dotenv import load_dotenv
from rich.console import Console
from sqlalchemy import func, select

# --- Rich Console Initialization ---
console = Console()
logger = logging.getLogger("schema_cli")

MESSAGE_BODY_CHUNK_TYPE = "message_body"
DEFAULT_LIMIT = 5


def cmd_schema_check(args: argparse.Namespace) -> None:
    """
    Analyzes a sample of conversations to report on the graph schema.
    """
    load_dotenv(".env")
    try:
        limit = int(getattr(args, "limit", DEFAULT_LIMIT))
    except (TypeError, ValueError):
        limit = DEFAULT_LIMIT

    if limit <= 0:
        logger.info("Limit must be positive; skipping schema check.")
        return

    node_types: Counter[str] = Counter()
    relations: Counter[str] = Counter()
    extractor = GraphExtractor()

    try:
        with get_db_session() as session:
            console.print(f"Fetching up to {limit} random conversations...")
            logger.info("Fetching up to %d conversations...", limit)
            conv_stmt = (
                select(Chunk.conversation_id)
                .where(Chunk.chunk_type == MESSAGE_BODY_CHUNK_TYPE)
                .group_by(Chunk.conversation_id)
                .order_by(Chunk.conversation_id)
                .limit(limit)
            )
            conv_ids = session.execute(conv_stmt).scalars().all()

            if not conv_ids:
                logger.info("No conversations matched the sampling window.")
                return

            console.print("Extracting graphs sequentially...")
            logger.info("Extracting graphs sequentially...")
            processed = 0

            def _process_text(conv_id: Any, text_value: str) -> None:
                nonlocal processed
                processed += 1
                console.print(
                    f"Processing text {processed}/{len(conv_ids)}",
                )
                logger.info(
                    "Processing text %d/%d (%d chars)",
                    processed,
                    len(conv_ids),
                    len(text_value),
                )
                try:
                    G = extractor.extract_graph(text_value)
                    for _, data in G.nodes(data=True):
                        node_types[data.get("type", "UNKNOWN")] += 1
                    for _, _, data in G.edges(data=True):
                        relations[data.get("relation", "UNKNOWN")] += 1
                except Exception:
                    logger.exception(
                        "Failed to extract graph for conversation %s", conv_id
                    )

            for conv_id in conv_ids:
                chunk_stmt = (
                    select(Chunk.text)
                    .where(
                        Chunk.conversation_id == conv_id,
                        Chunk.chunk_type == MESSAGE_BODY_CHUNK_TYPE,
                        Chunk.text.isnot(None),
                    )
                    .order_by(Chunk.position)
                )
                texts = session.execute(chunk_stmt).scalars().all()
                combined = "\n".join(
                    text for text in texts if isinstance(text, str) and text
                )
                if combined:
                    _process_text(conv_id, combined)
    except Exception:
        logger.exception("Schema check failed during database operations.")
        return

    console.print("\n=== FINAL SCHEMA REPORT ===")
    console.print("Top Node Types:", node_types.most_common())
    console.print("Top Relations:", relations.most_common())


def setup_schema_parser(subparsers: Any) -> None:
    """Add schema subcommands to the CLI parser."""
    schema_parser = subparsers.add_parser(
        "schema",
        help="Schema management commands",
        description="Manage the Cortex schema: view stats, run analysis.",
    )

    schema_subparsers = schema_parser.add_subparsers(
        dest="schema_command", title="Schema Commands"
    )

    # schema check
    check_parser = schema_subparsers.add_parser(
        "check",
        help="Analyze a sample of conversations",
        description="Analyzes a sample of conversations to report on the graph schema.",
    )
    check_parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=5,
        help="Number of random conversations to analyze.",
    )
    check_parser.set_defaults(func=cmd_schema_check)

    def _default_schema_handler(args: argparse.Namespace) -> None:
        if not args.schema_command:
            args.limit = getattr(args, "limit", 5)
            cmd_schema_check(args)

    schema_parser.set_defaults(func=_default_schema_handler)
