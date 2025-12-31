import argparse
import logging
from collections import Counter
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from sqlalchemy import func, select

load_dotenv(".env")

from cortex.db.models import Chunk
from cortex.db.session import get_db_session
from cortex.intelligence.graph import GraphExtractor

# --- Rich Console Initialization ---
console = Console()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("schema_cli")


def cmd_schema_check(args: argparse.Namespace) -> None:
    """
    Analyzes a sample of conversations to report on the graph schema.
    """
    limit = args.limit
    logger.info(f"Fetching up to {limit} random conversations...")
    with get_db_session() as session:
        stmt = (
            select(Chunk.conversation_id)
            .where(Chunk.chunk_type == "message_body")
            .group_by(Chunk.conversation_id)
            .order_by(func.random())
            .limit(limit)
        )
        conv_ids = session.execute(stmt).scalars().all()

        texts = []
        for cid in conv_ids:
            chunks = (
                session.execute(
                    select(Chunk.text)
                    .where(
                        Chunk.conversation_id == cid,
                        Chunk.chunk_type == "message_body",
                        Chunk.text.isnot(None),
                    )
                    .order_by(Chunk.position)
                )
                .scalars()
                .all()
            )
            texts.append("\n".join(chunks))

    extractor = GraphExtractor()
    node_types: Counter[str] = Counter()
    relations: Counter[str] = Counter()

    logger.info("Extracting graphs sequentially...")
    for i, text in enumerate(texts):
        logger.info(f"Processing text {i + 1}/{limit} ({len(text)} chars)")
        try:
            G = extractor.extract_graph(text)
            for _, data in G.nodes(data=True):
                node_types[data.get("type", "UNKNOWN")] += 1
            for _, _, data in G.edges(data=True):
                relations[data.get("relation", "UNKNOWN")] += 1
        except Exception as e:
            logger.error(f"Failed: {e}")

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
