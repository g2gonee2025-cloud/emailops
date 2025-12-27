import logging
from collections import Counter

from dotenv import load_dotenv
from sqlalchemy import func, select

load_dotenv(".env")

import builtins

from cortex.db.models import Chunk  # noqa: E402
from cortex.db.session import get_db_session  # noqa: E402
from cortex.intelligence.graph import GraphExtractor  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quick_schema")


def _print_to_logger(*args: object, **kwargs: object) -> None:
    msg = " ".join(str(a) for a in args)
    logger.info(msg)


builtins.print = _print_to_logger


def run_quick_analysis() -> None:
    logger.info("Fetching up to 5 random conversations...")
    with get_db_session() as session:
        stmt = (
            select(Chunk.conversation_id)
            .where(Chunk.chunk_type == "message_body")
            .group_by(Chunk.conversation_id)
            .order_by(func.random())
            .limit(5)
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
    node_types = Counter()
    relations = Counter()

    logger.info("Extracting graphs sequentially...")
    for i, text in enumerate(texts):
        logger.info(f"Processing text {i + 1}/5 ({len(text)} chars)")
        try:
            G = extractor.extract_graph(text)
            for _, data in G.nodes(data=True):
                node_types[data.get("type", "UNKNOWN")] += 1
            for _, _, data in G.edges(data=True):
                relations[data.get("relation", "UNKNOWN")] += 1
        except Exception as e:
            logger.error(f"Failed: {e}")

    print("\n=== FINAL SCHEMA REPORT ===")
    print("Top Node Types:", node_types.most_common())
    print("Top Relations:", relations.most_common())


if __name__ == "__main__":
    run_quick_analysis()
