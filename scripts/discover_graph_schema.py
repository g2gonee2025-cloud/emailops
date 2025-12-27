import concurrent.futures
import logging
import random
from collections import Counter

from dotenv import load_dotenv
from sqlalchemy import func, select

load_dotenv(".env")

from cortex.db.models import Chunk  # noqa: E402
from cortex.db.session import get_db_session  # noqa: E402
from cortex.intelligence.graph import GraphExtractor  # noqa: E402

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("graph_discovery")


def get_sample_texts(sample_size: int = 100) -> list[str]:
    """Fetch the largest 100 conversations (by text size) from the DB."""
    logger.info(f"Fetching {sample_size} largest conversations...")
    with get_db_session() as session:
        # 1. Find IDs of largest conversations by summing chunk lengths
        # Using a subquery/CTE approach or grouping
        # 1 Switch to random sampling to avoid OOM/Timeout on massive files
        stmt = (
            select(Chunk.conversation_id)
            .where(Chunk.chunk_type == "message_body")
            .group_by(Chunk.conversation_id)
            .order_by(func.random())
            .limit(20)  # Reduced to 20 for speed/stability
        )
        conv_ids = session.execute(stmt).scalars().all()

        texts = []
        for cid in conv_ids:
            # 2. Reconstruct full text for each conversation (or at least the first 20k chars)
            chunks_stmt = (
                select(Chunk.text)
                .where(Chunk.conversation_id == cid, Chunk.chunk_type == "message_body")
                .order_by(Chunk.position)
            )
            chunk_texts = session.execute(chunks_stmt).scalars().all()
            full_text = "\n".join(chunk_texts)
            texts.append(full_text)

        return texts


def analyze_schema(texts: list[str]):
    extractor = GraphExtractor()

    node_types = Counter()
    relations = Counter()
    entity_names = []

    logger.info(f"Starting graph extraction on {len(texts)} texts with ThreadPool...")

    def process_text(text: str) -> dict:
        try:
            G = extractor.extract_graph(text)
            n_types = [data.get("type", "UNKNOWN") for _, data in G.nodes(data=True)]
            rels = [
                data.get("relation", "UNKNOWN") for _, _, data in G.edges(data=True)
            ]
            names = list(G.nodes())
            return {"types": n_types, "relations": rels, "names": names}
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {"types": [], "relations": [], "names": []}

    # Parallelize LLM calls
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(process_text, texts))
    except Exception as e:
        logger.critical(f"ThreadPoolExecutor failed: {e}")
        return node_types, relations

    # Aggregate results
    for res in results:
        node_types.update(res["types"])
        relations.update(res["relations"])
        entity_names.extend(res["names"])

    logger.info("\n=== SCHEMA DISCOVERY REPORT ===")

    logger.info("\nTOP NODE TYPES:")
    for k, v in node_types.most_common(20):
        logger.info(f"  {k}: {v}")

    logger.info("\nTOP RELATIONSHIPS:")
    for k, v in relations.most_common(20):
        logger.info(f"  {k}: {v}")

    # Heuristic check for "Missed" standard types or "Hallucinated" weird ones
    logger.info("\nSample Entities (Random 10):")
    for name in random.sample(entity_names, min(10, len(entity_names))):
        logger.info(f"  - {name}")

    return node_types, relations


if __name__ == "__main__":
    texts = get_sample_texts(100)
    if texts:
        analyze_schema(texts)
    else:
        logger.error("No texts found in DB.")
