"""
Verification script for vector search explains.
"""

import logging
import os

from cortex.config.loader import get_config
from cortex.ingestion.backfill import get_openai_client
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_EMBED_DIM = 768  # Fallback
DEFAULT_MODEL = "tencent/KaLM-Embedding-Gemma3-12B-2511"


def test_search() -> None:
    try:
        client = get_openai_client()
        if not client:
            logger.error("No OpenAI client avail")
            return

        config = get_config()
        embed_dim = (
            config.embedding.output_dimensionality
            if config.embedding
            else DEFAULT_EMBED_DIM
        )

        query_text = "project update"
        model_name = os.getenv("EMBED_MODEL", DEFAULT_MODEL)

        logger.info("Generating embedding for '%s'...", query_text)
        try:
            resp = client.embeddings.create(input=[query_text], model=model_name)
            if not resp.data:
                logger.error("No embedding data returned")
                return
            query_vec = resp.data[0].embedding
        except Exception as e:
            logger.error("Embedding generation failed: %s", e)
            return

        # Use efficient engine usage
        engine = create_engine(config.database.url)

        try:
            with engine.connect() as conn:
                logger.info("Running Vector Search with EXPLAIN ANALYZE...")
                # Use f-string for dimension in type cast (safe constant)
                # CAST(:query_vec AS halfvec(N))
                stmt = text(
                    f"""
                    EXPLAIN ANALYZE
                    SELECT chunk_id, embedding <=> CAST(:query_vec AS halfvec({embed_dim})) as distance
                    FROM chunks
                    ORDER BY distance ASC
                    LIMIT 5
                """
                )

                result = conn.execute(stmt, {"query_vec": query_vec}).fetchall()

                for row in result:
                    line = row[0]
                    if (
                        "Scan" in line
                        or "Index" in line
                        or "Execution" in line
                        or "Filter" in line
                    ):
                        logger.info(line)
        except Exception as e:
            logger.error("Database query failed: %s", e)

    except Exception as e:
        logger.error("Script failed: %s", e)


if __name__ == "__main__":
    test_search()
