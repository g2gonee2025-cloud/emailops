import logging
import os
import time

from cortex.db.models import Chunk
from cortex.db.session import SessionLocal
from sqlalchemy import func, or_, select

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backfill")

BATCH_SIZE = 64
EMBED_DIM = 3840


def get_openai_client():
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed")
        return None

    # Internal service DNS for the embeddings-api service
    base_url = os.getenv(
        "DO_LLM_BASE_URL", "http://embeddings-api.emailops.svc.cluster.local/v1"
    )
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    api_key = os.getenv("DO_LLM_API_KEY", "EMPTY")

    logger.info(f"Connecting to Embedding API at {base_url}")
    return OpenAI(base_url=base_url, api_key=api_key)


def backfill_embeddings(
    tenant_id: str = "default", batch_size: int = BATCH_SIZE, limit: int = None
):
    client = get_openai_client()
    if not client:
        return

    model_name = os.getenv("EMBED_MODEL", "tencent/KaLM-Embedding-Gemma3-12B-2511")

    with SessionLocal() as session:
        # Check total to process
        total_stmt = (
            select(func.count(Chunk.chunk_id))
            .where(Chunk.embedding.is_(None))
            .where(
                or_(
                    Chunk.extra_data.is_(None),
                    Chunk.extra_data.op("->>")("skipped").is_(None),
                )
            )
        )
        total_missing = session.execute(total_stmt).scalar()
        logger.info(f"Total chunks missing embeddings: {total_missing}")

        if total_missing == 0:
            return

        processed = 0
        while True:
            # Check limit
            if limit and processed >= limit:
                break

            # Fetch batch
            stmt = (
                select(Chunk)
                .where(Chunk.embedding.is_(None))
                .where(
                    or_(
                        Chunk.extra_data.is_(None),
                        Chunk.extra_data.op("->>")("skipped").is_(None),
                    )
                )
                .limit(batch_size)
            )
            chunks = session.execute(stmt).scalars().all()

            if not chunks:
                break

            texts = [c.text for c in chunks]

            try:
                start_time = time.time()
                # logger.info(f"Processing chunks: {[str(c.chunk_id) for c in chunks]}")
                resp = client.embeddings.create(
                    input=texts, model=model_name, encoding_format="float"
                )
                duration = time.time() - start_time

                # Default OpenAI response structure
                embeddings = [data.embedding for data in resp.data]

                # Update DB
                for i, chunk in enumerate(chunks):
                    chunk.embedding = embeddings[i]

                session.commit()

                processed += len(chunks)
                logger.info(
                    f"Processed {processed}/{total_missing} chunks. Batch took {duration:.2f}s"
                )

            except Exception as e:
                # Fallback to serial processing if batch fails
                logger.warning(
                    f"Batch failed with error: {e}. Retrying chunks serially..."
                )

                for chunk in chunks:
                    try:
                        resp = client.embeddings.create(
                            input=[chunk.text],
                            model=model_name,
                            encoding_format="float",
                        )
                        chunk.embedding = resp.data[0].embedding
                        # Commit safe chunks immediately or in small groups?
                        # Committing individually to be safe
                        session.commit()
                        processed += 1

                    except Exception as inner_e:
                        logger.warning(
                            f"Skipping bad chunk {chunk.chunk_id}: {inner_e}"
                        )
                        # Mark as skipped
                        ed = dict(chunk.extra_data) if chunk.extra_data else {}
                        ed["skipped"] = True
                        ed["error"] = str(inner_e)[:200]
                        chunk.extra_data = ed
                        session.commit()

                continue

    logger.info("Backfill complete!")


if __name__ == "__main__":
    backfill_embeddings()
