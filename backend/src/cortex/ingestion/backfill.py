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
        "DO_LLM_BASE_URL",
        "http://embeddings-api.emailops.svc.cluster.local/v1",  # NOSONAR
    )
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    api_key = os.getenv("DO_LLM_API_KEY", "EMPTY")

    logger.info(f"Connecting to Embedding API at {base_url}")
    return OpenAI(base_url=base_url, api_key=api_key)


def get_gguf_provider():
    """Get GGUF provider for CPU-based embeddings."""
    try:
        from cortex.llm.gguf_provider import GGUFProvider

        provider = GGUFProvider()
        if provider.is_available():
            logger.info("Using GGUF CPU embedding via llama-server")
            return provider
        else:
            logger.error("llama-server not available at %s", provider._endpoint)
            return None
    except Exception as e:
        logger.error(f"Failed to initialize GGUF provider: {e}")
        return None


def backfill_embeddings(
    tenant_id: str = "default", batch_size: int = BATCH_SIZE, limit: int = None
):
    embed_mode = os.getenv("OUTLOOKCORTEX_EMBED_MODE", os.getenv("EMBED_MODE", "gpu"))
    use_cpu = embed_mode.lower() == "cpu"

    if use_cpu:
        provider = get_gguf_provider()
        if not provider:
            logger.error("CPU mode requested but GGUF provider not available")
            return {"processed": 0, "failed": 0, "error": "GGUF provider not available"}
        client = None
    else:
        client = get_openai_client()
        if not client:
            return {"processed": 0, "failed": 0, "error": "OpenAI client not available"}
        provider = None

    model_name = os.getenv("EMBED_MODEL", "tencent/KaLM-Embedding-Gemma3-12B-2511")

    with SessionLocal() as session:
        total_missing = _get_missing_embeddings_count(session)
        logger.info(f"Total chunks missing embeddings: {total_missing}")

        if total_missing == 0:
            return {"processed": 0, "failed": 0}

        processed = 0
        failed = 0
        while not (limit and processed >= limit):
            chunks = _get_chunks_without_embeddings(session, batch_size)
            if not chunks:
                break

            texts = [c.text for c in chunks]

            if use_cpu:
                batch_processed, batch_failed = _process_embedding_batch_cpu(
                    provider, session, chunks, texts
                )
            else:
                batch_processed = _process_embedding_batch(
                    client, session, chunks, texts, model_name
                )
                batch_failed = len(chunks) - batch_processed

            processed += batch_processed
            failed += batch_failed
            logger.info(
                f"Processed {processed}/{total_missing} chunks (failed: {failed})."
            )

    logger.info("Backfill complete!")
    return {"processed": processed, "failed": failed}


def _get_missing_embeddings_count(session) -> int:
    """Get count of chunks missing embeddings."""
    stmt = (
        select(func.count(Chunk.chunk_id))
        .where(Chunk.embedding.is_(None))
        .where(
            or_(
                Chunk.extra_data.is_(None),
                Chunk.extra_data.op("->>")("skipped").is_(None),
            )
        )
    )
    return session.execute(stmt).scalar()


def _get_chunks_without_embeddings(session, batch_size: int) -> list:
    """Fetch a batch of chunks without embeddings."""
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
    return session.execute(stmt).scalars().all()


def _process_embedding_batch(client, session, chunks, texts, model_name) -> int:
    """Process a batch of embeddings, falling back to serial on error."""
    try:
        start_time = time.time()
        resp = client.embeddings.create(
            input=texts, model=model_name, encoding_format="float"
        )
        duration = time.time() - start_time

        embeddings = [data.embedding for data in resp.data]
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]

        session.commit()
        logger.info(f"Batch took {duration:.2f}s")
        return len(chunks)

    except Exception as e:
        logger.warning(f"Batch failed with error: {e}. Retrying chunks serially...")
        return _process_chunks_serially(client, session, chunks, model_name)


def _process_chunks_serially(client, session, chunks, model_name) -> int:
    """Process chunks one by one after batch failure."""
    processed = 0
    for chunk in chunks:
        try:
            resp = client.embeddings.create(
                input=[chunk.text], model=model_name, encoding_format="float"
            )
            chunk.embedding = resp.data[0].embedding
            session.commit()
            processed += 1
        except Exception as inner_e:
            logger.warning(f"Skipping bad chunk {chunk.chunk_id}: {inner_e}")
            ed = dict(chunk.extra_data) if chunk.extra_data else {}
            ed["skipped"] = True
            ed["error"] = str(inner_e)[:200]
            chunk.extra_data = ed
            session.commit()
    return processed


def _process_embedding_batch_cpu(provider, session, chunks, texts) -> tuple[int, int]:
    """Process a batch of embeddings using CPU GGUF provider."""
    import numpy as np

    start_time = time.time()
    processed = 0
    failed = 0

    try:
        # GGUF provider handles batching internally
        embeddings = provider.embed(texts)

        for i, chunk in enumerate(chunks):
            try:
                emb = (
                    embeddings[i].tolist()
                    if isinstance(embeddings[i], np.ndarray)
                    else embeddings[i]
                )

                # Validate embedding
                if emb is None or len(emb) == 0 or all(v == 0 for v in emb[:10]):
                    logger.warning(
                        f"Empty embedding for chunk {chunk.chunk_id}, skipping"
                    )
                    ed = dict(chunk.extra_data) if chunk.extra_data else {}
                    ed["skipped"] = True
                    ed["error"] = "empty_embedding"
                    chunk.extra_data = ed
                    failed += 1
                else:
                    chunk.embedding = emb
                    processed += 1
            except Exception as e:
                logger.warning(
                    f"Failed to save embedding for chunk {chunk.chunk_id}: {e}"
                )
                ed = dict(chunk.extra_data) if chunk.extra_data else {}
                ed["skipped"] = True
                ed["error"] = str(e)[:200]
                chunk.extra_data = ed
                failed += 1

        session.commit()
        duration = time.time() - start_time
        logger.info(f"CPU batch took {duration:.2f}s ({processed} ok, {failed} failed)")

    except Exception as e:
        logger.error(f"CPU batch embedding failed: {e}")
        failed = len(chunks)

    return processed, failed


if __name__ == "__main__":
    backfill_embeddings()
