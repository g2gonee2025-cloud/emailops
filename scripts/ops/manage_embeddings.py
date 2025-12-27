#!/usr/bin/env python3
"""
Embeddings Management CLI.
Tools to fix null embeddings or force full re-embedding.
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add backend/src to path
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir / "backend" / "src"))

from cortex.config.loader import get_config
from cortex.embeddings.client import get_embedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("embed_ops")


def fix_nulls(args):
    """Scan and patch chunks with NULL embeddings."""
    logger.info("Scanning for NULL embeddings...")
    config = get_config()
    db_url = config.database.url

    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    count = session.execute(
        text("SELECT count(*) FROM chunks WHERE embedding IS NULL")
    ).scalar()
    logger.info(f"Found {count} chunks with NULL embeddings.")

    if count == 0:
        return

    limit = args.batch_size
    workers = getattr(args, "workers", 5)

    # Pre-fetch all IDs to distribute work
    # (If dataset is huge, we might want pagination, but for 3k chunks this is fine)
    rows = session.execute(
        text("SELECT chunk_id, text FROM chunks WHERE embedding IS NULL")
    ).fetchall()

    total = len(rows)
    logger.info(f"Loaded {total} chunks to fix. Workers: {workers}")

    # Chunking utility
    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    batches = list(chunk_list(rows, limit))

    total_fixed = 0
    errors = 0

    def _process_batch(batch_rows):
        # New session per thread recommended
        _engine = create_engine(db_url)
        with _engine.connect() as conn:
            local_fixed = 0
            for r in batch_rows:
                if not r.text:
                    continue
                try:
                    # This calls the LLM client which handles its own HTTP connection
                    emb = get_embedding(r.text)
                    conn.execute(
                        text(
                            "UPDATE chunks SET embedding = CAST(:emb AS halfvec(3840)) WHERE chunk_id = :id"
                        ),
                        {"emb": str(emb), "id": r.chunk_id},
                    )
                    local_fixed += 1
                except Exception as e:
                    logger.error(f"Failed chunk {r.chunk_id}: {e}")
            conn.commit()
            return local_fixed

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_batch, b) for b in batches]

        for i, future in enumerate(as_completed(futures)):
            try:
                fixed = future.result()
                total_fixed += fixed
                if i % 5 == 0:
                    logger.info(f"Progress: {total_fixed}/{total} fixed")
            except Exception as e:
                logger.error(f"Batch failed: {e}")
                errors += 1

    logger.info(f"Done. Fixed: {total_fixed}, Errors: {errors}")


def force_reembed(args):
    """Force re-embedding of ALL chunks using parallel workers."""
    config = get_config()
    db_url = config.database.url

    # Use environment overrides or config
    embed_api = args.api_url or "http://localhost:8081/v1"
    embed_model = args.model or "tencent/KaLM-Embedding-Gemma3-12B-2511"
    batch_size = args.batch_size
    workers = args.workers

    logger.info(
        f"Forcing Re-embed: API={embed_api}, Model={embed_model}, Workers={workers}"
    )

    engine = create_engine(db_url)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT chunk_id, text FROM chunks ORDER BY chunk_id")
        ).fetchall()

    total = len(rows)
    logger.info(f"Total chunks: {total}")

    # Batches
    batches = []
    for i in range(0, total, batch_size):
        chunk_rows = rows[i : i + batch_size]
        cids = [str(r[0]) for r in chunk_rows]
        txts = [r[1] for r in chunk_rows]
        batches.append((cids, txts))

    logger.info(f"Created {len(batches)} batches.")

    processed = 0
    errors = 0

    def _process(batch_data):
        cids, txts = batch_data
        client = OpenAI(base_url=embed_api, api_key="dummy")
        eng = create_engine(db_url)
        try:
            resp = client.embeddings.create(input=txts, model=embed_model)
            embs = [e.embedding for e in resp.data]

            with eng.connect() as conn:
                for cid, emb in zip(cids, embs, strict=False):
                    conn.execute(
                        text(
                            "UPDATE chunks SET embedding = CAST(:emb AS halfvec(3840)) WHERE chunk_id = :cid"
                        ),
                        {"emb": str(emb), "cid": cid},
                    )
                conn.commit()
            return len(cids), None
        except Exception as e:
            return 0, str(e)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, b): i for i, b in enumerate(batches)}

        for future in as_completed(futures):
            cnt, err = future.result()
            if err:
                errors += 1
                logger.error(f"Batch error: {err}")
            else:
                processed += cnt
                if processed % 1000 == 0:
                    print(
                        f"Progress: {processed}/{total} ({100 * processed / total:.1f}%)"
                    )

    logger.info(f"Done. Re-embedded {processed} chunks. Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(description="Embeddings Management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # fix
    p_fix = subparsers.add_parser("fix", help="Fix NULL embeddings")
    p_fix.add_argument("--batch-size", type=int, default=50)
    p_fix.add_argument(
        "--workers", type=int, default=5, help="Number of parallel workers"
    )

    # force
    p_force = subparsers.add_parser("force", help="Force re-embed EVERYTHING")
    p_force.add_argument("--batch-size", type=int, default=200)
    p_force.add_argument("--workers", type=int, default=4)
    p_force.add_argument("--api-url", help="Override standard API URL")
    p_force.add_argument("--model", help="Override embedding model")

    args = parser.parse_args()

    if args.command == "fix":
        fix_nulls(args)
    elif args.command == "force":
        force_reembed(args)


if __name__ == "__main__":
    main()
