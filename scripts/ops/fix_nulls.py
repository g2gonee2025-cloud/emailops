"""Fix the 3 remaining NULL embeddings."""

import os

from openai import OpenAI
from sqlalchemy import create_engine, text

EMBED_API = os.getenv("EMBED_API_URL", "http://localhost:8081/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "tencent/KaLM-Embedding-Gemma3-12B-2511")

import sys

sys.path.insert(0, "backend/src")
from cortex.config.loader import get_config


def main():
    client = OpenAI(base_url=EMBED_API, api_key="dummy")
    engine = create_engine(get_config().database.url)

    with engine.connect() as conn:
        # Find NULL embedding chunks
        rows = conn.execute(
            text(
                """
            SELECT chunk_id, text FROM chunks WHERE embedding IS NULL
        """
            )
        ).fetchall()

        print(f"Found {len(rows)} chunks with NULL embedding")

        if not rows:
            print("Nothing to fix!")
            return

        for row in rows:
            cid, txt = str(row[0]), row[1]
            print(f"Processing chunk {cid[:8]}... text length: {len(txt)}")

            try:
                resp = client.embeddings.create(input=[txt], model=EMBED_MODEL)
                emb = resp.data[0].embedding
                emb_str = str(emb)

                conn.execute(
                    text(
                        """
                    UPDATE chunks SET embedding = CAST(:emb AS halfvec(3840))
                    WHERE chunk_id = :cid
                """
                    ),
                    {"emb": emb_str, "cid": cid},
                )
                conn.commit()
                print("  Fixed!")
            except Exception as e:
                print(f"  Error: {e}")

        # Verify
        remaining = conn.execute(
            text("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")
        ).scalar()
        print(f"Remaining NULL embeddings: {remaining}")


if __name__ == "__main__":
    main()
