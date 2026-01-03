"""
A script to find and fix NULL embeddings in the chunks table.

It iterates through all chunks with NULL embeddings, generates a new embedding using
an external API, and updates the database record. It is designed to be run as a
module from the project root directory.

Example:
    python -m scripts.ops.fix_nulls
"""

import os

from openai import OpenAI
from sqlalchemy import create_engine, text

EMBED_API = os.getenv("EMBED_API_URL", "https://localhost:8081/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "tencent/KaLM-Embedding-Gemma3-12B-2511")

from cortex.config.loader import get_config


def main():
    config = get_config()
    client = OpenAI(base_url=EMBED_API, api_key="dummy", timeout=30.0)
    engine = create_engine(config.database.url)

    with engine.connect() as conn:
        with conn.begin():
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

            embed_dim = config.embedding.output_dimensionality
            update_query = text(
                f"""
                UPDATE chunks SET embedding = CAST(:emb AS halfvec({embed_dim}))
                WHERE chunk_id = :cid
                """
            )
            for row in rows:
                cid, txt = row[0], row[1]
                if not txt:
                    print(f"  Skipping chunk {str(cid)[:8]} due to empty text.")
                    continue

                print(f"Processing chunk {str(cid)[:8]}... text length: {len(txt)}")

                try:
                    resp = client.embeddings.create(input=[txt], model=EMBED_MODEL)
                    if not resp.data or not resp.data[0].embedding:
                        print(
                            f"  Error: Received no embedding for chunk {str(cid)[:8]}"
                        )
                        continue
                    emb = resp.data[0].embedding

                    conn.execute(
                        update_query,
                        {"emb": emb, "cid": cid},
                    )
                    print("  Fixed!")
                except OpenAI.APIError as e:
                    print(f"  API Error for chunk {str(cid)[:8]}: {e}")
                except Exception as e:
                    print(
                        f"  An unexpected error occurred for chunk {str(cid)[:8]}: {e}"
                    )
                    raise

        # Verify
        remaining = conn.execute(
            text("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")
        ).scalar()
        print(f"Remaining NULL embeddings: {remaining}")


if __name__ == "__main__":
    main()
