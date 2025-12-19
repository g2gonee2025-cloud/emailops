import os

from cortex.config.loader import get_config
from cortex.ingestion.backfill import get_openai_client
from sqlalchemy import create_engine, text


def test_search():
    client = get_openai_client()
    if not client:
        print("No OpenAI client avail")
        return

    # Generate dummy embedding
    query_text = "project update"
    model_name = os.getenv("EMBED_MODEL", "tencent/KaLM-Embedding-Gemma3-12B-2511")

    print(f"Generating embedding for '{query_text}'...")
    resp = client.embeddings.create(input=[query_text], model=model_name)
    query_vec = resp.data[0].embedding

    # Run Search
    config = get_config()
    engine = create_engine(config.database.url)

    with engine.connect() as conn:
        print("Running Vector Search with EXPLAIN ANALYZE...")
        stmt = text(
            """
            EXPLAIN ANALYZE
            SELECT chunk_id, embedding <=> CAST(:query_vec AS halfvec(3840)) as distance
            FROM chunks
            ORDER BY distance ASC
            LIMIT 5
        """
        )
        # We need to pass the vector as a string format or list, depending on driver.
        # pgvector-python usually handles list -> vector string casting if param is set correctly.
        # But for raw text sql with cast, passing string `[1,2...]` is safest.

        vec_str = str(query_vec)

        result = conn.execute(stmt, {"query_vec": vec_str}).fetchall()

        for row in result:
            line = row[0]
            if "Scan" in line or "Index" in line or "Execution" in line:
                print(line)


if __name__ == "__main__":
    test_search()
