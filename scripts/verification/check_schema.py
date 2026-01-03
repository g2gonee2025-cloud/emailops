from cortex.config.loader import get_config
from sqlalchemy import create_engine
from sqlalchemy import inspect as sqlalchemy_inspect
from sqlalchemy import text


def check_schema():
    config = get_config()
    # Strip sslmode for inspection if needed, but sqlalchemy handles it
    engine = create_engine(config.database.url)
    try:
        inspector = sqlalchemy_inspect(engine)
        schema = inspector.default_schema_name
        table_name = "chunks"

        # 1. Check Table Exists
        if not inspector.has_table(table_name, schema=schema):
            print(f"FAIL: Table '{table_name}' does not exist in schema '{schema}'")
            return

        # 2. Check Columns
        columns = {
            c["name"]: c for c in inspector.get_columns(table_name, schema=schema)
        }

        # Check embedding dim
        # Vector type might show up as NullType or UserDefinedType depending on reflection
        emb_col = columns.get("embedding")
        if not emb_col:
            print("FAIL: Column 'embedding' missing")
        else:
            # It's hard to check exact dim via standard reflection without pgvector support loaded in reflection
            # We can check specific type string if available
            print(f"Embedding column type: {emb_col['type']}")

        required_cols = ["tsv_text", "chunk_type", "tenant_id"]
        for col in required_cols:
            if col not in columns:
                print(f"FAIL: Column '{col}' missing")
            else:
                print(f"OK: Column '{col}' exists")

        # 3. Check Indexes
        indexes = inspector.get_indexes(table_name, schema=schema)
        index_names = [i["name"] for i in indexes]

        print("Indexes found:", index_names)

        if "ix_chunks_tsv_text" in index_names:
            print("OK: FTS index exists")
        else:
            print("FAIL: FTS index missing")

        if "ix_chunks_embedding" in index_names:
            print("OK: Vector index exists")
            # Verify it is hnsw
            with engine.connect() as conn:
                stmt = text(
                    "SELECT indexdef FROM pg_indexes WHERE schemaname = :schema AND tablename = :table AND indexname = 'ix_chunks_embedding'"
                ).bindparams(schema=schema, table=table_name)
                defn = conn.execute(stmt).scalar()
                print(f"Index Def: {defn}")
                if defn and "hnsw" in defn.lower():
                    print("OK: Index is HNSW")
                else:
                    print("WARN: Index might not be HNSW")
        else:
            print("FAIL: Vector index missing")
    finally:
        engine.dispose()


if __name__ == "__main__":
    try:
        check_schema()
    except Exception as e:
        print(f"Error verifying schema: {e}")
        raise
