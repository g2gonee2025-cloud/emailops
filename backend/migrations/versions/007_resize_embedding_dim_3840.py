"""Resize embedding vector to 3840 dims.

Revision ID: 007_resize_embedding_dim_3840
Revises: 006
Create Date: 2025-12-07
"""
from alembic import op
from psycopg2 import errors as psycopg_errors

revision = "007_resize_embedding_dim_3840"
down_revision = "006"
branch_labels = None
depends_on = None


HNSW_INDEX_SQL = """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS chunks_embedding_hnsw_idx 
    ON chunks 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
"""

IVFFLAT_INDEX_SQL = """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS chunks_embedding_hnsw_idx 
    ON chunks 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 200)
"""

LEGACY_INDEXES = [
    "DROP INDEX IF EXISTS idx_chunks_embedding_hnsw",
    "DROP INDEX IF EXISTS ix_chunks_embedding",
    "DROP INDEX IF EXISTS chunks_embedding_hnsw_idx",
]


def _recreate_vector_index() -> None:
    try:
        op.execute(HNSW_INDEX_SQL)
    except Exception as exc:  # pragma: no cover - migration-time only
        orig = getattr(exc, "orig", exc)
        if isinstance(orig, psycopg_errors.ProgramLimitExceeded):
            # Fallback when managed Postgres caps HNSW dimensions
            op.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw_idx")
            op.execute(IVFFLAT_INDEX_SQL)
            return
        raise


def upgrade() -> None:
    # Drop indexes that depend on embedding dimension
    for stmt in LEGACY_INDEXES:
        op.execute(stmt)

    # Resize the vector column to 3840 dims
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(3840)")

    # Recreate vector + FTS indexes
    ctx = op.get_context()
    with ctx.autocommit_block():
        _recreate_vector_index()
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS chunks_tsv_text_gin_idx
            ON chunks
            USING gin (tsv_text)
        """
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS chunks_tenant_id_idx
            ON chunks (tenant_id)
        """
        )


def downgrade() -> None:
    # Drop new indexes
    op.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw_idx")
    op.execute("DROP INDEX IF EXISTS chunks_tsv_text_gin_idx")
    op.execute("DROP INDEX IF EXISTS chunks_tenant_id_idx")

    # Revert to 3072 to match prior migration 005/006 defaults
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(3072)")

    # Restore legacy index if needed
    try:
        op.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks 
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """
        )
    except Exception as exc:  # pragma: no cover - migration-time only
        orig = getattr(exc, "orig", exc)
        if isinstance(orig, psycopg_errors.ProgramLimitExceeded):
            op.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
            op.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 200)
            """
            )
        else:
            raise
