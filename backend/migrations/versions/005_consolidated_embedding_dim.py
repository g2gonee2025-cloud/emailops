"""Consolidated embedding dimension migration.

This migration replaces the following chained migrations:
- 005_update_embedding_dim.py (1536 → 3072)
- 007_resize_embedding_dim_3840.py (3072 → 3840)
- 008_resize_embedding_dim_1024.py (3840 → 1024)
- 009_resize_embedding_dim_3840.py (1024 → 3840)

Net effect: Sets embedding dimension to 3840 (for KaLM embeddings).

Revision ID: 005_consolidated_embedding_dim
Revises: ec7386d2401e
Create Date: 2025-12-12
"""
from alembic import op
from psycopg2 import errors as psycopg_errors

revision = "005_consolidated_embedding_dim"
down_revision = "ec7386d2401e"
branch_labels = None
depends_on = None

# Target embedding dimension
TARGET_DIM = 3840
ORIGINAL_DIM = 1536

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
    """Create vector index with HNSW, falling back to IVFFlat if dimension limit exceeded."""
    try:
        op.execute(HNSW_INDEX_SQL)
    except Exception as exc:
        orig = getattr(exc, "orig", exc)
        if isinstance(orig, psycopg_errors.ProgramLimitExceeded):
            # Fallback when managed Postgres caps HNSW dimensions
            op.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw_idx")
            try:
                op.execute(IVFFLAT_INDEX_SQL)
            except Exception as ivf_exc:
                ivf_orig = getattr(ivf_exc, "orig", ivf_exc)
                if isinstance(ivf_orig, psycopg_errors.ProgramLimitExceeded):
                    print(
                        "WARNING: Vector dimensions exceed limit for both HNSW and IVFFlat. Skipping vector index."
                    )
                    return
                raise
            return
        raise


def upgrade() -> None:
    """Upgrade embedding column to target dimension."""
    # Drop indexes that depend on embedding dimension
    for stmt in LEGACY_INDEXES:
        op.execute(stmt)

    # Resize the vector column to target dimension
    op.execute(f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({TARGET_DIM})")

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
    """Revert to original embedding dimension."""
    # Drop new indexes
    op.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw_idx")
    op.execute("DROP INDEX IF EXISTS chunks_tsv_text_gin_idx")
    op.execute("DROP INDEX IF EXISTS chunks_tenant_id_idx")

    # Revert to original dimension
    op.execute(f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({ORIGINAL_DIM})")

    # Restore legacy HNSW index
    try:
        op.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """
        )
    except Exception as exc:
        orig = getattr(exc, "orig", exc)
        if isinstance(orig, psycopg_errors.ProgramLimitExceeded):
            op.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
            op.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 200)
            """
            )
        else:
            raise
