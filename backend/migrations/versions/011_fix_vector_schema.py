"""Fix vector schema and index (HalfVec + HNSW).

Revision ID: 011_fix_vector_schema
Revises: 010_add_fts_support
Create Date: 2025-12-17

Adds:
- Alters embedding column to halfvec(3840)
- Adds HNSW index for efficient high-dim search
"""
from alembic import op

# revision identifiers
revision = "011_fix_vector_schema"
down_revision = "010_add_fts_support"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Alter column type to halfvec
    # Note: Requires USING clause if casting from vector, but starting from null or compatible is easier.
    # Since we are essentially starting fresh or backfilling, we can cast.
    # If the column was vector(3840), we need to cast it.
    op.execute(
        """
        ALTER TABLE chunks
        ALTER COLUMN embedding TYPE halfvec(3840)
        USING embedding::halfvec(3840);
    """
    )

    # 2. Create HNSW index
    # HNSW on halfvec supports up to 4000 dims.
    # m=16, ef_construction=64 are reasonable defaults.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_chunks_embedding
        ON chunks USING hnsw (embedding halfvec_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding;")
    # Convert back to vector(3840)
    op.execute(
        """
        ALTER TABLE chunks
        ALTER COLUMN embedding TYPE vector(3840)
        USING embedding::vector(3840);
    """
    )
