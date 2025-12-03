"""Update embedding dimension to 3072 for Gemini.

Revision ID: 005
Revises: ec7386d2401e
Create Date: 2025-12-03 12:00:00.000000

"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "005"
down_revision = "ec7386d2401e"
branch_labels = None
depends_on = None


def upgrade():
    # Drop the index first as it depends on the column type/dimension
    # We try dropping both potential names to be safe
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding")

    # Alter the column type
    # This assumes the table is empty or the user is okay with the schema change.
    # If there is data, this ALTER might fail if the existing vectors are not compatible,
    # but typically changing vector dimension requires re-embedding anyway.
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(3072)")

    # Recreate the index with the new dimension
    op.execute(
        """
        CREATE INDEX idx_chunks_embedding_hnsw ON chunks 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding")

    # Revert to 1536
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1536)")

    op.execute(
        """
        CREATE INDEX idx_chunks_embedding_hnsw ON chunks 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """
    )
