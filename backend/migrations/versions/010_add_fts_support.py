"""Add FTS support to chunks table.

Revision ID: 010_add_fts_support
Revises: None
Create Date: 2024-12-16

Adds:
- tenant_id column (denormalized for efficient filtering)
- chunk_type column (for quoted history detection)
- tsv_text tsvector column (for FTS)
- GIN index on tsv_text
- Trigger to auto-update tsv_text
"""
from alembic import op

# revision identifiers
revision = "010_add_fts_support"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add tenant_id column
    op.execute(
        """
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(64);
    """
    )

    # 2. Add chunk_type column
    op.execute(
        """
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_type VARCHAR(32) DEFAULT 'message_body';
    """
    )

    # 3. Add tsvector column for FTS
    op.execute(
        """
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv_text tsvector;
    """
    )

    # 4. Populate tenant_id from conversations
    op.execute(
        """
        UPDATE chunks c
        SET tenant_id = conv.tenant_id
        FROM conversations conv
        WHERE c.conversation_id = conv.conversation_id
          AND c.tenant_id IS NULL;
    """
    )

    # 5. Populate chunk_type from extra_data if available
    op.execute(
        """
        UPDATE chunks
        SET chunk_type = COALESCE(extra_data->>'chunk_type', 'message_body')
        WHERE chunk_type IS NULL OR chunk_type = 'message_body';
    """
    )

    # 6. Populate tsv_text from text column
    op.execute(
        """
        UPDATE chunks
        SET tsv_text = to_tsvector('english', COALESCE(text, ''))
        WHERE tsv_text IS NULL;
    """
    )

    # 7. Create GIN index for FTS
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_chunks_tsv_text ON chunks USING GIN (tsv_text);
    """
    )

    # 8. Create index on tenant_id
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_chunks_tenant_id ON chunks (tenant_id);
    """
    )

    # 9. Create trigger to auto-update tsv_text
    op.execute(
        """
        CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
        BEGIN
            NEW.tsv_text := to_tsvector('english', COALESCE(NEW.text, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """
    )

    op.execute(
        """
        DROP TRIGGER IF EXISTS trig_chunks_tsv ON chunks;
    """
    )

    op.execute(
        """
        CREATE TRIGGER trig_chunks_tsv
            BEFORE INSERT OR UPDATE OF text ON chunks
            FOR EACH ROW
            EXECUTE FUNCTION chunks_tsv_trigger();
    """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trig_chunks_tsv ON chunks;")
    op.execute("DROP FUNCTION IF EXISTS chunks_tsv_trigger();")
    op.execute("DROP INDEX IF EXISTS ix_chunks_tsv_text;")
    op.execute("DROP INDEX IF EXISTS ix_chunks_tenant_id;")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS tsv_text;")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS chunk_type;")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS tenant_id;")
