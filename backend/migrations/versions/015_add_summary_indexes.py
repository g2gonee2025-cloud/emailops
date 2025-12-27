"""Add GIN indexes for conversation subject and summary.

Revision ID: 015_add_summary_indexes
Revises: 014_add_summary_text
Create Date: 2025-12-25

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "015_add_summary_indexes"
down_revision = "014_add_summary_text"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Index on subject (if not exists, but we'll add it safely)
    # Using 'english' config as defined in _FTS_CONFIG
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_conversations_subject_fts ON conversations "
        "USING GIN (to_tsvector('english', coalesce(subject, '')))"
    )

    # 2. Index on summary_text
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_conversations_summary_fts ON conversations "
        "USING GIN (to_tsvector('english', coalesce(summary_text, '')))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_conversations_summary_fts")
    op.execute("DROP INDEX IF EXISTS ix_conversations_subject_fts")
