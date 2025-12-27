"""Add summary_text column to conversations table.

Revision ID: 014
Revises: 013_add_graph_entities
Create Date: 2025-12-25

Stores AI-generated summary text directly on conversations for fast access.
The summary is also stored as an embedded chunk in the chunks table for semantic search.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "014_add_summary_text"
down_revision = "013_add_graph_entities"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add summary_text column to conversations table."""
    op.add_column(
        "conversations",
        sa.Column(
            "summary_text",
            sa.Text(),
            nullable=True,
            comment="AI-generated summary of the conversation",
        ),
    )


def downgrade() -> None:
    """Remove summary_text column."""
    op.drop_column("conversations", "summary_text")
