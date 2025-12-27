"""Update audit_log schema for enhanced tracking.

Revision ID: 012_update_audit_log
Revises: 011_fix_vector_schema
Create Date: 2025-12-23 07:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "012_update_audit_log"
down_revision: Union[str, None] = "011_fix_vector_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Update audit_log table to new schema."""
    # Rename existing columns
    op.alter_column("audit_log", "timestamp", new_column_name="ts")
    op.alter_column("audit_log", "actor", new_column_name="user_or_agent")

    # Increase user_or_agent column size (was likely text, now String(256))
    op.alter_column(
        "audit_log", "user_or_agent", type_=sa.String(256), existing_type=sa.Text()
    )

    # Add new columns
    op.add_column("audit_log", sa.Column("input_hash", sa.String(64), nullable=True))
    op.add_column("audit_log", sa.Column("output_hash", sa.String(64), nullable=True))
    op.add_column(
        "audit_log",
        sa.Column("policy_decisions", sa.dialects.postgresql.JSONB, nullable=True),
    )
    op.add_column(
        "audit_log",
        sa.Column("risk_level", sa.String(32), server_default="low", nullable=False),
    )
    op.add_column(
        "audit_log", sa.Column("metadata_", sa.dialects.postgresql.JSONB, nullable=True)
    )

    # Rename details to metadata_ if it exists (migrate old data)
    # Check if 'details' column exists before migrating
    conn = op.get_bind()
    result = conn.execute(
        sa.text(
            "SELECT column_name FROM information_schema.columns WHERE table_name='audit_log' AND column_name='details'"
        )
    )
    if result.fetchone():
        # Migrate data from details to metadata_
        op.execute("UPDATE audit_log SET metadata_ = details WHERE details IS NOT NULL")
        op.drop_column("audit_log", "details")


def downgrade() -> None:
    """Revert to original schema."""
    # Re-add details column
    op.add_column(
        "audit_log", sa.Column("details", sa.dialects.postgresql.JSONB, nullable=True)
    )

    # Migrate data back
    op.execute("UPDATE audit_log SET details = metadata_ WHERE metadata_ IS NOT NULL")

    # Drop new columns
    op.drop_column("audit_log", "metadata_")
    op.drop_column("audit_log", "risk_level")
    op.drop_column("audit_log", "policy_decisions")
    op.drop_column("audit_log", "output_hash")
    op.drop_column("audit_log", "input_hash")

    # Rename columns back
    op.alter_column("audit_log", "user_or_agent", new_column_name="actor")
    op.alter_column("audit_log", "ts", new_column_name="timestamp")
