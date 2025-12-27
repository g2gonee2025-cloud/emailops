"""add_node_merge_func

Revision ID: 015
Revises: 014
Create Date: 2025-12-25 01:15:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "015_add_node_merge_func"
down_revision: Union[str, None] = "014_add_summary_text"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
    CREATE OR REPLACE FUNCTION merge_entity_nodes(keep_id UUID, discard_id UUID)
    RETURNS VOID AS $$
    DECLARE
        discard_props JSONB;
    BEGIN
        -- Get properties from discard node
        SELECT properties INTO discard_props FROM entity_nodes WHERE node_id = discard_id;

        -- Update keep node: merge properties (keep node properties take precedence over discard node properties)
        -- We do discard_props || keep_props.
        -- Note: Standard || operator for jsonb merges keys. distinct keys from both are kept. common keys take value from right operand.
        UPDATE entity_nodes
        SET properties = COALESCE(discard_props, '{}'::jsonb) || COALESCE(properties, '{}'::jsonb)
        WHERE node_id = keep_id;

        -- Re-point outgoing edges
        -- Note: This might create duplicate edges if keep_id already has an edge to the same target with same relation
        -- Ideally we would handle that, but for now strict re-pointing is the MVP.
        -- If uniqueness constraints exist, this might fail.
        UPDATE entity_edges
        SET source_id = keep_id
        WHERE source_id = discard_id;

        -- Re-point incoming edges
        UPDATE entity_edges
        SET target_id = keep_id
        WHERE target_id = discard_id;

        -- Delete the discarded node
        DELETE FROM entity_nodes WHERE node_id = discard_id;
    END;
    $$ LANGUAGE plpgsql;
    """
    )


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS merge_entity_nodes(UUID, UUID);")
