"""
Database cleanup script.

This script provides functionality to clean up the database by performing tasks such as:
- Deleting conversations with noisy subjects (e.g., "lunch," "booking").
- Pruning isolated entity nodes that have no connecting edges.
- Pruning orphaned entity edges that lack conversation provenance.

The script is tenant-aware and can be run for a specific tenant, all tenants,
or a dry run to preview changes.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from cortex.config.loader import get_config
from cortex.db.models import Conversation, EntityEdge, EntityNode
from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import aliased, sessionmaker


async def get_all_tenants(session: AsyncSession) -> list[str]:
    """Fetches a list of all unique tenant IDs from the database."""
    result = await session.execute(select(Conversation.tenant_id).distinct())
    # Filter out potential NULL tenant_ids from the query result
    return [tid for tid in result.scalars().all() if tid is not None]


async def cleanup_db(
    tenant_id: str | None = None, all_tenants: bool = False, dry_run: bool = False
) -> None:
    """
    Performs cleanup operations on the database for a specified tenant or all tenants.

    Args:
        tenant_id: The ID of the tenant to clean up.
        all_tenants: Flag to run cleanup for all tenants.
        dry_run: If True, simulates the cleanup without committing changes.
    """
    config = get_config()
    if not config.database or not config.database.url:
        print("Database URL is not configured. Exiting.")
        return

    db_url = config.database.url.replace("postgresql://", "postgresql+asyncpg://")
    db_url = db_url.replace("sslmode=require", "ssl=require")

    print("Connecting to database...")
    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session() as session:
            tenants_to_process: list[str] = []
            if all_tenants:
                print("Fetching all tenants...")
                tenants_to_process = await get_all_tenants(session)
                print(f"Found {len(tenants_to_process)} tenants.")
            elif tenant_id:
                tenants_to_process.append(tenant_id)

            if not tenants_to_process:
                print("No tenants specified or found. Exiting.")
                return

            for current_tenant_id in tenants_to_process:
                print(f"\n--- Starting Cleanup for Tenant: {current_tenant_id} ---")

                # 1. Delete Clutter Conversations
                print("\n[1/3] Deleting clutter conversations...")
                keyword_filters = ["%lunch%", "%booking%", "%automatic reply%"]

                clutter_conditions = or_(
                    *[Conversation.subject.ilike(kw) for kw in keyword_filters]
                )
                base_stmt = delete(Conversation).where(
                    Conversation.tenant_id == current_tenant_id, clutter_conditions
                )

                if dry_run:
                    # More accurate count for dry run
                    count_stmt = (
                        select(func.count(Conversation.id.distinct()))
                        .where(Conversation.tenant_id == current_tenant_id)
                        .where(clutter_conditions)
                    )
                    result = await session.execute(count_stmt)
                    total_deleted_convs = result.scalar_one()
                    print(
                        f"  [DRY RUN] Would delete {total_deleted_convs} conversations matching keywords."
                    )
                else:
                    result = await session.execute(base_stmt)
                    print(
                        f"  -> Deleted {result.rowcount} conversations matching keywords."
                    )

                # 2. Prune Isolated Entities (LEFT JOIN approach)
                print("\n[2/3] Pruning isolated entities...")

                # Use aliased joins to correctly identify nodes with no connections
                source_edges = aliased(EntityEdge)
                target_edges = aliased(EntityEdge)

                # Find nodes that do not appear as a source or target in any edge within the tenant
                orphaned_nodes_subquery = (
                    select(EntityNode.node_id)
                    .outerjoin(
                        source_edges,
                        and_(
                            EntityNode.node_id == source_edges.source_id,
                            source_edges.tenant_id == current_tenant_id,
                        ),
                    )
                    .outerjoin(
                        target_edges,
                        and_(
                            EntityNode.node_id == target_edges.target_id,
                            target_edges.tenant_id == current_tenant_id,
                        ),
                    )
                    .where(
                        EntityNode.tenant_id == current_tenant_id,
                        source_edges.edge_id.is_(None),
                        target_edges.edge_id.is_(None),
                    )
                    .distinct()
                ).alias()

                if dry_run:
                    # Count distinct nodes for an accurate dry-run report
                    count_stmt = select(func.count()).select_from(
                        orphaned_nodes_subquery
                    )
                    result = await session.execute(count_stmt)
                    orphan_count = result.scalar_one()
                    print(f"  [DRY RUN] Would delete {orphan_count} isolated nodes.")
                else:
                    delete_stmt = (
                        delete(EntityNode)
                        .where(EntityNode.node_id.in_(select(orphaned_nodes_subquery)))
                        .execution_options(synchronize_session=False)
                    )
                    result = await session.execute(delete_stmt)
                    if result.rowcount > 0:
                        print(f"  -> Deleted {result.rowcount} isolated nodes.")
                    else:
                        print("No isolated nodes found.")

                # 3. Prune Orphaned Edges (NULL conversation_id)
                print("\n[3/3] Pruning orphaned edges...")
                stmt = delete(EntityEdge).where(
                    EntityEdge.tenant_id == current_tenant_id,
                    EntityEdge.conversation_id.is_(None),
                )

                if dry_run:
                    count_stmt = (
                        select(func.count())
                        .select_from(stmt.table)
                        .where(stmt.whereclause)
                    )
                    result = await session.execute(count_stmt)
                    count = result.scalar_one()
                    print(f"  [DRY RUN] Would delete {count} orphaned edges.")
                else:
                    result = await session.execute(stmt)
                    if result.rowcount > 0:
                        print(f"  -> Deleted {result.rowcount} orphaned edges.")
                    else:
                        print("No orphaned edges found.")

                if dry_run:
                    print(
                        f"\n[DRY RUN] No changes were made for tenant {current_tenant_id}."
                    )
                    await session.rollback()
                else:
                    print(f"\nCommitting changes for tenant {current_tenant_id}...")
                    await session.commit()
    finally:
        await engine.dispose()
        print("\n--- Cleanup Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cortex Database Cleanup Tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tenant-id", type=str, help="The specific tenant ID to clean up."
    )
    group.add_argument(
        "--all-tenants", action="store_true", help="Run cleanup for all tenants."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the cleanup without making any changes to the database.",
    )

    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(
        cleanup_db(
            tenant_id=args.tenant_id, all_tenants=args.all_tenants, dry_run=args.dry_run
        )
    )
