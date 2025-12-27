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
from typing import List, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Ensure backend path is in sys.path
sys.path.append(str(Path(__file__).parent.parent / "backend/src").resolve())

from cortex.config.loader import get_config
from cortex.db.models import Conversation, EntityEdge, EntityNode


async def get_all_tenants(session: AsyncSession) -> List[str]:
    """Fetches a list of all unique tenant IDs from the database."""
    result = await session.execute(select(Conversation.tenant_id).distinct())
    return result.scalars().all()


async def cleanup_db(
    tenant_id: Optional[str] = None, all_tenants: bool = False, dry_run: bool = False
) -> None:
    """
    Performs cleanup operations on the database for a specified tenant or all tenants.

    Args:
        tenant_id: The ID of the tenant to clean up.
        all_tenants: Flag to run cleanup for all tenants.
        dry_run: If True, simulates the cleanup without committing changes.
    """
    config = get_config()
    db_url = config.database.url.replace("postgresql://", "postgresql+asyncpg://")
    db_url = db_url.replace("sslmode=require", "ssl=require")

    print(f"Connecting to database...")
    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        tenants_to_process: List[str] = []
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
            base_stmt = delete(Conversation).where(
                Conversation.tenant_id == current_tenant_id
            )
            total_deleted_convs = 0

            for keyword in keyword_filters:
                stmt = base_stmt.where(Conversation.subject.ilike(keyword))
                if dry_run:
                    # Estimate count for dry run
                    count_stmt = select(func.count()).select_from(stmt.table).where(*stmt.where.clauses)
                    result = await session.execute(count_stmt)
                    count = result.scalar_one()
                    print(f"  [DRY RUN] Would delete {count} conversations matching '{keyword}'.")
                else:
                    result = await session.execute(stmt)
                    count = result.rowcount
                    print(f"  -> Deleted {count} conversations matching '{keyword}'.")
                total_deleted_convs += count

            print(f"Total clutter conversations deleted: {total_deleted_convs}")

            # 2. Prune Isolated Entities (LEFT JOIN approach)
            print("\n[2/3] Pruning isolated entities...")

            # Find nodes that do not appear as a source or target in any edge
            orphaned_nodes_stmt = (
                select(EntityNode)
                .outerjoin(EntityEdge, EntityNode.node_id == EntityEdge.source_id)
                .outerjoin(EntityEdge, EntityNode.node_id == EntityEdge.target_id)
                .where(
                    EntityNode.tenant_id == current_tenant_id,
                    EntityEdge.edge_id.is_(None)
                )
            )

            if dry_run:
                count_stmt = select(func.count()).select_from(orphaned_nodes_stmt.alias())
                result = await session.execute(count_stmt)
                orphan_count = result.scalar_one()
                print(f"  [DRY RUN] Would delete {orphan_count} isolated nodes.")
            else:
                result = await session.execute(orphaned_nodes_stmt)
                orphan_nodes = result.scalars().all()
                if orphan_nodes:
                    print(f"Found {len(orphan_nodes)} isolated nodes. Deleting...")

                    # Extract IDs for deletion
                    orphan_ids = [node.node_id for node in orphan_nodes]
                    delete_stmt = delete(EntityNode).where(EntityNode.node_id.in_(orphan_ids))
                    delete_result = await session.execute(delete_stmt)
                    print(f"  -> Deleted {delete_result.rowcount} nodes.")
                else:
                    print("No isolated nodes found.")

            # 3. Prune Orphaned Edges (NULL conversation_id)
            print("\n[3/3] Pruning orphaned edges...")
            stmt = delete(EntityEdge).where(
                EntityEdge.tenant_id == current_tenant_id,
                EntityEdge.conversation_id.is_(None),
            )

            if dry_run:
                count_stmt = select(func.count()).select_from(stmt.table).where(*stmt.where.clauses)
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
                print(f"\n[DRY RUN] No changes were made for tenant {current_tenant_id}.")
                await session.rollback()
            else:
                print(f"\nCommitting changes for tenant {current_tenant_id}...")
                await session.commit()

    await engine.dispose()
    print("\n--- Cleanup Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cortex Database Cleanup Tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tenant-id", type=str, help="The specific tenant ID to clean up.")
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

    asyncio.run(cleanup_db(tenant_id=args.tenant_id, all_tenants=args.all_tenants, dry_run=args.dry_run))
