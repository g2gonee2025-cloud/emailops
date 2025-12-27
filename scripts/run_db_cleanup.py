import asyncio
import sys
from pathlib import Path

from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Ensure backend path is in sys.path
sys.path.append(str(Path("backend/src").resolve()))

from cortex.config.loader import get_config
from cortex.db.models import Conversation, EntityNode


async def cleanup_db():
    config = get_config()
    # Patch for async driver
    db_url = config.database.url.replace("postgresql://", "postgresql+asyncpg://")
    db_url = db_url.replace("sslmode=require", "ssl=require")

    print(f"Connecting to: {db_url}")
    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        print("\n--- Starting Cleanup ---")

        # 1. Delete Clutter Conversations
        keyword_filters = [
            "%lunch%",
            "%booking%",
            "%automatic reply%",
        ]  # Removed 'out of office' as it had 0 matches

        total_deleted_convs = 0

        for keyword in keyword_filters:
            print(f"Deleting conversations matching '{keyword}'...")
            stmt = delete(Conversation).where(Conversation.subject.ilike(keyword))
            result = await session.execute(stmt)
            count = result.rowcount
            print(f"  -> Deleted {count} conversations.")
            total_deleted_convs += count

        await session.commit()
        print(f"Total conversations deleted: {total_deleted_convs}")

        # 2. Prune Isolated Entities (Nodes with no edges)
        print("\nPruning isolated entities...")
        # Complex deletion: Delete nodes where NOT EXISTS in edges sources OR targets
        # We can do this with a subquery or 2 separate checks.
        # Safest is to find IDs first, then delete.

        find_orphans_sql = text(
            """
            SELECT node_id FROM entity_nodes n
            WHERE NOT EXISTS (SELECT 1 FROM entity_edges e WHERE e.source_id = n.node_id)
            AND NOT EXISTS (SELECT 1 FROM entity_edges e WHERE e.target_id = n.node_id)
        """
        )

        result = await session.execute(find_orphans_sql)
        orphan_ids = result.scalars().all()

        if orphan_ids:
            print(f"Found {len(orphan_ids)} isolated nodes. Deleting...")
            stmt = delete(EntityNode).where(EntityNode.node_id.in_(orphan_ids))
            result = await session.execute(stmt)
            print(f"  -> Deleted {result.rowcount} nodes.")
            await session.commit()
        else:
            print("No isolated nodes found.")

        # 3. Prune Orphaned Edges (Edges with no Conversation provenance)
        print("\nPruning orphaned edges (NULL conversation)...")
        from cortex.db.models import EntityEdge  # Ensure imported

        stmt = delete(EntityEdge).where(EntityEdge.conversation_id.is_(None))
        result = await session.execute(stmt)
        if result.rowcount > 0:
            print(f"  -> Deleted {result.rowcount} orphaned edges.")
            await session.commit()
        else:
            print("No orphaned edges found.")

    await engine.dispose()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(cleanup_db())
