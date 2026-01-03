import asyncio
import sys
import urllib.parse
from pathlib import Path

from cortex.config.loader import get_config
from cortex.db.models import Conversation
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

SAMPLE_IDS_LIMIT = 5


async def check_summaries():
    config = get_config()

    if not config or not config.database or not config.database.url:
        raise ValueError("Database configuration is missing or incomplete.")

    # Safely parse and modify the database URL
    parsed_url = urllib.parse.urlparse(config.database.url)

    if parsed_url.scheme != "postgresql":
        raise ValueError(f"Unsupported database scheme: {parsed_url.scheme}")

    # Replace scheme and handle SSL parameters
    new_scheme = "postgresql+asyncpg"
    query_params = urllib.parse.parse_qs(parsed_url.query)

    if query_params.get("sslmode") == ["require"]:
        query_params.pop("sslmode")
        query_params["ssl"] = ["require"]

    # Reconstruct the URL
    db_url = urllib.parse.urlunparse(
        (
            new_scheme,
            parsed_url.netloc,
            parsed_url.path,
            parsed_url.params,
            urllib.parse.urlencode(query_params, doseq=True),
            parsed_url.fragment,
        )
    )

    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        print("Connecting to database...")
        async with async_session() as session:
            print("\n--- Summary Verification ---")

            # Execute a single, combined query for efficiency
            stmt = select(
                func.count(Conversation.conversation_id),
                func.count(
                    case(
                        (
                            Conversation.summary_text.is_(None),
                            Conversation.conversation_id,
                        )
                    )
                ),
                func.count(
                    case(
                        (Conversation.summary_text == "", Conversation.conversation_id)
                    )
                ),
            )
            total_count, null_count, empty_count = (await session.execute(stmt)).one()

            # 4. Valid Summaries
            valid_count = total_count - null_count - empty_count

            print(f"Total Conversations: {total_count}")
            print(f"Valid Summaries:     {valid_count}")
            print(f"Missing (NULL):      {null_count}")
            print(f"Empty Strings:       {empty_count}")

            if total_count > 0:
                coverage = (valid_count / total_count) * 100
                print(f"Coverage:            {coverage:.2f}%")
            else:
                print("Coverage: N/A (No conversations found)")

            if null_count > 0 or empty_count > 0:
                print("\nWARNING: Some conversations are missing summaries.")

                # optional: list a few IDs that are missing
                stmt_missing_ids = (
                    select(Conversation.conversation_id)
                    .where(
                        (Conversation.summary_text.is_(None))
                        | (Conversation.summary_text == "")
                    )
                    .limit(SAMPLE_IDS_LIMIT)
                )
                missing_ids = (await session.execute(stmt_missing_ids)).scalars().all()
                print(f"Sample IDs missing summaries: {missing_ids}")

    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        # In a real script, you might want to exit with a non-zero status code
        # sys.exit(1)
    finally:
        if "engine" in locals():
            print("\nClosing database connection...")
            await engine.dispose()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(check_summaries())
