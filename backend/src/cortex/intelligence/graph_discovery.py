"""
Standalone library for discovering graph schema from conversation data.
"""

import asyncio
import logging
import random
from collections import Counter
from typing import List

from cortex.db.models import Chunk
from cortex.db.session import get_db_session
from cortex.intelligence.graph import GraphExtractor
from rich.console import Console
from rich.table import Table
from sqlalchemy import func, select

logger = logging.getLogger(__name__)


def discover_graph_schema(*, tenant_id: str, sample_size: int = 20) -> None:
    """
    Analyzes a sample of conversations to discover the graph schema.

    Args:
        tenant_id: The tenant ID to analyze.
        sample_size: The number of conversations to sample.
    """

    console = Console()

    def get_sample_texts(db_session, *, tenant_id: str, sample_size: int) -> list[str]:
        """Fetch a random sample of conversation texts."""
        console.print(
            f"Fetching {sample_size} random conversations for tenant '{tenant_id}'..."
        )
        stmt = (
            select(Chunk.conversation_id)
            .where(Chunk.tenant_id == tenant_id, Chunk.chunk_type == "message_body")
            .group_by(Chunk.conversation_id)
            .order_by(func.random())
            .limit(sample_size)
        )
        conv_ids = db_session.execute(stmt).scalars().all()

        if not conv_ids:
            console.print("[yellow]No conversations found for this tenant.[/yellow]")
            return []

        texts = []
        for cid in conv_ids:
            chunks_stmt = (
                select(Chunk.text)
                .where(
                    Chunk.conversation_id == cid,
                    Chunk.chunk_type == "message_body",
                )
                .order_by(Chunk.position)
            )
            chunk_texts = db_session.execute(chunks_stmt).scalars().all()
            full_text = "\n".join(chunk_texts)
            texts.append(full_text)
        return texts

    async def analyze_schema(texts: list[str]):
        """Analyze the schema of the sampled texts."""
        extractor = GraphExtractor()
        node_types = Counter()
        relations = Counter()
        entity_names = []

        console.print(
            f"Starting graph extraction on {len(texts)} texts with asyncio..."
        )

        async def process_text(text: str) -> dict:
            try:
                G = await extractor.extract_graph(text)
                n_types = [
                    data.get("type", "UNKNOWN") for _, data in G.nodes(data=True)
                ]
                rels = [
                    data.get("relation", "UNKNOWN") for _, _, data in G.edges(data=True)
                ]
                names = list(G.nodes())
                return {"types": n_types, "relations": rels, "names": names}
            except Exception as e:
                logger.error(f"Extraction failed: {e}")
                return {"types": [], "relations": [], "names": []}

        try:
            tasks = [process_text(text) for text in texts]
            results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.critical(f"Asyncio gather failed: {e}")
            return

        for res in results:
            node_types.update(res["types"])
            relations.update(res["relations"])
            entity_names.extend(res["names"])

        console.print("\n[bold green]=== SCHEMA DISCOVERY REPORT ===[/bold green]")

        # Display Node Types
        table = Table(title="Top 20 Node Types")
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="magenta")
        for k, v in node_types.most_common(20):
            table.add_row(k, str(v))
        console.print(table)

        # Display Relationships
        table = Table(title="Top 20 Relationships")
        table.add_column("Relation", style="cyan")
        table.add_column("Count", style="magenta")
        for k, v in relations.most_common(20):
            table.add_row(k, str(v))
        console.print(table)

        # Display Sample Entities
        table = Table(title="Random Sample of 10 Entities")
        table.add_column("Entity Name", style="cyan")
        for name in random.sample(entity_names, min(10, len(entity_names))):
            table.add_row(name)
        console.print(table)

    with get_db_session() as session:
        texts = get_sample_texts(session, tenant_id=tenant_id, sample_size=sample_size)
        if texts:
            asyncio.run(analyze_schema(texts))
