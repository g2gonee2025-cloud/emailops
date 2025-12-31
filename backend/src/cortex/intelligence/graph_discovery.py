"""
Standalone library for discovering graph schema from conversation data.
"""

import asyncio
import hashlib
import logging
import random
from collections import Counter
from typing import Optional

from cortex.db.models import Chunk
from cortex.db.session import get_db_session
from cortex.intelligence.graph import GraphExtractor
from rich.console import Console
from rich.table import Table
from sqlalchemy import func, select

logger = logging.getLogger(__name__)
_MAX_TEXT_CHARS = 4000
_MAX_CONCURRENCY = 5
_ENTITY_SAMPLE_SIZE = 10
_MAX_TABLE_ROWS = 20


def _emit_message(
    console: Console | None, message: str, *, level: int = logging.INFO
) -> None:
    if console:
        console.print(message)
    else:
        logger.log(level, message)


def _get_sample_texts(
    db_session, *, tenant_id: str, sample_size: int, console: Console | None
) -> list[str]:
    """Fetch a random sample of conversation texts."""
    tenant_hash = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:8]
    _emit_message(
        console,
        f"Fetching {sample_size} sampled conversations for tenant {tenant_hash}...",
    )

    total = (
        db_session.execute(
            select(func.count(func.distinct(Chunk.conversation_id))).where(
                Chunk.tenant_id == tenant_id, Chunk.chunk_type == "message_body"
            )
        ).scalar_one()
        or 0
    )
    if total == 0:
        _emit_message(
            console,
            "No conversations found for this tenant.",
            level=logging.WARNING,
        )
        return []

    offset = 0
    if total > sample_size:
        offset = random.randint(0, max(0, total - sample_size))

    stmt = (
        select(Chunk.conversation_id)
        .where(Chunk.tenant_id == tenant_id, Chunk.chunk_type == "message_body")
        .distinct()
        .order_by(Chunk.conversation_id)
        .offset(offset)
        .limit(sample_size)
    )
    conv_ids = db_session.execute(stmt).scalars().all()

    if not conv_ids:
        _emit_message(
            console,
            "No conversations found for this tenant.",
            level=logging.WARNING,
        )
        return []

    texts: list[str] = []
    chunks_stmt = (
        select(Chunk.conversation_id, Chunk.text)
        .where(
            Chunk.conversation_id.in_(conv_ids),
            Chunk.chunk_type == "message_body",
            Chunk.char_start == 0,
        )
        .order_by(Chunk.conversation_id)
    )
    for _, chunk_text in db_session.execute(chunks_stmt):
        if not chunk_text:
            continue
        truncated = chunk_text[:_MAX_TEXT_CHARS]
        texts.append(truncated)
    return texts


async def _analyze_schema(
    texts: list[str],
    *,
    console: Console | None,
    show_entities: bool,
) -> None:
    """Analyze the schema of the sampled texts."""
    extractor = GraphExtractor()
    node_types = Counter()
    relations = Counter()
    entity_sample: list[str] = []
    entity_seen = 0

    _emit_message(console, f"Starting graph extraction on {len(texts)} texts...")

    # Limit concurrency to prevent OOM
    # Semaphore allows speedup without memory explosion
    sem = asyncio.Semaphore(_MAX_CONCURRENCY)

    async def _process_with_sem(idx, text):
        async with sem:
            if console:
                console.print(f"  Processing {idx}/{len(texts)}...")
            else:
                logger.debug("Processing %s/%s", idx, len(texts))
            try:
                G = await extractor.extract_graph(text)

                n_types = [
                    data.get("type", "UNKNOWN") for _, data in G.nodes(data=True)
                ]
                rels = [
                    data.get("relation", "UNKNOWN") for _, _, data in G.edges(data=True)
                ]
                names = list(G.nodes())

                # Return results to main loop for aggregation
                return n_types, rels, names, False
            except Exception:
                logger.exception("Extraction failed for text %s", idx)
                return [], [], [], True

    tasks = [_process_with_sem(idx, text) for idx, text in enumerate(texts, 1)]
    results = await asyncio.gather(*tasks)

    error_count = 0
    for n_types, rels, names, failed in results:
        node_types.update(n_types)
        relations.update(rels)
        if failed:
            error_count += 1
        for name in names:
            entity_seen += 1
            if len(entity_sample) < _ENTITY_SAMPLE_SIZE:
                entity_sample.append(name)
            else:
                target = random.randint(0, entity_seen - 1)
                if target < _ENTITY_SAMPLE_SIZE:
                    entity_sample[target] = name

    if error_count:
        _emit_message(
            console,
            f"Extraction failed for {error_count} of {len(texts)} texts.",
            level=logging.WARNING,
        )

    if console:
        console.print("\n[bold green]=== SCHEMA DISCOVERY REPORT ===[/bold green]")

    # Display Node Types
    if console:
        table = Table(title="Top Node Types")
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="magenta")
        for k, v in node_types.most_common(_MAX_TABLE_ROWS):
            table.add_row(k, str(v))
        console.print(table)
    else:
        logger.info("Top node types: %s", node_types.most_common(_MAX_TABLE_ROWS))

    # Display Relationships
    if console:
        table = Table(title="Top Relationships")
        table.add_column("Relation", style="cyan")
        table.add_column("Count", style="magenta")
        for k, v in relations.most_common(_MAX_TABLE_ROWS):
            table.add_row(k, str(v))
        console.print(table)
    else:
        logger.info("Top relationships: %s", relations.most_common(_MAX_TABLE_ROWS))

    # Display Sample Entities
    if show_entities and entity_sample:
        if console:
            table = Table(title="Sample Entities")
            table.add_column("Entity Name", style="cyan")
            for name in entity_sample:
                table.add_row(name)
            console.print(table)
        else:
            logger.info("Sample entities: %s", entity_sample)
    elif show_entities:
        _emit_message(
            console,
            "No entities found in the sample.",
            level=logging.WARNING,
        )


def discover_graph_schema(
    *,
    tenant_id: str,
    sample_size: int = 20,
    console: Console | None = None,
    show_entities: bool = False,
) -> None:
    """
    Analyzes a sample of conversations to discover the graph schema.

    Args:
        tenant_id: The tenant ID to analyze.
        sample_size: The number of conversations to sample.
    """

    with get_db_session() as session:
        texts = _get_sample_texts(
            session, tenant_id=tenant_id, sample_size=sample_size, console=console
        )
        if texts:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(
                    _analyze_schema(texts, console=console, show_entities=show_entities)
                )
            else:
                raise RuntimeError(
                    "discover_graph_schema cannot run inside an active event loop. "
                    "Use discover_graph_schema_async instead."
                )


async def discover_graph_schema_async(
    *,
    tenant_id: str,
    sample_size: int = 20,
    console: Console | None = None,
    show_entities: bool = False,
) -> None:
    """Async entrypoint for schema discovery in running event loops."""
    with get_db_session() as session:
        texts = _get_sample_texts(
            session, tenant_id=tenant_id, sample_size=sample_size, console=console
        )
    if texts:
        await _analyze_schema(texts, console=console, show_entities=show_entities)
