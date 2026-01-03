"""
Graph-based Retrieval for Knowledge Graph RAG.

Implements entity-aware search that leverages the EntityNode and EntityEdge
tables to find contextually related conversations via graph traversal.
"""

from __future__ import annotations

import asyncio
import logging
import re
from uuid import UUID

from cortex.common.types import Err, Ok, Result
from cortex.db.models import EntityEdge, EntityNode
from pydantic import BaseModel
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Constants
DEFAULT_ENTITY_LIMIT = 10
DEFAULT_MAX_HOPS = 1
MIN_TRIGRAM_SIMILARITY = 0.3
DEFAULT_NEIGHBOR_LIMIT = 100
MAX_GRAPH_CONCURRENCY = 10

# Semaphore to limit concurrent graph retrieval operations
_graph_semaphore: asyncio.Semaphore | None = None


def _get_graph_semaphore() -> asyncio.Semaphore:
    """Get or create the graph retrieval semaphore."""
    global _graph_semaphore
    if _graph_semaphore is None:
        _graph_semaphore = asyncio.Semaphore(MAX_GRAPH_CONCURRENCY)
    return _graph_semaphore


def escape_like_pattern(value: str) -> str:
    """
    Escape SQL LIKE/ILIKE special characters in a string.

    Escapes %, _, and \\ to prevent wildcard injection when using
    ILIKE with escape='\\'.

    Args:
        value: The string to escape

    Returns:
        Escaped string safe for use in LIKE patterns
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class GraphSearchInput(BaseModel):
    """Input for graph-based retrieval."""

    tenant_id: str
    query: str
    entity_names: list[str] | None = None  # Pre-extracted entities (optional)
    k: int = 20
    max_hops: int = DEFAULT_MAX_HOPS


class GraphSearchResult(BaseModel):
    """A single graph search result."""

    conversation_id: str
    entity_name: str
    entity_type: str
    relation: str | None = None
    connected_entity: str | None = None
    pagerank_score: float = 0.0
    connection_count: int = 1
    source: str = "graph"


class EntityMatch(BaseModel):
    """Matched entity from the knowledge graph."""

    node_id: str
    name: str
    entity_type: str
    pagerank: float
    similarity: float = 1.0


def _build_entity_match(node: EntityNode, *, similarity: float) -> EntityMatch:
    pagerank_value = float(getattr(node, "pagerank", 0.0) or 0.0)
    name = node.name or ""
    if not isinstance(name, str):
        name = str(name)
    entity_type = node.entity_type or "UNKNOWN"
    if not isinstance(entity_type, str):
        entity_type = str(entity_type)
    return EntityMatch(
        node_id=str(node.node_id),
        name=name,
        entity_type=entity_type,
        pagerank=pagerank_value,
        similarity=float(similarity),
    )


def search_entities_by_name(
    session: Session,
    tenant_id: str,
    entity_names: list[str],
    limit: int = DEFAULT_ENTITY_LIMIT,
) -> list[EntityMatch]:
    """
    Search for entities by name using exact match first, then trigram fallback.

    Args:
        session: Database session
        tenant_id: Tenant ID for filtering
        entity_names: List of entity names to search for
        limit: Maximum entities to return

    Returns:
        List of matched EntityMatch objects
    """
    if not entity_names:
        return []

    filtered_names = [name for name in entity_names if name]
    if not filtered_names:
        return []

    matches: list[EntityMatch] = []
    matched_ids: set[str] = set()
    pagerank_column = getattr(EntityNode, "pagerank", None)

    # Build ILIKE conditions for exact/partial matching
    ilike_conditions = []
    for name in filtered_names:
        sanitized = escape_like_pattern(name)
        ilike_conditions.append(EntityNode.name.ilike(f"%{sanitized}%", escape="\\"))

    if not ilike_conditions:
        return []

    # First try: exact/substring match
    exact_stmt = select(EntityNode).where(
        EntityNode.tenant_id == tenant_id, or_(*ilike_conditions)
    )
    if pagerank_column is not None:
        exact_stmt = exact_stmt.order_by(pagerank_column.desc())
    exact_stmt = exact_stmt.limit(limit)

    exact_results = session.execute(exact_stmt).scalars().all()

    for node in exact_results:
        node_id = str(node.node_id)
        if node_id in matched_ids:
            continue
        matches.append(_build_entity_match(node, similarity=1.0))
        matched_ids.add(node_id)

    if len(matches) >= limit:
        return matches[:limit]

    remaining = limit - len(matches)
    existing_uuid_ids = {UUID(node_id) for node_id in matched_ids}

    similarity_exprs = [
        func.similarity(EntityNode.name, name) for name in filtered_names
    ]
    if len(similarity_exprs) == 1:
        similarity_score = similarity_exprs[0]
    else:
        similarity_score = func.greatest(*similarity_exprs)
    similarity_label = similarity_score.label("sim")

    trgm_stmt = select(EntityNode, similarity_label).where(
        EntityNode.tenant_id == tenant_id,
        similarity_score > MIN_TRIGRAM_SIMILARITY,
    )
    if existing_uuid_ids:
        trgm_stmt = trgm_stmt.where(EntityNode.node_id.notin_(existing_uuid_ids))

    order_cols = [similarity_label.desc()]
    if pagerank_column is not None:
        order_cols.append(pagerank_column.desc())
    trgm_stmt = trgm_stmt.order_by(*order_cols).limit(remaining)

    try:
        trgm_results = session.execute(trgm_stmt).all()
    except Exception:
        logger.exception("Trigram search failed")
        raise

    for node, sim in trgm_results:
        node_id = str(node.node_id)
        if node_id in matched_ids:
            continue
        matches.append(_build_entity_match(node, similarity=float(sim or 0.0)))
        matched_ids.add(node_id)
        if len(matches) >= limit:
            break

    return matches[:limit]


def _safe_parse_uuid(value: str) -> UUID | None:
    """
    Safely parse a string as UUID, returning None if invalid.

    Args:
        value: String to parse as UUID

    Returns:
        UUID if valid, None otherwise
    """
    try:
        return UUID(value)
    except (ValueError, TypeError):
        logger.warning("Invalid UUID format: %s", value[:50] if value else "None")
        return None


def expand_entity_neighbors(
    session: Session,
    entity_ids: list[str],
    tenant_id: str,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> list[tuple[EntityEdge, EntityNode]]:
    """
    Expand entities via their edges to find connected entities.

    Args:
        session: Database session
        entity_ids: List of entity node IDs to expand from
        tenant_id: Tenant ID
        max_hops: Maximum traversal depth (1 = direct neighbors only)

    Note:
        The DEFAULT_NEIGHBOR_LIMIT (100) is applied per hop and per direction.
        This may truncate neighbors in early hops, preventing discovery of
        relevant nodes in later hops.

    Returns:
        List of (edge, neighbor_node) tuples
    """
    if not entity_ids or max_hops < 1:
        return []

    # Safely parse UUIDs, filtering out invalid ones
    frontier: set[UUID] = set()
    for eid in entity_ids:
        parsed = _safe_parse_uuid(eid)
        if parsed is not None:
            frontier.add(parsed)

    if not frontier:
        return []
    seen_nodes = set(frontier)
    seen_edges: set[UUID] = set()
    unique_results: list[tuple[EntityEdge, EntityNode]] = []

    for _ in range(max_hops):
        if not frontier:
            break

        edges_stmt = (
            select(EntityEdge, EntityNode)
            .join(EntityNode, EntityEdge.target_id == EntityNode.node_id)
            .where(
                EntityEdge.tenant_id == tenant_id,
                EntityEdge.source_id.in_(frontier),
            )
            .order_by(EntityEdge.weight.desc())
            .limit(DEFAULT_NEIGHBOR_LIMIT)
        )

        reverse_stmt = (
            select(EntityEdge, EntityNode)
            .join(EntityNode, EntityEdge.source_id == EntityNode.node_id)
            .where(
                EntityEdge.tenant_id == tenant_id,
                EntityEdge.target_id.in_(frontier),
            )
            .order_by(EntityEdge.weight.desc())
            .limit(DEFAULT_NEIGHBOR_LIMIT)
        )

        all_results = (
            session.execute(edges_stmt).all() + session.execute(reverse_stmt).all()
        )

        next_frontier: set[UUID] = set()
        for edge, node in all_results:
            if edge.edge_id in seen_edges:
                continue
            seen_edges.add(edge.edge_id)
            unique_results.append((edge, node))
            node_id = node.node_id
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                next_frontier.add(node_id)

        frontier = next_frontier

    return unique_results


def get_conversations_from_entities(
    session: Session,
    entity_ids: list[str],
    tenant_id: str,
    limit: int = 50,
) -> list[tuple[str, int]]:
    """
    Get conversation IDs connected to the given entities via edges.

    Args:
        session: Database session
        entity_ids: List of entity node IDs
        tenant_id: Tenant ID
        limit: Maximum conversations to return

    Returns:
        List of (conversation_id, connection_count) tuples, sorted by count DESC
    """
    if not entity_ids:
        return []

    # Safely parse UUIDs, filtering out invalid ones
    uuid_ids = [
        parsed for eid in entity_ids if (parsed := _safe_parse_uuid(eid)) is not None
    ]
    if not uuid_ids:
        return []

    conn_count = func.count().label("conn_count")
    conv_stmt = (
        select(EntityEdge.conversation_id, conn_count)
        .where(
            EntityEdge.tenant_id == tenant_id,
            EntityEdge.conversation_id.isnot(None),
            or_(
                EntityEdge.source_id.in_(uuid_ids),
                EntityEdge.target_id.in_(uuid_ids),
            ),
        )
        .group_by(EntityEdge.conversation_id)
        .order_by(conn_count.desc())
        .limit(limit)
    )

    results = session.execute(conv_stmt).all()

    return [(str(row.conversation_id), int(row.conn_count)) for row in results]


def get_conversation_entity_map(
    session: Session,
    conversation_ids: list[str],
    entity_ids: set[str],
    tenant_id: str,
) -> dict[str, set[str]]:
    """Map conversations to connected entity IDs from the provided entity set."""
    if not conversation_ids or not entity_ids:
        return {}

    # Safely parse UUIDs, filtering out invalid ones
    conv_uuid_ids = {
        parsed for cid in conversation_ids if (parsed := _safe_parse_uuid(cid)) is not None
    }
    entity_uuid_ids = {
        parsed for eid in entity_ids if (parsed := _safe_parse_uuid(eid)) is not None
    }
    if not conv_uuid_ids or not entity_uuid_ids:
        return {}

    stmt = (
        select(EntityEdge.conversation_id, EntityEdge.source_id, EntityEdge.target_id)
        .where(
            EntityEdge.tenant_id == tenant_id,
            EntityEdge.conversation_id.in_(conv_uuid_ids),
            or_(
                EntityEdge.source_id.in_(entity_uuid_ids),
                EntityEdge.target_id.in_(entity_uuid_ids),
            ),
        )
        .order_by(EntityEdge.conversation_id)
    )

    conv_map: dict[str, set[str]] = {cid: set() for cid in conversation_ids}
    for conv_id, source_id, target_id in session.execute(stmt).all():
        conv_key = str(conv_id)
        if source_id in entity_uuid_ids:
            conv_map.setdefault(conv_key, set()).add(str(source_id))
        if target_id in entity_uuid_ids:
            conv_map.setdefault(conv_key, set()).add(str(target_id))

    return conv_map


_ENTITY_STOP_WORDS: set[str] = {
    "what",
    "when",
    "where",
    "why",
    "who",
    "how",
    "can",
    "could",
    "should",
    "would",
    "does",
    "do",
    "did",
    "is",
    "are",
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "from",
    "for",
    "of",
    "in",
    "on",
    "at",
    "by",
    "please",
    "tell",
    "explain",
    "find",
    "search",
    "show",
    "get",
    "about",
    "with",
    "this",
    "that",
    "these",
    "those",
    "have",
    "has",
    "had",
    "been",
    "being",
    "was",
    "were",
    "will",
    "may",
    "might",
    "must",
    "shall",
    "latest",
    "recent",
    "all",
    "policy",
    "policies",
    "claim",
    "claims",
    "email",
    "emails",
    "document",
    "documents",
}


def extract_query_entities(query: str) -> list[str]:
    """
    Extract entity names from a user query using heuristics.

    Returns a deterministic, sorted list of entity candidates extracted from:
    - Quoted strings
    - Capitalized phrases (including hyphenated names like "ACME-Corp")
    - Single capitalized words

    Stop words are filtered case-insensitively and punctuation is stripped.

    Args:
        query: User search query

    Returns:
        List of extracted entity names (sorted for determinism)
    """
    if not query:
        return []

    candidates: list[str] = []
    seen_lower: set[str] = set()

    def add_candidate(s: str) -> None:
        """Add candidate if not already seen (case-insensitive dedup)."""
        stripped = s.strip().strip(".,!?;:")
        if not stripped:
            return
        lower = stripped.lower()
        if lower not in seen_lower and lower not in _ENTITY_STOP_WORDS:
            seen_lower.add(lower)
            candidates.append(stripped)

    quoted = re.findall(r'["\']([^"\']{2,100})["\']', query)
    for s in quoted:
        add_candidate(s)

    capitalized_phrases = re.findall(r"\b[A-Z][\w&-]*(?:\s+[A-Z][\w&-]*)+\b", query)
    for s in capitalized_phrases:
        add_candidate(s)

    single_caps = re.findall(r"\b[A-Z][a-z]{2,}\b", query)
    for s in single_caps:
        add_candidate(s)

    return sorted(candidates, key=str.lower)


def _graph_retrieve_sync(
    args: GraphSearchInput,
) -> Result[list[GraphSearchResult], str]:
    """
    Perform graph-based retrieval to find related conversations.

    This is the main entry point for graph search, orchestrating:
    1. Entity extraction from query (if not provided)
    2. Entity name matching in the graph
    3. Neighbor expansion via edges
    4. Conversation ID retrieval
    """
    from cortex.db.session import SessionLocal, set_session_tenant

    with SessionLocal() as session:
        set_session_tenant(session, args.tenant_id)

        # 1. Extract entities from query if not provided
        entity_names = args.entity_names
        if not entity_names:
            entity_names = extract_query_entities(args.query)

        if not entity_names:
            logger.debug("No entities extracted from query")
            return Ok([])

        logger.info("Graph search extracted %d entities", len(entity_names))

        # 2. Search for entities in the graph
        matched_entities = search_entities_by_name(
            session,
            args.tenant_id,
            entity_names,
            limit=DEFAULT_ENTITY_LIMIT,
        )

        if not matched_entities:
            logger.debug("No entity matches found")
            return Ok([])

        logger.info("Graph search matched %d entities", len(matched_entities))

        entity_info_by_id = {e.node_id: e for e in matched_entities}
        entity_ids = set(entity_info_by_id.keys())

        # 3. Expand via edges to get related entities
        neighbors = expand_entity_neighbors(
            session, list(entity_ids), args.tenant_id, max_hops=args.max_hops
        )

        # Add neighbor entity IDs to our set
        for _, node in neighbors:
            node_id = str(node.node_id)
            if node_id not in entity_info_by_id:
                entity_info_by_id[node_id] = _build_entity_match(node, similarity=0.0)
            entity_ids.add(node_id)

        # 4. Get conversations connected to these entities
        conv_results = get_conversations_from_entities(
            session, list(entity_ids), args.tenant_id, limit=args.k
        )

        if not conv_results:
            return Ok([])

        conv_ids = [conv_id for conv_id, _ in conv_results]
        conv_entity_map = get_conversation_entity_map(
            session, conv_ids, entity_ids, args.tenant_id
        )

        fallback_entity = max(matched_entities, key=lambda e: e.pagerank)

        # 5. Build result objects
        results: list[GraphSearchResult] = []

        for conv_id, conn_count in conv_results:
            connected_ids = conv_entity_map.get(conv_id, set())
            best_entity = None
            if connected_ids:
                candidates = [
                    entity_info_by_id.get(entity_id)
                    for entity_id in connected_ids
                    if entity_id in entity_info_by_id
                ]
                if candidates:
                    best_entity = max(candidates, key=lambda e: e.pagerank)

            if best_entity is None:
                best_entity = fallback_entity

            results.append(
                GraphSearchResult(
                    conversation_id=conv_id,
                    entity_name=best_entity.name if best_entity else "Unknown",
                    entity_type=best_entity.entity_type if best_entity else "UNKNOWN",
                    pagerank_score=best_entity.pagerank if best_entity else 0.0,
                    connection_count=conn_count,
                    source="graph",
                )
            )

        logger.info("Graph retrieval returned %d conversations", len(results))
        return Ok(results)


async def graph_retrieve(
    args: GraphSearchInput,
) -> Result[list[GraphSearchResult], str]:
    """
    Async wrapper for graph retrieval to avoid blocking the event loop.

    Uses a semaphore to limit concurrent graph retrieval operations and
    prevent connection pool exhaustion under high load.
    """
    semaphore = _get_graph_semaphore()
    try:
        async with semaphore:
            return await asyncio.to_thread(_graph_retrieve_sync, args)
    except TimeoutError:
        logger.error("Graph retrieval timed out for tenant %s", args.tenant_id)
        return Err("Graph retrieval timed out")
    except ValueError as e:
        logger.error("Graph retrieval value error: %s", str(e))
        return Err(f"Graph retrieval invalid input: {e}")
    except Exception as e:
        error_type = type(e).__name__
        logger.exception("Graph retrieval failed with %s", error_type)
        return Err(f"Graph retrieval failed: {error_type}")
