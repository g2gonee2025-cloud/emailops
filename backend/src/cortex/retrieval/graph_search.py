"""
Graph-based Retrieval for Knowledge Graph RAG.

Implements entity-aware search that leverages the EntityNode and EntityEdge
tables to find contextually related conversations via graph traversal.
"""

from __future__ import annotations

import asyncio
import logging
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
        sanitized = name.replace("%", "\\%").replace("_", "\\_")
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

    Returns:
        List of (edge, neighbor_node) tuples
    """
    if not entity_ids or max_hops < 1:
        return []

    frontier = {UUID(eid) for eid in entity_ids}
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

    uuid_ids = [UUID(eid) for eid in entity_ids]

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

    conv_uuid_ids = {UUID(cid) for cid in conversation_ids}
    entity_uuid_ids = {UUID(eid) for eid in entity_ids}

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


def extract_query_entities(query: str) -> list[str]:
    """
    Extract entity names from a user query using lightweight LLM call.

    Args:
        query: User search query

    Returns:
        List of extracted entity names
    """
    # For now, use simple heuristics to avoid LLM latency
    # Can be upgraded to LLM-based extraction later
    entities = []

    # Simple extraction: capitalize words that look like names
    words = query.split()
    current_entity = []

    for word in words:
        # Skip common stop words
        if word.lower() in {
            "the",
            "a",
            "an",
            "and",
            "or",
            "for",
            "to",
            "from",
            "in",
            "on",
            "at",
            "by",
            "with",
            "about",
            "find",
            "search",
            "show",
            "get",
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
        }:
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
            continue

        # Look for capitalized words or quoted strings
        if word[0].isupper() or word.startswith('"') or word.startswith("'"):
            current_entity.append(word.strip("\"'"))
        elif current_entity:
            # End of entity
            entities.append(" ".join(current_entity))
            current_entity = []

    if current_entity:
        entities.append(" ".join(current_entity))

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for e in entities:
        if e.lower() not in seen:
            seen.add(e.lower())
            unique.append(e)

    return unique


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
    """Async wrapper for graph retrieval to avoid blocking the event loop."""
    try:
        return await asyncio.to_thread(_graph_retrieve_sync, args)
    except Exception:
        logger.exception("Graph retrieval failed")
        return Err("Graph retrieval failed")
