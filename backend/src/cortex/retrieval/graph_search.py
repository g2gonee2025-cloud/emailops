"""
Graph-based Retrieval for Knowledge Graph RAG.

Implements entity-aware search that leverages the EntityNode and EntityEdge
tables to find contextually related conversations via graph traversal.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from cortex.common.types import Err, Ok, Result
from cortex.db.models import EntityEdge, EntityNode
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Constants
DEFAULT_ENTITY_LIMIT = 10
DEFAULT_MAX_HOPS = 1
MIN_TRIGRAM_SIMILARITY = 0.3


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

    matches: list[EntityMatch] = []

    # Build ILIKE conditions for exact/partial matching
    ilike_conditions = [
        func.lower(EntityNode.name).contains(func.lower(name)) for name in entity_names
    ]

    # First try: exact/substring match
    exact_stmt = (
        select(EntityNode)
        .where(EntityNode.tenant_id == tenant_id, or_(*ilike_conditions))
        .order_by(EntityNode.pagerank.desc())
        .limit(limit)
    )

    exact_results = session.execute(exact_stmt).scalars().all()

    for node in exact_results:
        matches.append(
            EntityMatch(
                node_id=str(node.node_id),
                name=node.name,
                entity_type=node.type,
                pagerank=node.pagerank if hasattr(node, "pagerank") else 0.0,
                similarity=1.0,
            )
        )

    # If we have enough matches, return
    if len(matches) >= limit:
        return matches[:limit]

    # Second try: trigram similarity for fuzzy matching
    remaining = limit - len(matches)
    existing_ids = {m.node_id for m in matches}

    for name in entity_names:
        if remaining <= 0:
            break

        # Use pg_trgm similarity function
        trgm_stmt = text(
            """
            SELECT node_id, name, type,
                   COALESCE(pagerank, 0.0) as pagerank,
                   similarity(name, :search_name) as sim
            FROM entity_nodes
            WHERE tenant_id = :tenant_id
              AND similarity(name, :search_name) > :min_sim
              AND node_id NOT IN :existing_ids
            ORDER BY sim DESC, pagerank DESC
            LIMIT :limit
        """
        )

        try:
            trgm_results = session.execute(
                trgm_stmt,
                {
                    "tenant_id": tenant_id,
                    "search_name": name,
                    "min_sim": MIN_TRIGRAM_SIMILARITY,
                    "existing_ids": tuple(existing_ids) if existing_ids else ("",),
                    "limit": remaining,
                },
            ).fetchall()

            for row in trgm_results:
                node_id = str(row.node_id)
                if node_id not in existing_ids:
                    matches.append(
                        EntityMatch(
                            node_id=node_id,
                            name=row.name,
                            entity_type=row.type,
                            pagerank=float(row.pagerank),
                            similarity=float(row.sim),
                        )
                    )
                    existing_ids.add(node_id)
                    remaining -= 1
        except Exception as e:
            logger.warning(f"Trigram search failed for '{name}': {e}")
            continue

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
        List of (edge, target_node) tuples
    """
    if not entity_ids or max_hops < 1:
        return []

    # Convert to UUIDs
    uuid_ids = [UUID(eid) for eid in entity_ids]

    # Single hop: find edges where source is in our entity set
    edges_stmt = (
        select(EntityEdge, EntityNode)
        .join(EntityNode, EntityEdge.target_id == EntityNode.node_id)
        .where(
            EntityEdge.tenant_id == tenant_id,
            EntityEdge.source_id.in_(uuid_ids),
        )
        .order_by(EntityEdge.weight.desc())
        .limit(100)  # Cap to prevent explosion
    )

    results = session.execute(edges_stmt).all()

    # Also get reverse edges (where target is in our set)
    reverse_stmt = (
        select(EntityEdge, EntityNode)
        .join(EntityNode, EntityEdge.source_id == EntityNode.node_id)
        .where(
            EntityEdge.tenant_id == tenant_id,
            EntityEdge.target_id.in_(uuid_ids),
        )
        .order_by(EntityEdge.weight.desc())
        .limit(100)
    )

    reverse_results = session.execute(reverse_stmt).all()

    # Combine and deduplicate
    all_results = list(results) + list(reverse_results)
    seen_edges = set()
    unique_results = []

    for edge, node in all_results:
        if edge.edge_id not in seen_edges:
            seen_edges.add(edge.edge_id)
            unique_results.append((edge, node))

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

    # Find conversations via edges (both source and target)
    conv_stmt = text(
        """
        SELECT conversation_id, COUNT(*) as conn_count
        FROM entity_edges
        WHERE tenant_id = :tenant_id
          AND conversation_id IS NOT NULL
          AND (source_id = ANY(:entity_ids) OR target_id = ANY(:entity_ids))
        GROUP BY conversation_id
        ORDER BY conn_count DESC
        LIMIT :limit
    """
    )

    results = session.execute(
        conv_stmt,
        {
            "tenant_id": tenant_id,
            "entity_ids": uuid_ids,
            "limit": limit,
        },
    ).fetchall()

    return [(str(row.conversation_id), row.conn_count) for row in results]


async def extract_query_entities(query: str) -> list[str]:
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


async def graph_retrieve(
    session: Session,
    args: GraphSearchInput,
) -> Result[list[GraphSearchResult], str]:
    """
    Perform graph-based retrieval to find related conversations.

    This is the main entry point for graph search, orchestrating:
    1. Entity extraction from query (if not provided)
    2. Entity name matching in the graph
    3. Neighbor expansion via edges
    4. Conversation ID retrieval

    Args:
        session: Database session
        args: Graph search input parameters

    Returns:
        Result containing list of GraphSearchResult or error string
    """
    try:
        # 1. Extract entities from query if not provided
        entity_names = args.entity_names
        if not entity_names:
            entity_names = await extract_query_entities(args.query)

        if not entity_names:
            logger.debug(f"No entities extracted from query: {args.query}")
            return Ok([])

        logger.info(f"Graph search with entities: {entity_names}")

        # 2. Search for entities in the graph
        matched_entities = search_entities_by_name(
            session,
            args.tenant_id,
            entity_names,
            limit=DEFAULT_ENTITY_LIMIT,
        )

        if not matched_entities:
            logger.debug(f"No entity matches found for: {entity_names}")
            return Ok([])

        logger.info(
            f"Found {len(matched_entities)} entity matches: "
            f"{[e.name for e in matched_entities[:5]]}"
        )

        entity_ids = [e.node_id for e in matched_entities]

        # 3. Expand via edges to get related entities
        neighbors = expand_entity_neighbors(
            session, entity_ids, args.tenant_id, max_hops=args.max_hops
        )

        # Add neighbor entity IDs to our set
        for _, node in neighbors:
            if str(node.node_id) not in entity_ids:
                entity_ids.append(str(node.node_id))

        # 4. Get conversations connected to these entities
        conv_results = get_conversations_from_entities(
            session, entity_ids, args.tenant_id, limit=args.k
        )

        # 5. Build result objects
        results: list[GraphSearchResult] = []

        for conv_id, conn_count in conv_results:
            # Find the best matching entity for context
            best_entity = matched_entities[0] if matched_entities else None

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

        logger.info(f"Graph retrieval returned {len(results)} conversations")
        return Ok(results)

    except Exception as e:
        logger.exception("Graph retrieval failed")
        return Err(f"Graph retrieval failed: {e}")
