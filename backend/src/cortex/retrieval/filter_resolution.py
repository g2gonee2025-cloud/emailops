import logging
from typing import List, Optional, Set

from sqlalchemy import or_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Query

from cortex.db.models import Conversation
from cortex.retrieval.filters import SearchFilters

logger = logging.getLogger(__name__)


def _add_email_filters(query: Query, emails: Set[str]) -> Query:
    """Adds email filters to the query."""
    if not emails:
        return query

    clauses = []
    for email in emails:
        clauses.append(Conversation.participants.contains([{"email": email}]))
        # Also try "smtp" key if schema varies
        clauses.append(Conversation.participants.contains([{"smtp": email}]))

    if clauses:
        return query.filter(or_(*clauses))

    return query


def _resolve_filter_conversation_ids(
    session,
    filters: SearchFilters,
    tenant_id: str,
) -> Optional[List[str]]:
    """
    Resolve filters to a list of conversation IDs.

    Handles:
    - conv_ids (explicit)
    - date_from / date_to
    - from_emails / to_emails / cc_emails
    - subject_contains

    Returns None if no filters restrict the conversation set.
    Returns empty list if filters match nothing.
    """
    if filters.is_empty():
        return None

    # If we only have conv_ids and nothing else, return them directly
    if (
        filters.conv_ids
        and not filters.date_from
        and not filters.date_to
        and not filters.from_emails
        and not filters.to_emails
        and not filters.cc_emails
        and not filters.subject_contains
    ):
        return list(filters.conv_ids)

    # Start simply - we can construct a SQL query dynamically or use ORM
    # using ORM for safety and readability
    # We always need to filter by tenant
    query = session.query(Conversation.conversation_id).filter(
        Conversation.tenant_id == tenant_id
    )

    has_restrictions = False

    if filters.conv_ids:
        query = query.filter(Conversation.conversation_id.in_(filters.conv_ids))
        has_restrictions = True

    if filters.date_from:
        query = query.filter(Conversation.latest_date >= filters.date_from)
        has_restrictions = True

    if filters.date_to:
        query = query.filter(Conversation.latest_date <= filters.date_to)
        has_restrictions = True

    if filters.subject_contains:
        for term in filters.subject_contains:
            query = query.filter(Conversation.subject.ilike(f"%%{term}%%"))
        has_restrictions = True

    # Participant filters (JSONB)
    if filters.from_emails:
        query = _add_email_filters(query, filters.from_emails)
        has_restrictions = True

    if filters.to_emails:
        query = _add_email_filters(query, filters.to_emails)
        has_restrictions = True

    if filters.cc_emails:
        query = _add_email_filters(query, filters.cc_emails)
        has_restrictions = True

    if not has_restrictions:
        return None

    # Execute
    try:
        result_ids = [str(row[0]) for row in query.all()]
        return result_ids
    except SQLAlchemyError as e:
        logger.error(f"Failed to resolve conversation IDs from DB: {e}")
        # If DB query fails, fall back strictly to filter IDs if present, or return empty
        if filters.conv_ids:
            return list(filters.conv_ids)
        return []
