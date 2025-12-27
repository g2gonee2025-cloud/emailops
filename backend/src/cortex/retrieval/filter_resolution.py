from typing import List, Optional

from cortex.retrieval.filters import SearchFilters


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

    from cortex.db.models import Conversation
    from sqlalchemy import or_

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
            query = query.filter(Conversation.subject.ilike(f"%{term}%"))
        has_restrictions = True

    # Participant filters (JSONB)
    # participants is list of dicts: [{"name":..., "email":...}]
    # We use jsonb_path_exists or similar.
    # Postgres JSONB containment: participants @> '[{"email": "foo"}]'

    if filters.from_emails:
        # Note: In our schema, we don't strictly distinguish from/to in the participants list
        # We just have a list of participants.
        # Ideally we check message-level data but that's in `messages` JSONB column.
        # Check if ANY participant matches.
        # OR: Check `messages` JSONB for `from` field?
        # `messages` is list of objects.
        # This is expensive. For now, let's assume filtering by "participant in thread" is close enough.
        # OR: strictly use participants list for "people involved".

        # Construct OR clause for emails
        # participants @> '[{"email": "email1"}]' OR ...
        clauses = []
        for email in filters.from_emails:
            clauses.append(Conversation.participants.contains([{"email": email}]))
            # Also try "smtp" key if schema varies
            clauses.append(Conversation.participants.contains([{"smtp": email}]))

        if clauses:
            query = query.filter(or_(*clauses))
            has_restrictions = True

    if filters.to_emails:
        clauses = []
        for email in filters.to_emails:
            clauses.append(Conversation.participants.contains([{"email": email}]))
            clauses.append(Conversation.participants.contains([{"smtp": email}]))
        if clauses:
            query = query.filter(or_(*clauses))
            has_restrictions = True

    if filters.cc_emails:
        clauses = []
        for email in filters.cc_emails:
            clauses.append(Conversation.participants.contains([{"email": email}]))
            clauses.append(Conversation.participants.contains([{"smtp": email}]))
        if clauses:
            query = query.filter(or_(*clauses))
            has_restrictions = True

    if not has_restrictions:
        return None

    # Execute
    try:
        result_ids = [str(row[0]) for row in query.all()]
        return result_ids
    except Exception:
        # If DB query fails, fall back strictly to filter IDs if present, or return empty
        if filters.conv_ids:
            return list(filters.conv_ids)
        return []
