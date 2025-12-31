import logging
from typing import Optional

from cortex.db.models import Conversation
from cortex.retrieval.filters import SearchFilters
from sqlalchemy import or_
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


# Safeguard against queries that could return a huge number of conversation IDs
MAX_FILTER_CONVERSATION_IDS = 1000
MAX_EMAIL_FILTERS = 50


def _sanitize_like_term(term: str) -> str:
    """Sanitize a term for use in a LIKE query by escaping wildcards."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _add_email_filters(query, emails: set[str], *, role: str | None = None):
    """
    Adds email filters to the query.

    NOTE: For performance, a GIN index on the `participants` JSONB column
    is highly recommended. Example migration:
    op.create_index(
        'ix_conversations_participants_gin',
        'conversations',
        ['participants'],
        postgresql_using='gin'
    )
    """
    if not emails:
        return query

    emails_list = sorted(emails)
    if len(emails_list) > MAX_EMAIL_FILTERS:
        logger.warning(
            "Email filter set too large (%s); truncating to %s",
            len(emails_list),
            MAX_EMAIL_FILTERS,
        )
        emails_list = emails_list[:MAX_EMAIL_FILTERS]

    # Creates a series of OR clauses to find any of the provided emails
    # in the participants JSONB array.
    if role:
        clauses = [
            Conversation.participants.contains([{"email": email, "role": role}])
            for email in emails_list
        ] + [
            Conversation.participants.contains([{"smtp": email, "role": role}])
            for email in emails_list
        ]
    else:
        clauses = [
            Conversation.participants.contains([{"email": email}])
            for email in emails_list
        ] + [
            Conversation.participants.contains([{"smtp": email}])
            for email in emails_list
        ]

    return query.filter(or_(*clauses))


def _resolve_filter_conversation_ids(
    session,
    filters: SearchFilters,
    tenant_id: str,
) -> list[str] | None:
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
            if not isinstance(term, str):
                continue
            sanitized_term = _sanitize_like_term(term)
            query = query.filter(
                Conversation.subject.ilike(f"%{sanitized_term}%", escape="\\")
            )
        has_restrictions = True

    if filters.from_emails:
        query = _add_email_filters(query, filters.from_emails, role="from")
        has_restrictions = True
    if filters.to_emails:
        query = _add_email_filters(query, filters.to_emails, role="to")
        has_restrictions = True
    if filters.cc_emails:
        query = _add_email_filters(query, filters.cc_emails, role="cc")
        has_restrictions = True

    if not has_restrictions:
        return None

    # Execute
    try:
        # Add a safeguard limit to prevent unbounded queries from overwhelming
        # the system.
        rows = query.limit(MAX_FILTER_CONVERSATION_IDS + 1).all()

        if len(rows) > MAX_FILTER_CONVERSATION_IDS:
            logger.warning(
                "Filter resolution query returned more than the safeguard limit of %s "
                "conversation IDs. Results have been truncated.",
                MAX_FILTER_CONVERSATION_IDS,
            )
            rows = rows[:MAX_FILTER_CONVERSATION_IDS]

        result_ids = [str(row[0]) for row in rows]
        return result_ids
    except SQLAlchemyError:
        logger.exception("Failed to resolve conversation IDs from DB")
        # Re-raise the exception to be handled by the global error handler.
        # Silently returning partial data can lead to inconsistencies.
        raise
