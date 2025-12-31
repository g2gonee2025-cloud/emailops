"""
Search Filters and Grammar Parser.

Implements rich filtering capabilities for search queries, ported from
reference code feature_search_draft.py.

Supports filters:
- from:email@example.com
- to:email@example.com
- cc:email@example.com
- subject:"keyword"
- after:2024-01-01
- before:2024-12-31
- has:attachment
- type:pdf,docx
- -excludeterm (minus prefix for exclusion)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Python <3.11 compatibility
try:
    from datetime import UTC  # type: ignore
except ImportError:
    UTC = timezone.utc  # type: ignore


@dataclass
class SearchFilters:
    """
    Structured search filters extracted from query.

    Ported from reference code feature_search_draft.py.
    """

    # Conversation/thread filtering
    conv_ids: set[str] | None = None

    # Participant filtering
    from_emails: set[str] | None = None
    to_emails: set[str] | None = None
    cc_emails: set[str] | None = None

    # Content filtering
    subject_contains: list[str] | None = None
    exclude_terms: list[str] | None = None

    # Attachment filtering
    has_attachment: bool | None = None
    file_types: set[str] | None = None  # {'pdf', 'docx', ...}

    # Date range filtering
    date_from: datetime | None = None
    date_to: datetime | None = None

    def is_empty(self) -> bool:
        """Check if no filters are set."""
        return (
            self.conv_ids is None
            and self.from_emails is None
            and self.to_emails is None
            and self.cc_emails is None
            and self.subject_contains is None
            and self.exclude_terms is None
            and self.has_attachment is None
            and self.file_types is None
            and self.date_from is None
            and self.date_to is None
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {}
        if self.conv_ids:
            result["conv_ids"] = list(self.conv_ids)
        if self.from_emails:
            result["from_emails"] = list(self.from_emails)
        if self.to_emails:
            result["to_emails"] = list(self.to_emails)
        if self.cc_emails:
            result["cc_emails"] = list(self.cc_emails)
        if self.subject_contains:
            result["subject_contains"] = self.subject_contains
        if self.exclude_terms:
            result["exclude_terms"] = self.exclude_terms
        if self.has_attachment is not None:
            result["has_attachment"] = self.has_attachment
        if self.file_types:
            result["file_types"] = list(self.file_types)
        if self.date_from:
            result["date_from"] = self.date_from.isoformat()
        if self.date_to:
            result["date_to"] = self.date_to.isoformat()
        return result


# Regex for extracting filter tokens from query
# Matches: key:value or key:"quoted value"
_FILTER_TOKEN_RE = re.compile(
    r'(?P<key>subject|from|to|cc|after|before|has|type):(?P<value>"[^"]+"|[^\s]+)',
    re.IGNORECASE,
)


def _parse_iso_date(s: str) -> datetime | None:
    """Parse ISO date string to datetime."""
    s = s.strip()
    if not s:
        return None
    try:
        # Handle Z suffix
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        try:
            # Fallback: try email utils
            from email.utils import parsedate_to_datetime

            dt = parsedate_to_datetime(s)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except (ValueError, TypeError):
            return None


def parse_filter_grammar(raw_query: str) -> tuple[SearchFilters, str]:
    """
    Parse filter tokens from a query string.

    Returns (filters, cleaned_query) where:
    - filters: extracted SearchFilters
    - cleaned_query: query with filter tokens removed

    Supported syntax:
    - from:john@example.com
    - to:jane@example.com
    - cc:team@example.com
    - subject:"budget report"
    - after:2024-01-01
    - before:2024-12-31
    - has:attachment
    - type:pdf,docx
    - -excludeterm (excluded from results)

    Example:
        >>> filters, clean = parse_filter_grammar("from:john@test.com budget report")
        >>> filters.from_emails
        {'john@test.com'}
        >>> clean
        'budget report'
    """
    f = SearchFilters()
    q = raw_query or ""

    # Find all filter tokens
    tokens = list(_FILTER_TOKEN_RE.finditer(q))

    # Remove tokens from query string (in reverse to preserve positions)
    cleaned = q
    for m in reversed(tokens):
        start, end = m.span()
        cleaned = cleaned[:start] + cleaned[end:]

    # Collapse whitespace
    cleaned = " ".join(cleaned.split())

    # Extract exclusion terms (words starting with -)
    words = cleaned.split()
    exclude_terms = [w[1:].lower() for w in words if w.startswith("-") and len(w) > 1]
    if exclude_terms:
        f.exclude_terms = exclude_terms
        cleaned = " ".join(w for w in words if not w.startswith("-"))

    # Process filter tokens
    for m in tokens:
        key = m.group("key").lower()
        val = m.group("value")

        # Remove quotes from quoted values
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        val = val.strip()

        if not val:
            continue

        if key == "subject":
            if f.subject_contains is None:
                f.subject_contains = []
            f.subject_contains.append(val.lower())

        elif key == "from":
            if f.from_emails is None:
                f.from_emails = set()
            f.from_emails.add(val.lower())

        elif key == "to":
            if f.to_emails is None:
                f.to_emails = set()
            f.to_emails.add(val.lower())

        elif key == "cc":
            if f.cc_emails is None:
                f.cc_emails = set()
            f.cc_emails.add(val.lower())

        elif key == "after":
            parsed = _parse_iso_date(val)
            if parsed:
                f.date_from = parsed

        elif key == "before":
            parsed = _parse_iso_date(val)
            if parsed:
                f.date_to = parsed

        elif key == "has":
            val_lower = val.lower()
            if val_lower in {"attachment", "attachments"}:
                f.has_attachment = True
            elif val_lower in {"noattachment", "no-attachment", "none"}:
                f.has_attachment = False

        elif key == "type":
            # Support comma-separated extensions
            raw_exts = {
                e.strip().lower().lstrip(".") for e in val.split(",") if e.strip()
            }
            # SECURITY: Validate extensions to be simple alphanumeric strings to prevent path traversal
            # or other injection attacks in downstream consumers.
            valid_exts = {ext for ext in raw_exts if re.match(r"^[a-z0-9]+$", ext)}
            if f.file_types is None:
                f.file_types = set()
            f.file_types.update(valid_exts)

    return f, cleaned


def apply_filters_to_sql(
    filters: SearchFilters,
    table_alias: str = "c",
    conversation_table_alias: str = "conv",
) -> tuple[str, dict[str, Any]]:
    """
    Generate SQL WHERE clause fragments and params for filters.

    Returns (where_clause, params) to be appended to base query.

    Note: This requires the query to join with conversations table
    if using from/to/cc/subject filters.
    """
    # SECURITY: Prevent SQL injection by validating table aliases.
    # Only allow simple alphanumeric aliases.
    if not re.match(r"^[a-zA-Z0-9_]+$", table_alias):
        raise ValueError(f"Invalid table alias: {table_alias}")
    if not re.match(r"^[a-zA-Z0-9_]+$", conversation_table_alias):
        raise ValueError(
            f"Invalid conversation table alias: {conversation_table_alias}"
        )

    conditions: list[str] = []
    params: dict[str, Any] = {}

    # Date range filters (assuming chunk has related conversation with date)
    if filters.date_from:
        conditions.append(f"{conversation_table_alias}.latest_date >= :date_from")
        params["date_from"] = filters.date_from

    if filters.date_to:
        conditions.append(f"{conversation_table_alias}.latest_date <= :date_to")
        params["date_to"] = filters.date_to

    # Attachment filter
    if filters.has_attachment is True:
        conditions.append(f"{table_alias}.is_attachment = TRUE")
    elif filters.has_attachment is False:
        conditions.append(f"{table_alias}.is_attachment = FALSE")

    # File type filter on chunk metadata
    if filters.file_types:
        conditions.append(
            f"lower({table_alias}.metadata->>'file_type') = ANY(:file_types)"
        )
        params["file_types"] = list(filters.file_types)

    # Subject contains (uses ILIKE for case-insensitive)
    if filters.subject_contains:
        for i, term in enumerate(filters.subject_contains):
            param_name = f"subject_term_{i}"
            conditions.append(f"{conversation_table_alias}.subject ILIKE :{param_name}")
            params[param_name] = f"%{term}%"

    # Exclude terms (NOT ILIKE on text)
    if filters.exclude_terms:
        for i, term in enumerate(filters.exclude_terms):
            param_name = f"exclude_term_{i}"
            conditions.append(f"{table_alias}.text NOT ILIKE :{param_name}")
            params[param_name] = f"%{term}%"

    # Conversation ID filter
    if filters.conv_ids:
        conditions.append(f"{table_alias}.conversation_id = ANY(:conv_ids)")
        params["conv_ids"] = list(filters.conv_ids)

    # Participant filters using JSONB subqueries for performance and security.
    if filters.from_emails:
        conditions.append(
            f"EXISTS (SELECT 1 FROM jsonb_array_elements({conversation_table_alias}.participants) p "
            "WHERE p->>'role' = 'sender' AND lower(p->>'smtp') = ANY(:from_emails))",
        )
        params["from_emails"] = [e.lower() for e in filters.from_emails]

    if filters.to_emails:
        conditions.append(
            f"EXISTS (SELECT 1 FROM jsonb_array_elements({conversation_table_alias}.participants) p "
            "WHERE p->>'role' = 'recipient' AND lower(p->>'smtp') = ANY(:to_emails))",
        )
        params["to_emails"] = [e.lower() for e in filters.to_emails]

    if filters.cc_emails:
        conditions.append(
            f"EXISTS (SELECT 1 FROM jsonb_array_elements({conversation_table_alias}.participants) p "
            "WHERE p->>'role' = 'cc' AND lower(p->>'smtp') = ANY(:cc_emails))",
        )
        params["cc_emails"] = [e.lower() for e in filters.cc_emails]

    where_clause = " AND ".join(conditions) if conditions else ""
    return where_clause, params
