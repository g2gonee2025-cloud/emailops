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
from typing import Any, Dict, List, Optional, Set

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
    conv_ids: Optional[Set[str]] = None

    # Participant filtering
    from_emails: Optional[Set[str]] = None
    to_emails: Optional[Set[str]] = None
    cc_emails: Optional[Set[str]] = None

    # Content filtering
    subject_contains: Optional[List[str]] = None
    exclude_terms: Optional[List[str]] = None

    # Attachment filtering
    has_attachment: Optional[bool] = None
    file_types: Optional[Set[str]] = None  # {'pdf', 'docx', ...}

    # Date range filtering
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {}
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


def _parse_iso_date(s: str) -> Optional[datetime]:
    """Parse ISO date string to datetime."""
    s = s.strip()
    if not s:
        return None
    try:
        # Handle Z suffix
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        try:
            # Fallback: try email utils
            from email.utils import parsedate_to_datetime

            return parsedate_to_datetime(s).astimezone(UTC)
        except Exception:
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
            f.subject_contains = (f.subject_contains or []) + [val.lower()]

        elif key == "from":
            f.from_emails = (f.from_emails or set()) | {val.lower()}

        elif key == "to":
            f.to_emails = (f.to_emails or set()) | {val.lower()}

        elif key == "cc":
            f.cc_emails = (f.cc_emails or set()) | {val.lower()}

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
            exts = {e.strip().lower().lstrip(".") for e in val.split(",") if e.strip()}
            f.file_types = (f.file_types or set()) | exts

    return f, cleaned


def apply_filters_to_sql(
    filters: SearchFilters,
    table_alias: str = "c",
    conversation_table_alias: str = "conv",
) -> tuple[str, Dict[str, Any]]:
    """
    Generate SQL WHERE clause fragments and params for filters.

    Returns (where_clause, params) to be appended to base query.

    Note: This requires the query to join with conversations table
    if using from/to/cc/subject filters.
    """
    conditions: List[str] = []
    params: Dict[str, Any] = {}

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

    # Note: from_emails, to_emails, cc_emails require parsing conversation.participants
    # which is stored as JSONB. This is more complex and depends on schema.
    # For now, these are handled via post-filtering or extended schema.

    where_clause = " AND ".join(conditions) if conditions else ""
    return where_clause, params


def filter_results_post_query(
    results: List[Dict[str, Any]],
    filters: SearchFilters,
) -> List[Dict[str, Any]]:
    """
    Post-filter results that couldn't be filtered in SQL.

    Handles from/to/cc participant filtering using metadata.
    """
    if filters.is_empty():
        return results

    filtered = []
    for r in results:
        # Get participants from metadata
        metadata = r.get("metadata", {}) or {}
        participants = metadata.get("participants", [])

        # Normalize participant list
        if isinstance(participants, list):
            emails = {
                p.get("email", "").lower() for p in participants if isinstance(p, dict)
            }
        else:
            emails = set()

        # From filter (check if any participant matches)
        if filters.from_emails:
            from_match = emails & filters.from_emails
            if not from_match:
                continue

        # To filter
        if filters.to_emails:
            to_match = emails & filters.to_emails
            if not to_match:
                continue

        # CC filter
        if filters.cc_emails:
            cc_match = emails & filters.cc_emails
            if not cc_match:
                continue

        filtered.append(r)

    return filtered
