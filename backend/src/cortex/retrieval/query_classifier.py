"""
Query Classifier.

Implements §8.2 of the Canonical Blueprint.

Classifies user queries into:
- "navigational": looking for specific email/sender/subject (FTS fast-path)
- "semantic": analytical question requiring understanding (hybrid search)
- "drafting": request to compose/reply to email (draft flow)
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import List, Literal

from cortex.observability import trace_operation
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QueryClassification(BaseModel):
    """
    Query classification result.

    Blueprint §8.1:
    * query: str - the original query
    * type: navigational | semantic | drafting
    * flags: list of additional flags (followup, requires_grounding_check, time_sensitive)
    """

    query: str = Field(..., description="The original query")
    type: Literal["navigational", "semantic", "drafting"] = Field(
        ..., description="Query type determining retrieval strategy"
    )
    flags: List[str] = Field(
        default_factory=list,
        description="Additional flags: followup, requires_grounding_check, time_sensitive",
    )


class QueryClassificationInput(BaseModel):
    """Input payload for tool_classify_query."""

    query: str
    use_llm: bool = False


# -----------------------------------------------------------------------------
# Fast Pattern-Based Classification (no LLM)
# -----------------------------------------------------------------------------

# Patterns indicating navigational queries (specific lookup)
NAVIGATIONAL_PATTERNS = [
    # Sender/recipient lookups
    # Allow common email variants like "+tag" and underscores
    r"\b(?:from|sent by|email from)\s+[\w\.\-\+]+@[\w\.\-]+",
    r"\b(?:to|sent to|email to)\s+[\w\.\-\+]+@[\w\.\-]+",
    r"\b(?:from|by)\s+[A-Z][a-z]+\s+[A-Z][a-z]+",  # "from John Smith"
    # Subject lookups
    r"\b(?:subject|titled|called|named)\s*[:\"\']\s*.+[\"\']?",
    r"\bRE:\s*.+",
    r"\bFW:\s*.+",
    # Date-specific lookups
    r"\b(?:on|from|dated)\s+(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2})",
    r"\b(?:last|this)\s+(?:week|month|monday|tuesday|wednesday|thursday|friday)",
    r"\byesterday\b",
    r"\btoday\b",
    # Specific document lookups
    r"\b(?:find|show|get|retrieve)\s+(?:the|my|that)\s+email",
    r"\bthread\s+(?:about|regarding|with)\b",
    r"\b(?:attachment|file)\s+(?:called|named|titled)\b",
]

# Patterns indicating drafting requests
DRAFTING_PATTERNS = [
    r"\b(?:draft|write|compose|create|prepare)\s+(?:a|an)?\s*(?:email|reply|response|message)",
    r"\breply to\b",
    r"\brespond to\b",
    r"\bsend\s+(?:a|an)\s+(?:email|message)",
    r"\b(?:help me|can you)\s+(?:write|draft|reply)",
]

# Patterns indicating time sensitivity
TIME_SENSITIVE_PATTERNS = [
    r"\burgent\b",
    r"\basap\b",
    r"\bimmediately\b",
    r"\bdeadline\b",
    r"\bby\s+(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|end of day|eod|cob)",
]

# Patterns indicating follow-up context
FOLLOWUP_PATTERNS = [
    r"\bfollow.?up\b",
    r"\bprevious\b",
    r"\bearlier\b",
    r"\bcontinuing\b",
    r"\bregarding\s+(?:my|our|the)\s+(?:last|previous)\b",
]

# Patterns indicating need for grounding check (compliance/legal)
GROUNDING_CHECK_PATTERNS = [
    r"\b(?:policy|policies|regulation|regulations|compliance|legal)\b",
    r"\b(?:allowed|prohibited|permitted|forbidden|required)\b",
    r"\b(?:must|shall|may not|cannot)\b",
    r"\b(?:confirm|verify|validate)\s+(?:that|if|whether)\b",
]


# Compiled patterns
_NAVIGATIONAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in NAVIGATIONAL_PATTERNS]
_DRAFTING_PATTERNS = [re.compile(p, re.IGNORECASE) for p in DRAFTING_PATTERNS]
_TIME_SENSITIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in TIME_SENSITIVE_PATTERNS
]
_FOLLOWUP_PATTERNS = [re.compile(p, re.IGNORECASE) for p in FOLLOWUP_PATTERNS]
_GROUNDING_CHECK_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in GROUNDING_CHECK_PATTERNS
]


def _match_any_pattern(text: str, patterns: List[re.Pattern]) -> bool:
    """Check if text matches any of the given compiled patterns."""
    for pattern in patterns:
        if pattern.search(text):
            return True
    return False


def classify_query_fast(query: str) -> QueryClassification:
    """
    Fast pattern-based query classification without LLM.

    Blueprint §8.2:
    Uses regex patterns for common query types.
    Falls back to semantic for ambiguous queries.

    Args:
        query: The user's query string

    Returns:
        QueryClassification with type and flags
    """
    query = query.strip()
    flags: List[str] = []

    # Detect flags first
    if _match_any_pattern(query, _TIME_SENSITIVE_PATTERNS):
        flags.append("time_sensitive")

    if _match_any_pattern(query, _FOLLOWUP_PATTERNS):
        flags.append("followup")

    if _match_any_pattern(query, _GROUNDING_CHECK_PATTERNS):
        flags.append("requires_grounding_check")

    # Determine query type
    if _match_any_pattern(query, _DRAFTING_PATTERNS):
        return QueryClassification(query=query, type="drafting", flags=flags)

    if _match_any_pattern(query, _NAVIGATIONAL_PATTERNS):
        return QueryClassification(query=query, type="navigational", flags=flags)

    # Default to semantic for analytical/understanding queries
    return QueryClassification(query=query, type="semantic", flags=flags)


# -----------------------------------------------------------------------------
# LLM-Based Classification (more accurate, higher latency)
# -----------------------------------------------------------------------------


def classify_query_llm(query: str) -> QueryClassification:
    """
    LLM-based query classification for ambiguous cases.

    Uses the LLM to understand query intent when patterns are insufficient.

    Args:
        args: QueryClassificationInput carrying the user's query and related options

    Returns:
        QueryClassification with type and flags
    """
    try:
        from cortex.common.exceptions import (
            CircuitBreakerOpenError,
            LLMOutputSchemaError,
            ProviderError,
            RateLimitError,
        )
        from cortex.llm.client import complete_json
        from cortex.prompts import PROMPT_QUERY_CLASSIFY

        # Wrap user input in XML tags to mitigate prompt injection.
        prompt = f"{PROMPT_QUERY_CLASSIFY}\n\n<user_query>{query}</user_query>"

        schema = QueryClassification.model_json_schema()
        result = complete_json(prompt, schema)

        return QueryClassification(**result)

    except (
        ProviderError,
        RateLimitError,
        CircuitBreakerOpenError,
        LLMOutputSchemaError,
    ) as e:
        logger.warning(f"LLM classification failed: {e}. Using fast fallback.")
        return classify_query_fast(query)
    except Exception as e:
        logger.error(f"Unexpected error during LLM classification: {e}", exc_info=True)
        return classify_query_fast(query)


# -----------------------------------------------------------------------------
# Main Tool Interface (Blueprint §8.2)
# -----------------------------------------------------------------------------


@trace_operation("tool_classify_query")
@lru_cache(maxsize=128)
def tool_classify_query(query: str, use_llm: bool = False) -> QueryClassification:
    """
    Classify a user query for routing to appropriate retrieval strategy.

    Blueprint §8.2:
    * Determines if query is navigational (FTS fast-path),
      semantic (hybrid search), or drafting (draft flow)
    * Detects flags: followup, requires_grounding_check, time_sensitive

    Args:
        query: The user's query string
        use_llm: Whether to use LLM for classification (default: False)
                 Set True for ambiguous queries needing deeper understanding

    Returns:
        QueryClassification with type and flags
    """
    query = query.strip()

    if not query:
        return QueryClassification(query=query or "", type="semantic", flags=[])

    if use_llm:
        return classify_query_llm(query)

    return classify_query_fast(query)


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------


def is_navigational(query: str) -> bool:
    """Quick check if query is navigational (specific lookup)."""
    classification = tool_classify_query(query, use_llm=False)
    return classification.type == "navigational"


def is_drafting(query: str) -> bool:
    """Quick check if query is a drafting request."""
    classification = tool_classify_query(query, use_llm=False)
    return classification.type == "drafting"


def requires_grounding_check(query: str) -> bool:
    """Check if query involves compliance/legal content requiring grounding."""
    classification = tool_classify_query(query, use_llm=False)
    return "requires_grounding_check" in classification.flags


__all__ = [
    "QueryClassification",
    "tool_classify_query",
    "classify_query_fast",
    "classify_query_llm",
    "is_navigational",
    "is_drafting",
    "requires_grounding_check",
]
