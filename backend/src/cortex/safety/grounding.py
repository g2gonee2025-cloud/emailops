"""
Grounding Check.

Implements §9.4 of the Canonical Blueprint.
Verifies that LLM-generated answers are supported by retrieved context/facts.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

import numpy as np
import numpy.linalg as npla
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from cortex.embeddings.client import get_embeddings_client

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DEFAULT_EMBEDDING_SUPPORT_THRESHOLD = 0.75
"""Minimum similarity for a claim to be considered supported by a fact using embeddings."""

DEFAULT_EMBEDDING_GROUNDING_THRESHOLD = 0.7
"""Minimum ratio of supported claims for an answer to be considered grounded."""

DEFAULT_KEYWORD_SUPPORT_THRESHOLD = 0.3
"""Minimum similarity for a claim to be considered supported by a fact using keywords."""

MIN_CLAIM_LENGTH = 15
"""Minimum character length for a sentence to be considered a potential claim."""

GroundingMethod = Literal["llm", "embedding", "keyword", "not_applicable"]
"""The method used for grounding verification."""


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(v1, dtype=np.float64)
    b = np.array(v2, dtype=np.float64)
    norm_a = npla.norm(a)
    norm_b = npla.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class ClaimAnalysis(BaseModel):
    """Analysis of a single factual claim."""

    claim: str = Field(..., description="The factual claim being verified")
    is_supported: bool = Field(
        ..., description="Whether the claim is supported by facts"
    )
    supporting_fact: str | None = Field(
        None, description="The fact that supports this claim"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the assessment"
    )
    method: GroundingMethod = Field(..., description="The method used for verification")


class ClaimAnalysisInput(BaseModel):
    """Input model for LLM grounding analysis results."""

    claim: str = Field(..., description="The factual claim being verified")
    is_supported: bool = Field(
        ..., description="Whether the claim is supported by facts"
    )
    supporting_fact: str | None = Field(
        None, description="The fact that supports this claim"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the assessment"
    )
    method: GroundingMethod | None = Field(
        None, description="The method used for verification"
    )


class GroundingCheck(BaseModel):
    """
    Result of grounding verification.

    Blueprint §9.1:
    * answer_candidate: str
    * is_grounded: bool
    * confidence: float
    * unsupported_claims: list[str]
    """

    answer_candidate: str = Field(..., description="The answer being verified")
    is_grounded: bool = Field(
        ..., description="Whether the answer is grounded in facts"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims not supported by facts"
    )
    claim_analyses: list[ClaimAnalysis] = Field(
        default_factory=list, description="Detailed analysis per claim"
    )
    grounding_ratio: float = Field(
        0.0, ge=0.0, le=1.0, description="Ratio of supported claims"
    )
    method: GroundingMethod = Field(
        ..., description="The primary method used for the check"
    )


class GroundingAnalysisResult(BaseModel):
    """Full grounding analysis from LLM."""

    claims: list[ClaimAnalysisInput] = Field(default_factory=list)
    overall_grounded: bool = Field(True)
    overall_confidence: float = Field(1.0)
    unsupported_claims: list[str] = Field(default_factory=list)


class GroundingCheckInput(BaseModel):
    """Input payload for tool_check_grounding."""

    answer_candidate: str = Field(..., description="The answer to verify")
    facts: list[str] = Field(default_factory=list, description="Retrieved facts")
    use_llm: bool = Field(True, description="Whether to use LLM-based grounding")


# -----------------------------------------------------------------------------
# Claim Extraction
# -----------------------------------------------------------------------------

# Patterns indicating hedged/uncertain statements (not factual claims)
HEDGE_PATTERNS = [
    r"\bmight\b",
    r"\bcould\b",
    r"\bpossibly\b",
    r"\bperhaps\b",
    r"\bmaybe\b",
    r"\bprobably\b",
    r"\blikely\b",
    r"\bseems?\b",
    r"\bappears?\b",
    r"\bI think\b",
    r"\bI believe\b",
    r"\bIt's possible\b",
]

# Compiled patterns for efficiency
_HEDGE_REGEX = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)


def is_hedged_statement(text: str) -> bool:
    """Check if a statement is hedged/uncertain."""
    return bool(_HEDGE_REGEX.search(text))


def extract_claims_simple(text: str) -> list[str]:
    """
    Extract factual claims from text using simple heuristics.

    This is a fallback when LLM-based extraction isn't available.
    Filters out:
    - Questions
    - Hedged statements
    - Very short fragments
    """
    # Split into sentences while preserving punctuation for question detection
    sentences = re.split(r"(?<=[.!?])\s+", text.strip()) if text else []

    claims = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        is_question = sentence.endswith("?")
        sentence = sentence.rstrip(" .!?").strip()

        # Skip empty or very short
        if len(sentence) < MIN_CLAIM_LENGTH:
            continue

        # Skip questions
        if is_question or sentence.lower().startswith(
            (
                "who",
                "what",
                "where",
                "when",
                "why",
                "how",
                "is ",
                "are ",
                "do ",
                "does ",
            )
        ):
            continue

        # Skip hedged statements
        if is_hedged_statement(sentence):
            continue

        # Skip meta-statements about the answer itself
        if any(
            phrase in sentence.lower()
            for phrase in [
                "based on the context",
                "according to the provided",
                "from the information",
                "i found that",
                "the sources indicate",
            ]
        ):
            continue

        claims.append(sentence)

    return claims


def extract_claims_llm(text: str) -> list[str]:
    """
    Extract factual claims using LLM.

    More accurate than heuristics but requires LLM call.
    """
    try:
        from cortex.llm.client import complete_json
        from cortex.prompts import (
            SYSTEM_EXTRACT_CLAIMS,
            USER_EXTRACT_CLAIMS,
            construct_prompt_messages,
        )
        from cortex.security.defenses import sanitize_user_input

        messages = construct_prompt_messages(
            system_prompt_template=SYSTEM_EXTRACT_CLAIMS,
            user_prompt_template=USER_EXTRACT_CLAIMS,
            text=sanitize_user_input(text),
        )

        if len(messages) < 2:
            raise ValueError("Prompt construction returned insufficient messages")

        # Reconstruct prompt for the deprecated `complete_json`
        system_content = messages[0].get("content")
        user_content = messages[1].get("content")
        if system_content is None or user_content is None:
            raise ValueError("Prompt messages missing content")
        reconstructed_prompt = f"{system_content}\n\n{user_content}"

        result = complete_json(
            prompt=reconstructed_prompt,
            schema={
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of factual claims that can be verified",
                    }
                },
                "required": ["claims"],
            },
        )

        return result.get("claims", [])
    except ImportError as e:
        logger.error(
            f"LLM dependencies not found for claim extraction: {e}. Install with `pip install cortex-llm`."
        )
        return extract_claims_simple(text)
    except Exception:
        logger.exception(
            "LLM claim extraction failed due to an API error. Using heuristics."
        )
        return extract_claims_simple(text)


# -----------------------------------------------------------------------------
# Similarity Matching
# -----------------------------------------------------------------------------


def check_claim_against_facts_keyword(
    claim: str, facts: list[str]
) -> tuple[bool, float, str | None, GroundingMethod]:
    """
    Fallback: Check claim against facts using keyword overlap.

    Less accurate than embeddings but works without LLM.
    """
    claim_words = set(re.findall(r"\b\w+\b", claim.lower()))
    claim_words -= _STOPWORDS

    if not claim_words:
        return (False, 0.0, None, "keyword")

    best_score = 0.0
    best_fact = None

    for fact in facts:
        fact_words = set(re.findall(r"\b\w+\b", fact.lower())) - _STOPWORDS
        if not fact_words:
            continue

        # Jaccard-like overlap
        overlap = len(claim_words & fact_words)
        union = len(claim_words | fact_words)
        score = overlap / union if union > 0 else 0.0

        if score > best_score:
            best_score = score
            best_fact = fact

    # Lower threshold for keyword matching
    is_supported = best_score >= DEFAULT_KEYWORD_SUPPORT_THRESHOLD
    return (is_supported, best_score, best_fact if is_supported else None, "keyword")


def check_claim_against_facts_embedding(
    claim: str,
    facts: list[str],
    threshold: float = DEFAULT_EMBEDDING_SUPPORT_THRESHOLD,
    fact_embeddings: list[list[float]] | None = None,
) -> tuple[bool, float, str | None, GroundingMethod]:
    """
    Check a single claim against facts using embedding similarity.

    Args:
        claim: The claim to verify
        facts: List of facts from retrieved context
        threshold: Minimum similarity to consider supported
        fact_embeddings: Optional pre-computed fact embeddings

    Returns:
        (is_supported, confidence, supporting_fact, method)
    """
    if not facts:
        return (False, 0.0, None, "embedding")

    try:
        client = get_embeddings_client()

        if fact_embeddings is None:
            # Use embed_texts for efficiency
            fact_embeddings = client.embed_texts(facts)
        if not fact_embeddings:
            return (False, 0.0, None, "embedding")
        if len(fact_embeddings) != len(facts):
            logger.warning(
                "Fact embedding length mismatch (facts=%d embeddings=%d); using keyword fallback.",
                len(facts),
                len(fact_embeddings),
            )
            return check_claim_against_facts_keyword(claim, facts)

        claim_embedding = client.embed(claim)

        best_score = 0.0
        best_fact = None

        for fact, fact_emb in zip(facts, fact_embeddings):
            similarity = _cosine_similarity(claim_embedding, fact_emb)
            if similarity > best_score:
                best_score = similarity
                best_fact = fact

        is_supported = best_score >= threshold
        return (
            is_supported,
            best_score,
            best_fact if is_supported else None,
            "embedding",
        )
    except ImportError as e:
        logger.error(
            f"Embeddings client dependencies not found: {e}. Install with `pip install cortex-embeddings`."
        )
        return check_claim_against_facts_keyword(claim, facts)
    except Exception:
        logger.exception("Embedding-based matching failed. Using keyword fallback.")
        return check_claim_against_facts_keyword(claim, facts)


_STOPWORDS = {
    # Articles, pronouns, prepositions
    "the",
    "a",
    "an",
    "in",
    "on",
    "at",
    "for",
    "to",
    "of",
    "with",
    "by",
    "from",
    "about",
    "as",
    "into",
    "like",
    "through",
    "after",
    "before",
    "between",
    "out",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    # Conjunctions
    "and",
    "but",
    "or",
    "so",
    "nor",
    "yet",
    "if",
    "because",
    "while",
    # Verbs (auxiliary)
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "should",
    "can",
    "could",
    "may",
    "might",
    # Common adverbs/adjectives
    "not",
    "very",
    "just",
    "also",
}


# -----------------------------------------------------------------------------
# Grounding Check (LLM-based)
# -----------------------------------------------------------------------------


def check_grounding_llm(
    answer_candidate: str,
    facts: list[str],
    claims: list[str] | None = None,
) -> GroundingCheck:
    """
    Use LLM to verify if answer is grounded in facts.

    This is the most accurate method but requires LLM call.
    """
    try:
        from cortex.llm.client import complete_json
        from cortex.prompts import (
            SYSTEM_GROUNDING_CHECK,
            USER_GROUNDING_CHECK,
            construct_prompt_messages,
        )
        from cortex.security.defenses import sanitize_user_input

        facts_text = "\n".join(f"- {sanitize_user_input(fact)}" for fact in facts)

        claims_to_check = claims or []
        claims_text = (
            "\n".join(f"- {sanitize_user_input(claim)}" for claim in claims_to_check)
            if claims_to_check
            else "(no verifiable claims detected)"
        )

        messages = construct_prompt_messages(
            system_prompt_template=SYSTEM_GROUNDING_CHECK,
            user_prompt_template=USER_GROUNDING_CHECK,
            answer=sanitize_user_input(answer_candidate),
            facts=facts_text,
            claims=claims_text,
        )

        if len(messages) < 2:
            raise ValueError("Prompt construction returned insufficient messages")

        # Reconstruct prompt for the deprecated `complete_json`
        system_content = messages[0].get("content")
        user_content = messages[1].get("content")
        if system_content is None or user_content is None:
            raise ValueError("Prompt messages missing content")
        reconstructed_prompt = f"{system_content}\n\n{user_content}"

        result = complete_json(
            prompt=reconstructed_prompt,
            schema=GroundingAnalysisResult.model_json_schema(),
        )

        analysis = GroundingAnalysisResult(**result)

        claim_analyses_with_method = []
        for claim in analysis.claims:
            data = claim.model_dump()
            if not data.get("method"):
                data["method"] = "llm"
            claim_analyses_with_method.append(ClaimAnalysis(**data))

        # Calculate grounding ratio
        total_claims = len(claim_analyses_with_method)
        supported_claims = sum(1 for c in claim_analyses_with_method if c.is_supported)
        grounding_ratio = supported_claims / total_claims if total_claims > 0 else 0.0

        return GroundingCheck(
            answer_candidate=answer_candidate,
            is_grounded=analysis.overall_grounded,
            confidence=analysis.overall_confidence,
            unsupported_claims=analysis.unsupported_claims,
            claim_analyses=claim_analyses_with_method,
            grounding_ratio=grounding_ratio,
            method="llm",
        )
    except ImportError as e:
        logger.error(
            f"LLM dependencies not found for grounding check: {e}. Install with `pip install cortex-llm`."
        )
        return check_grounding_embedding(answer_candidate, facts)
    except Exception:
        logger.exception(
            "LLM grounding check failed due to an API error. Using embedding fallback."
        )
        return check_grounding_embedding(answer_candidate, facts)


def check_grounding_embedding(
    answer_candidate: str,
    facts: list[str],
    support_threshold: float = DEFAULT_EMBEDDING_SUPPORT_THRESHOLD,
    grounding_threshold: float = DEFAULT_EMBEDDING_GROUNDING_THRESHOLD,
) -> GroundingCheck:
    """
    Verify grounding using embedding similarity.

    Extracts claims, checks each against facts using embeddings.

    Args:
        answer_candidate: The answer to verify
        facts: List of facts from retrieved context
        support_threshold: Min similarity for a claim to be considered supported
        grounding_threshold: Min ratio of supported claims for answer to be grounded
    """
    # Extract claims from answer
    claims = extract_claims_simple(answer_candidate)

    if not claims:
        # No verifiable claims = not grounded.
        # This is a safety measure. An answer with no factual claims to verify
        # cannot be considered "grounded".
        return GroundingCheck(
            answer_candidate=answer_candidate,
            is_grounded=False,
            confidence=1.0,
            unsupported_claims=[],
            claim_analyses=[],
            grounding_ratio=0.0,
            method="embedding",
        )

    if not facts:
        # No facts to check against = not grounded
        return GroundingCheck(
            answer_candidate=answer_candidate,
            is_grounded=False,
            confidence=1.0,  # High confidence that it's not grounded
            unsupported_claims=claims,
            claim_analyses=[
                ClaimAnalysis(
                    claim=claim,
                    is_supported=False,
                    supporting_fact=None,
                    confidence=0.0,
                    method="embedding",
                )
                for claim in claims
            ],
            grounding_ratio=0.0,
            method="embedding",
        )

    # Check each claim
    claim_analyses = []
    unsupported = []

    # Pre-compute fact embeddings once if facts exist
    fact_embeddings = None
    if facts:
        try:
            client = get_embeddings_client()
            fact_embeddings = client.embed_texts(facts)
        except ImportError as e:
            logger.error(
                f"Embeddings client dependencies not found: {e}. Keyword fallback will be used for claims."
            )
        except Exception as e:
            logger.warning(
                f"Failed to pre-compute fact embeddings: {e}. Keyword fallback will be used for claims."
            )

    for claim in claims:
        is_supported, confidence, supporting_fact, method = (
            check_claim_against_facts_embedding(
                claim,
                facts,
                threshold=support_threshold,
                fact_embeddings=fact_embeddings,
            )
        )

        claim_analyses.append(
            ClaimAnalysis(
                claim=claim,
                is_supported=is_supported,
                supporting_fact=supporting_fact,
                confidence=confidence,
                method=method,
            )
        )

        if not is_supported:
            unsupported.append(claim)

    # Calculate overall grounding
    supported_count = sum(1 for c in claim_analyses if c.is_supported)
    grounding_ratio = supported_count / len(claims) if claims else 0.0

    is_grounded = grounding_ratio >= grounding_threshold
    avg_confidence = (
        sum(c.confidence for c in claim_analyses) / len(claim_analyses)
        if claim_analyses
        else 0.0
    )

    return GroundingCheck(
        answer_candidate=answer_candidate,
        is_grounded=is_grounded,
        confidence=avg_confidence,
        unsupported_claims=unsupported,
        claim_analyses=claim_analyses,
        grounding_ratio=grounding_ratio,
        method="embedding",
    )


# -----------------------------------------------------------------------------
# Main Tool Interface (Blueprint §9.4)
# -----------------------------------------------------------------------------


def tool_check_grounding(args: GroundingCheckInput) -> GroundingCheck:
    """
    Verify if an answer is grounded in the provided facts.

    This is the main tool interface per Blueprint §9.4.
    Used for high-risk use cases (e.g., compliance answers) to verify
    factual support in retrieved context.

    Args:
        answer_candidate: The generated answer to verify
        facts: List of facts from retrieved context
        use_llm: Whether to use LLM for verification (more accurate)

    Returns:
        GroundingCheck with grounding status and unsupported claims
    """
    if not args.answer_candidate or not args.answer_candidate.strip():
        return GroundingCheck(
            answer_candidate=args.answer_candidate or "",
            is_grounded=True,
            confidence=1.0,
            unsupported_claims=[],
            grounding_ratio=1.0,
            method="not_applicable",
        )

    if args.use_llm:
        claims = extract_claims_llm(args.answer_candidate)
        return check_grounding_llm(args.answer_candidate, args.facts, claims=claims)

    return check_grounding_embedding(args.answer_candidate, args.facts)


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------


def is_answer_grounded(
    answer: str,
    facts: list[str],
    threshold: float = DEFAULT_EMBEDDING_GROUNDING_THRESHOLD,
) -> bool:
    """
    Quick check if answer is grounded (returns bool only).

    Args:
        answer: The answer to check
        facts: Facts to check against
        threshold: Minimum grounding ratio

    Returns:
        True if answer is grounded, False otherwise
    """
    result = tool_check_grounding(
        GroundingCheckInput(answer_candidate=answer, facts=facts, use_llm=False)
    )
    return result.grounding_ratio >= threshold


def get_unsupported_claims(answer: str, facts: list[str]) -> list[str]:
    """
    Get list of claims in the answer that aren't supported by facts.

    Useful for highlighting potential issues to users.
    """
    result = tool_check_grounding(
        GroundingCheckInput(answer_candidate=answer, facts=facts, use_llm=False)
    )
    return result.unsupported_claims


__all__ = (
    "ClaimAnalysis",
    "GroundingAnalysisResult",
    "GroundingCheck",
    "GroundingCheckInput",
    "extract_claims_llm",
    "extract_claims_simple",
    "get_unsupported_claims",
    "is_answer_grounded",
    "tool_check_grounding",
)
