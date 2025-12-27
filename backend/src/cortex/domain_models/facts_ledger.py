"""
Facts Ledger Domain Models.

Implements §10.4 of the Canonical Blueprint - summarization models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, List, Literal, Optional

from pydantic import BaseModel, Field


class Ask(BaseModel):
    """An ask/request identified in the thread."""

    description: str
    from_participant: Optional[str] = None
    to_participant: Optional[str] = None
    status: Literal["open", "addressed", "declined"] = "open"


class Commitment(BaseModel):
    """A commitment made by a participant."""

    description: str
    by_participant: Optional[str] = None
    due_date: Optional[datetime] = None
    status: Literal["pending", "fulfilled", "missed"] = "pending"


class KeyDate(BaseModel):
    """A key date mentioned in the thread."""

    date: Optional[datetime] = None
    description: str = ""
    relevance: str = ""


class ParticipantAnalysis(BaseModel):
    """Analyst's view of a participant."""

    name: Optional[str] = None
    role: Optional[Literal["client", "broker", "underwriter", "internal", "other"]] = (
        "other"
    )
    tone: Optional[
        Literal[
            "professional", "frustrated", "urgent", "friendly", "demanding", "neutral"
        ]
    ] = "neutral"
    stance: Optional[str] = None
    email: Optional[str] = None

    def __repr__(self) -> str:
        """Redacts PII."""
        return f"ParticipantAnalysis(name='[REDACTED]', role='{self.role}', email='[REDACTED]')"


class FactsLedger(BaseModel):
    """Extracted facts from a thread - analyst output."""

    asks: List[Ask] = Field(default_factory=list)
    commitments: List[Commitment] = Field(default_factory=list)
    key_dates: List[KeyDate] = Field(default_factory=list)
    key_decisions: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    risks_concerns: List[str] = Field(default_factory=list)
    participants: List[ParticipantAnalysis] = Field(default_factory=list)

    def merge(self, other: "FactsLedger") -> "FactsLedger":
        """
        Merge another ledger into this one with improved performance and correctness.
        - Lists are unioned and deduplicated efficiently.
        - Participants are merged robustly, prioritizing non-default values.
        """
        if not other:
            return self

        # Helper for deduplicating lists of Pydantic models based on a key
        def _deduplicate_models(
            items: List[Any], key_fn: Callable[[Any], Any]
        ) -> List[Any]:
            seen = set()
            out = []
            for item in items:
                key = key_fn(item)
                if key not in seen:
                    seen.add(key)
                    out.append(item)
            return out

        # 1. Merge lists of Pydantic models efficiently
        asks = _deduplicate_models(
            self.asks + other.asks,
            lambda x: (x.description, x.from_participant, x.to_participant),
        )
        commitments = _deduplicate_models(
            self.commitments + other.commitments,
            lambda x: (x.description, x.by_participant),
        )
        key_dates = _deduplicate_models(
            self.key_dates + other.key_dates, lambda x: (x.date, x.description)
        )

        # 2. Merge simple string lists using sets for performance and deterministic order
        key_decisions = sorted(list(set(self.key_decisions + other.key_decisions)))
        open_questions = sorted(list(set(self.open_questions + other.open_questions)))
        risks_concerns = sorted(list(set(self.risks_concerns + other.risks_concerns)))

        # 3. Merge participants with a robust, deterministic strategy
        def get_participant_key(p: ParticipantAnalysis) -> Optional[str]:
            """Generates a unique key for a participant, prioritizing email."""
            if p.email:
                return f"email:{p.email.strip().lower()}"
            if p.name:
                return f"name:{p.name.strip().lower()}"
            return None

        # Start with the participants from this ledger
        participant_map: dict[str, ParticipantAnalysis] = {
            key: p.model_copy()
            for p in self.participants
            if (key := get_participant_key(p))
        }

        # Merge in participants from the other ledger
        for other_p in other.participants:
            key = get_participant_key(other_p)
            if not key:
                continue

            if key not in participant_map:
                participant_map[key] = other_p.model_copy()
            else:
                # Merge `other_p` into the existing record (`p`)
                p = participant_map[key]
                p.name = p.name or other_p.name
                p.email = p.email or other_p.email
                if p.role == "other":
                    p.role = other_p.role
                if p.tone == "neutral":
                    p.tone = other_p.tone
                p.stance = p.stance or other_p.stance

        return FactsLedger(
            asks=asks,
            commitments=commitments,
            key_dates=key_dates,
            key_decisions=key_decisions,
            open_questions=open_questions,
            risks_concerns=risks_concerns,
            participants=list(participant_map.values()),
        )


class CriticReview(BaseModel):
    """Critic's review of the facts ledger."""

    completeness_score: float = 0.0
    accuracy_concerns: List[str] = Field(default_factory=list)
    missing_items: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    overall_assessment: str = ""
