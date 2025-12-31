"""
Facts Ledger Domain Models.

Implements §10.4 of the Canonical Blueprint - summarization models.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Ask(BaseModel):
    """An ask/request identified in the thread."""

    description: str
    from_participant: str | None = None
    to_participant: str | None = None
    status: Literal["open", "addressed", "declined"] = "open"


class Commitment(BaseModel):
    """A commitment made by a participant."""

    description: str
    by_participant: str | None = None
    due_date: datetime | None = None
    status: Literal["pending", "fulfilled", "missed"] = "pending"


class KeyDate(BaseModel):
    """A key date mentioned in the thread."""

    date: datetime | None = None
    description: str = ""
    relevance: str = ""


class ParticipantAnalysis(BaseModel):
    """Analyst's view of a participant."""

    name: str | None = None
    role: Literal["client", "broker", "underwriter", "internal", "other"] | None = (
        "other"
    )
    tone: (
        Literal[
            "professional", "frustrated", "urgent", "friendly", "demanding", "neutral"
        ]
        | None
    ) = "neutral"
    stance: str | None = None
    email: str | None = None

    def __repr__(self) -> str:
        """Redacts PII."""
        return f"ParticipantAnalysis(name='[REDACTED]', role='{self.role}', email='[REDACTED]')"


class FactsLedger(BaseModel):
    """Extracted facts from a thread - analyst output."""

    asks: list[Ask] = Field(default_factory=list)
    commitments: list[Commitment] = Field(default_factory=list)
    key_dates: list[KeyDate] = Field(default_factory=list)
    key_decisions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    risks_concerns: list[str] = Field(default_factory=list)
    participants: list[ParticipantAnalysis] = Field(default_factory=list)

    def merge(self, other: FactsLedger) -> FactsLedger:
        """
        Merge another ledger into this one with improved performance and correctness.
        - Lists are unioned and deduplicated efficiently.
        - Participants are merged robustly, prioritizing non-default values.
        """
        if other is None or _is_empty_ledger(other):
            return self

        # Helper for deduplicating lists of Pydantic models based on a key
        def _deduplicate_models(
            items: list[Any], key_fn: Callable[[Any], Any]
        ) -> list[Any]:
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
            lambda x: (x.description, x.from_participant, x.to_participant, x.status),
        )
        commitments = _deduplicate_models(
            self.commitments + other.commitments,
            lambda x: (x.description, x.by_participant, x.due_date, x.status),
        )
        key_dates = _deduplicate_models(
            self.key_dates + other.key_dates,
            lambda x: (x.date, x.description, x.relevance),
        )

        # 2. Merge simple string lists using sets for performance and deterministic order
        key_decisions = sorted(set(self.key_decisions + other.key_decisions))
        open_questions = sorted(set(self.open_questions + other.open_questions))
        risks_concerns = sorted(set(self.risks_concerns + other.risks_concerns))

        # 3. Merge participants with a robust, deterministic strategy
        def get_participant_key(p: ParticipantAnalysis) -> str | None:
            """Generates a unique key for a participant, prioritizing email."""
            email = (p.email or "").strip().lower()
            if email:
                return f"email:{email}"
            name = (p.name or "").strip().lower()
            if name:
                return f"name:{name}"
            return None

        # Start with the participants from this ledger
        participant_map: dict[str, ParticipantAnalysis] = {}
        unkeyed_participants: list[ParticipantAnalysis] = []
        for participant in self.participants:
            key = get_participant_key(participant)
            if key:
                participant_map[key] = participant.model_copy()
            else:
                unkeyed_participants.append(participant.model_copy())

        # Merge in participants from the other ledger
        for other_p in other.participants:
            key = get_participant_key(other_p)
            if not key:
                unkeyed_participants.append(other_p.model_copy())
                continue

            if key not in participant_map:
                participant_map[key] = other_p.model_copy()
            else:
                # Merge `other_p` into the existing record (`p`)
                p = participant_map[key]
                p.name = p.name or other_p.name
                p.email = p.email or other_p.email
                p.role = _merge_default_value(p.role, other_p.role, "other")
                p.tone = _merge_default_value(p.tone, other_p.tone, "neutral")
                p.stance = p.stance or other_p.stance

        return FactsLedger(
            asks=asks,
            commitments=commitments,
            key_dates=key_dates,
            key_decisions=key_decisions,
            open_questions=open_questions,
            risks_concerns=risks_concerns,
            participants=list(participant_map.values()) + unkeyed_participants,
        )


def _is_empty_ledger(ledger: FactsLedger) -> bool:
    return not any(
        [
            ledger.asks,
            ledger.commitments,
            ledger.key_dates,
            ledger.key_decisions,
            ledger.open_questions,
            ledger.risks_concerns,
            ledger.participants,
        ]
    )


def _merge_default_value(
    current: str | None,
    candidate: str | None,
    default: str,
) -> str | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    if current == default:
        return candidate
    return current


class CriticReview(BaseModel):
    """Critic's review of the facts ledger."""

    completeness_score: float = 0.0
    accuracy_concerns: list[str] = Field(default_factory=list)
    missing_items: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    overall_assessment: str = ""
