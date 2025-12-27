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
        Merge another ledger into this one.
        - Lists are unioned and deduplicated by value.
        - Participants are merged by email/name.
        """
        if not other:
            return self

        # 1. Merge simple lists (set-like union)
        # Helper to deduplicate potentially unhashable Pydantic models by comparing their normalized representation (e.g., dict), rather than relying on hashing or JSON serialization.
        def _merge_lists(
            l1: list[Any], l2: list[Any], key_fn: Callable[[Any], Any] | None = None
        ):
            seen = set()
            out = []
            for item in l1 + l2:
                # Use key_fn if provided, otherwise invalid JSON for complex objects
                k = key_fn(item) if key_fn else item
                if isinstance(k, BaseModel):
                    k = k.model_dump_json()
                elif isinstance(k, (dict, list)):
                    # Fallback for unhashable: stringify
                    k = str(k)

                if k not in seen:
                    seen.add(k)
                    out.append(item)
            return out

        asks = _merge_lists(
            self.asks,
            other.asks,
            lambda x: (x.description, x.from_participant, x.to_participant),
        )
        commitments = _merge_lists(
            self.commitments,
            other.commitments,
            lambda x: (x.description, x.by_participant),
        )
        key_dates = _merge_lists(
            self.key_dates, other.key_dates, lambda x: (x.date, x.description)
        )
        key_decisions = _merge_lists(self.key_decisions, other.key_decisions)
        open_questions = _merge_lists(self.open_questions, other.open_questions)
        risks_concerns = _merge_lists(self.risks_concerns, other.risks_concerns)

        # 2. Merge participants (Keyed by Email or Name)
        # Logic: Existing participant data is preserved, 'other' fills gaps.
        # But 'other' might be 'newer/better', so we actually want to UNION info.
        # For simplicity in this port: we index by email (if present) or name.
        p_map = {}

        def _get_key(p: ParticipantAnalysis):
            if p.email:
                return f"email:{p.email.lower()}"
            if p.name:
                return f"name:{p.name.lower()}"
            return None

        # Process ALL participants (self + other)
        # Later entries override earlier ones for scalar fields if they are 'more complete'?
        # Actually, let's treat 'other' as 'new info' that refines 'self'.

        all_participants = self.participants + other.participants

        merged_participants = []
        for p in all_participants:
            k = _get_key(p)
            if not k:
                continue

            if k not in p_map:
                p_map[k] = p.model_copy()
                merged_participants.append(p_map[k])
            else:
                existing = p_map[k]
                # Update scalars if new one is present/better.
                # e.g. if existing role is "other" and new is "broker", take "broker".
                if p.name and not existing.name:
                    existing.name = p.name
                if p.email and not existing.email:
                    existing.email = p.email
                if p.role != "other" and existing.role == "other":
                    existing.role = p.role
                if p.tone != "neutral" and existing.tone == "neutral":
                    existing.tone = p.tone
                if p.stance and not existing.stance:
                    existing.stance = p.stance

        return FactsLedger(
            asks=asks,
            commitments=commitments,
            key_dates=key_dates,
            key_decisions=key_decisions,
            open_questions=open_questions,
            risks_concerns=risks_concerns,
            participants=merged_participants,
        )


class CriticReview(BaseModel):
    """Critic's review of the facts ledger."""

    completeness_score: float = 0.0
    accuracy_concerns: list[str] = Field(default_factory=list)
    missing_items: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    overall_assessment: str = ""
