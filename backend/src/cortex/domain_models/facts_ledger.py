"""
Facts Ledger Domain Models.

Implements §10.4 of the Canonical Blueprint - summarization models.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

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


class FactsLedger(BaseModel):
    """Extracted facts from a thread - analyst output."""

    asks: List[Ask] = Field(default_factory=list)
    commitments: List[Commitment] = Field(default_factory=list)
    key_dates: List[KeyDate] = Field(default_factory=list)
    key_decisions: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    risks_concerns: List[str] = Field(default_factory=list)


class CriticReview(BaseModel):
    """Critic's review of the facts ledger."""

    completeness_score: float = 0.0
    accuracy_concerns: List[str] = Field(default_factory=list)
    missing_items: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    overall_assessment: str = ""
