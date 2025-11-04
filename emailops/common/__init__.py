"""
Common types and models shared across EmailOps modules.

This package provides foundational types for error handling and data modeling.

Quick Start:
    >>> from emailops.common import Result, Participant
    >>> # Type-safe validation
    >>> result = Result.success(42)
    >>> value = result.unwrap()
    >>> # Data validation
    >>> p = Participant(name="Alice", email="alice@example.com")
"""

from .models import Participant, ParticipantRole
from .types import Result, collect_results, sequence_results, traverse_results

__all__ = [
    "Participant",
    "ParticipantRole",
    "Result",
    "collect_results",
    "sequence_results",
    "traverse_results",
]
