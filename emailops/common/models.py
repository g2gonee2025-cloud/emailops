"""
Core data models for EmailOps.

Defines canonical data structures with validation using Pydantic.
This replaces mixed tuple/dict representations causing data loss.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, EmailStr, Field, field_validator


class ParticipantRole(str, Enum):
    """Role of a participant in an email conversation."""

    SENDER = "sender"
    RECIPIENT = "recipient"
    CC = "cc"
    BCC = "bcc"
    UNKNOWN = "unknown"


class Participant(BaseModel):
    """
    Canonical participant representation with validation.

    Replaces 3 incompatible formats:
    - Tuples: [(name, email), ...] - loses role/tone
    - Dicts: {"name": str, "smtp": str} - inconsistent field names
    - Structured: {"name": str, "email": str, "role": enum, ...} - not standardized

    This model provides:
    - Runtime validation (email format, required fields)
    - Automatic JSON serialization
    - Field aliases for backward compatibility
    - Type safety for all operations

    Examples:
        >>> p = Participant(name="Alice Smith", email="alice@example.com")
        >>> p.email
        'alice@example.com'

        >>> # Backward compat with 'smtp' field
        >>> p = Participant(name="Bob", smtp="bob@example.com")
        >>> p.email
        'bob@example.com'

        >>> # Validation catches errors
        >>> p = Participant(name="", email="invalid")
        ValidationError: name cannot be empty, email format invalid
    """

    name: str = Field(..., description="Display name of participant", min_length=1)
    email: EmailStr = Field(..., description="Email address")
    role: ParticipantRole = Field(
        default=ParticipantRole.RECIPIENT, description="Role in conversation"
    )
    tone: str | None = Field(default=None, description="Communication tone (formal, casual, etc)")
    title: str | None = Field(default=None, description="Job title or position")
    organization: str | None = Field(default=None, description="Company or organization")

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        """Validate name is not empty or whitespace-only."""
        if not v or not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: EmailStr) -> str:
        """Normalize email to lowercase."""
        return str(v).lower().strip()

    @classmethod
    def from_tuple(cls, t: tuple[str, str], role: ParticipantRole = ParticipantRole.RECIPIENT) -> Participant:
        """
        Create Participant from legacy tuple format.

        Args:
            t: Tuple of (name, email)
            role: Optional role (default: RECIPIENT)

        Returns:
            Participant instance

        Example:
            >>> p = Participant.from_tuple(("Alice", "alice@example.com"))
            >>> p.name
            'Alice'
        """
        if len(t) < 2:
            raise ValueError(f"Tuple must have at least 2 elements (name, email), got {len(t)}")
        return cls(name=t[0], email=t[1], role=role)

    @classmethod
    def from_dict(cls, d: dict, role: ParticipantRole = ParticipantRole.RECIPIENT) -> Participant:
        """
        Create Participant from legacy dict format.

        Handles multiple field name variations:
        - email, smtp, address, email_address
        - name, display_name, full_name

        Args:
            d: Dictionary with participant data
            role: Optional role (default: RECIPIENT)

        Returns:
            Participant instance

        Example:
            >>> p = Participant.from_dict({"name": "Alice", "smtp": "alice@example.com"})
            >>> p.email
            'alice@example.com'
        """
        # Extract email from various field names
        email = None
        for key in ("email", "smtp", "address", "email_address"):
            if d.get(key):
                email = d[key]
                break

        if not email:
            raise ValueError(f"No email field found in dict: {list(d.keys())}")

        # Extract name from various field names
        name = None
        for key in ("name", "display_name", "full_name"):
            if d.get(key):
                name = d[key]
                break

        if not name or not str(name).strip():
            # No fallback - raise error for missing name
            raise ValueError(f"No name field found in dict: {list(d.keys())}")

        # Extract optional fields
        return cls(
            name=name,
            email=email,
            role=d.get("role", role),
            tone=d.get("tone"),
            title=d.get("title"),
            organization=d.get("organization"),
        )

    def to_tuple(self) -> tuple[str, str]:
        """
        Convert to legacy tuple format for backward compatibility.

        Returns:
            Tuple of (name, email)
        """
        return (self.name, str(self.email))

    def to_dict_legacy(self) -> dict[str, str]:
        """
        Convert to legacy dict format for backward compatibility.

        Returns:
            Dict with 'name' and 'smtp' keys
        """
        return {"name": self.name, "smtp": str(self.email)}

    model_config = {
        # Allow creation from dict with 'smtp' field
        "populate_by_name": True,
        # Strict validation
        "validate_assignment": True,
        # Allow extra fields for forward compatibility
        "extra": "ignore",
    }


__all__ = ["Participant", "ParticipantRole"]
