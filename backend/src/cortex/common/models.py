"""
Common data models for the Cortex API.
"""
from pydantic import BaseModel


class SecureBaseModel(BaseModel):
    """
    A Pydantic BaseModel that redacts fields containing Personally Identifiable Information (PII)
    in its string representation to prevent accidental logging of sensitive data.

    Child classes must define a `_PII_FIELDS` tuple or set containing the names of
    the fields to be redacted.
    """

    _PII_FIELDS: set[str] = set()

    def __repr_args__(self):
        args = super().__repr_args__()
        if not self._PII_FIELDS:
            return args

        return [
            (key, "*****" if key in self._PII_FIELDS else value)
            for key, value in args
        ]
