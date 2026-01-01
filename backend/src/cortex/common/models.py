"""
Common data models for the Cortex API.
"""

from __future__ import annotations

import json
from collections.abc import Set as AbstractSet
from typing import Any, ClassVar

from pydantic import BaseModel


class SecureBaseModel(BaseModel):
    """
    A Pydantic BaseModel that redacts fields containing Personally Identifiable Information (PII)
    in its string representation to prevent accidental logging of sensitive data.

    Child classes must define a `_PII_FIELDS` set-like collection containing the
    names of the fields to be redacted.
    """

    _PII_FIELDS: ClassVar[AbstractSet[str]] = frozenset()

    def __repr__(self) -> str:
        redacted = self.redacted_dump()
        return f"{self.__class__.__name__}({redacted})"

    def __str__(self) -> str:
        return self.__repr__()

    def redacted_dump(self, **kwargs: Any) -> dict[str, Any]:
        by_alias = bool(kwargs.get("by_alias"))
        data = self.model_dump(**kwargs)
        if not isinstance(data, dict) or not self._PII_FIELDS:
            return data if isinstance(data, dict) else {}

        pii_keys = self._resolve_pii_keys(by_alias=by_alias)
        return {
            key: "*****" if key in pii_keys else value for key, value in data.items()
        }

    def redacted_json(self, **kwargs: Any) -> str:
        data = self.redacted_dump(mode="json", **kwargs)
        return json.dumps(data, default=str)

    def _resolve_pii_keys(self, *, by_alias: bool) -> set[str]:
        if not self._PII_FIELDS:
            return set()

        pii_keys: set[str] = set()
        for field_name, field_info in type(self).model_fields.items():
            alias = field_info.alias
            is_pii = field_name in self._PII_FIELDS or (
                alias is not None and alias in self._PII_FIELDS
            )
            if not is_pii:
                continue
            if by_alias and alias:
                pii_keys.add(alias)
            else:
                pii_keys.add(field_name)
        return pii_keys
