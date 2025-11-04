from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

log = logging.getLogger(__name__)


class ExportState:
    """Delta sync state stored in _state.json under output root."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                payload = self.path.read_text(encoding="utf-8")
                self.data = json.loads(payload)
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("Failed to load state from %s: %s", self.path, exc)
                self.data = {}
        else:
            self.data = {}

    def save(self) -> None:
        """Save state to disk. Raises exception if save fails to prevent silent data loss."""
        try:
            self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            log.error("Failed to save state to %s: %s", self.path, exc)
            raise RuntimeError(f"State persistence failed: {exc}") from exc

    @property
    def last_sync_utc(self) -> datetime | None:
        iso = self.data.get("LastSyncUTC")
        if not iso:
            return None
        try:
            if iso.endswith("Z"):
                iso = iso[:-1]
            dt = datetime.fromisoformat(iso)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            return None

    @last_sync_utc.setter
    def last_sync_utc(self, dt: datetime) -> None:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        self.data["LastSyncUTC"] = dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def set_folders(self, folders: list[str]) -> None:
        self.data["Folders"] = list(folders)

    def mark_conversation_exported(self, conv_key: str, dir_name: str) -> None:
        convs = cast(dict[str, str], self.data.setdefault("Conversations", {}))
        convs[conv_key] = dir_name

    def get_conversation_dir(self, conv_key: str) -> str | None:
        conversations = self.data.get("Conversations")
        if isinstance(conversations, dict):
            mapped = cast(dict[str, str], conversations)
            return mapped.get(conv_key)
        return None
