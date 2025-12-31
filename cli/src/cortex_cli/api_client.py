from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Any

import httpx
from cortex.common.exceptions import CortexError

JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None

_API_CLIENT: ApiClient | None = None


def get_default_token_path() -> Path:
    config_root = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_root / "cortex_cli" / "token"


def _load_token_from_file(token_path: Path) -> str | None:
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


class ApiClient:
    """A client for interacting with the Cortex API."""

    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        env_url = os.getenv("CORTEX_API_URL", "https://localhost:8000/api/v1")
        self.base_url: str = base_url if base_url is not None else env_url
        self.token = token or os.getenv("CORTEX_API_TOKEN")
        if self.token is None:
            self.token = _load_token_from_file(get_default_token_path())
        self._closed = False

        url = httpx.URL(self.base_url)
        if self.token and url.scheme != "https":
            if url.host not in {"localhost", "127.0.0.1", "::1"}:
                raise CortexError("Refusing to send credentials over insecure HTTP.")

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        self.client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=60.0
        )

    def close(self) -> None:
        if self._closed:
            return
        self.client.close()
        self._closed = True

    def post(self, endpoint: str, data: dict[str, Any]) -> JsonValue:
        """Make a POST request to the API."""
        if endpoint.startswith("/") and "://" not in endpoint:
            endpoint = endpoint.lstrip("/")
        try:
            response = self.client.post(endpoint, json=data)
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError as exc:
                if not response.content:
                    return {}
                raise CortexError("API response was not valid JSON.") from exc
            return payload
        except httpx.HTTPStatusError as e:
            raise CortexError(f"API request failed: {e.response.text}") from e
        except httpx.RequestError as e:
            raise CortexError(f"API request failed: {e}") from e

    def answer(self, query: str, tenant_id: str, user_id: str) -> JsonValue:
        """Get an answer from the RAG API."""
        return self.post(
            "answer", data={"query": query, "tenant_id": tenant_id, "user_id": user_id}
        )


def get_api_client() -> ApiClient:
    """Get a singleton instance of the API client."""
    global _API_CLIENT
    if _API_CLIENT is None:
        _API_CLIENT = ApiClient()
        atexit.register(_close_api_client)
    return _API_CLIENT


def _close_api_client() -> None:
    global _API_CLIENT
    if _API_CLIENT is None:
        return
    _API_CLIENT.close()
    _API_CLIENT = None
