from __future__ import annotations

import os
from typing import Any

import httpx
from cortex.common.exceptions import CortexError


class ApiClient:
    """A client for interacting with the Cortex API."""

    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        env_url = os.getenv("CORTEX_API_URL", "http://localhost:8000/api/v1")
        self.base_url: str = base_url if base_url is not None else env_url
        self.token = token or os.getenv("CORTEX_API_TOKEN")

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        self.client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=60.0
        )

    def post(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make a POST request to the API."""
        try:
            response = self.client.post(endpoint, json=data)
            response.raise_for_status()
            return response.json()  # type: ignore
        except httpx.HTTPStatusError as e:
            raise CortexError(f"API request failed: {e.response.text}") from e
        except httpx.RequestError as e:
            raise CortexError(f"API request failed: {e}") from e

    def answer(self, query: str, tenant_id: str, user_id: str) -> dict[str, Any]:
        """Get an answer from the RAG API."""
        return self.post(
            "/answer", data={"query": query, "tenant_id": tenant_id, "user_id": user_id}
        )


def get_api_client() -> ApiClient:
    """Get a singleton instance of the API client."""
    return ApiClient()
