"""
Async LLM Runtime.

- Uses `httpx.AsyncClient` for non-blocking HTTP requests.
- Structured for FastAPI integration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx
from cortex.llm.runtime import (
    ConfigurationError,
    ProviderError,
    _config,
    _try_load_json,
)

logger = logging.getLogger(__name__)


class AsyncLLMRuntime:
    """
    Asynchronous orchestrator for LLM provider interactions.
    """

    _client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-load the httpx.AsyncClient."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def complete_text(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Asynchronously completes text using an OpenAI-compatible API.
        """
        base_url = getattr(_config, "llm_base_url", None)
        api_key = getattr(_config, "llm_api_key", "EMPTY")
        model_name = kwargs.get("model", getattr(_config, "llm_model", "openai-gpt-oss-120b"))

        if not base_url:
            raise ConfigurationError("LLM base URL not configured.")

        headers = {"Authorization": f"Bearer {api_key}"}

        req_body = {
            "model": model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 50),
        }

        try:
            resp = await self.client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=req_body,
            )
            resp.raise_for_status()

            json_response = resp.json()

            if not json_response.get("choices"):
                raise ProviderError("No choices in completion response")

            message = json_response["choices"][0].get("message", {})
            content = message.get("content", "")

            return content if isinstance(content, str) else str(content)

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during LLM call: {e.response.text}")
            raise ProviderError(f"LLM API request failed with status {e.response.status_code}") from e
        except Exception as e:
            logger.error(f"Error during async LLM call: {e}")
            raise ProviderError(f"LLM call failed: {e}") from e


# Initialized at module load time to prevent race conditions in async environments.
_async_runtime_instance = AsyncLLMRuntime()


def get_async_runtime() -> AsyncLLMRuntime:
    """Returns the singleton instance of the AsyncLLMRuntime."""
    return _async_runtime_instance
