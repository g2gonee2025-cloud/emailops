"""
GGUF Embedding Provider via llama-server HTTP API.

Provides CPU-based embedding by calling a llama-server instance running
the quantized GGUF model. This enables real-time query embedding without
requiring a GPU cluster.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import requests

logger = logging.getLogger(__name__)


class GGUFProvider:
    """
    CPU-based embedding provider using llama-server HTTP API.

    Calls a locally running llama-server instance that hosts the
    KaLM-Embedding-Gemma3-12B-2511 GGUF model.
    """

    _instance: GGUFProvider | None = None

    def __new__(cls, endpoint: str | None = None) -> GGUFProvider:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, endpoint: str | None = None) -> None:
        """
        Initialize the GGUF provider.

        Args:
            endpoint: llama-server endpoint URL (default: http://127.0.0.1:8090)
        """
        if getattr(self, "_initialized", False):
            return

        self._endpoint = endpoint or self._resolve_endpoint()
        self._expected_dim = 3840  # KaLM-12B dimension
        self._timeout = 180  # seconds - increased for CPU inference
        self._initialized = True

        logger.info("GGUFProvider initialized (endpoint: %s)", self._endpoint)

    def _resolve_endpoint(self) -> str:
        """Resolve llama-server endpoint from config or environment."""
        # Try config first
        try:
            from cortex.config.loader import get_config

            config = get_config()
            endpoint = getattr(config.embedding, "llama_server_endpoint", None)
            if endpoint:
                return endpoint
        except Exception:
            pass

        # Fall back to environment variable
        return os.getenv(
            "OUTLOOKCORTEX_LLAMA_SERVER_ENDPOINT",
            os.getenv("LLAMA_SERVER_ENDPOINT", "http://127.0.0.1:8090"),
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for texts via llama-server.

        Args:
            texts: List of texts to embed

        Returns:
            numpy array of embeddings with shape (len(texts), dim)
        """
        if not texts:
            return np.empty((0, self._expected_dim), dtype=np.float32)

        embeddings = []
        url = f"{self._endpoint.rstrip('/')}/embedding"

        for text in texts:
            if not text or not text.strip():
                embeddings.append(np.zeros(self._expected_dim, dtype=np.float32))
                continue

            # Truncate long texts (KaLM recommends 512 tokens max)
            max_chars = 512 * 4
            if len(text) > max_chars:
                text = text[:max_chars]

            try:
                resp = requests.post(
                    url,
                    json={"content": text},
                    headers={"Content-Type": "application/json"},
                    timeout=self._timeout,
                )
                resp.raise_for_status()

                data = resp.json()
                # Response format: [{"index": 0, "embedding": [[...vector...]]}]
                if isinstance(data, list) and len(data) > 0:
                    emb = data[0].get("embedding", [[]])[0]
                else:
                    emb = data.get("embedding", [])

                emb_array = np.array(emb, dtype=np.float32)

                # L2 normalize
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    emb_array = emb_array / norm

                embeddings.append(emb_array)

            except requests.exceptions.RequestException as e:
                logger.error("llama-server embedding request failed: %s", e)
                embeddings.append(np.zeros(self._expected_dim, dtype=np.float32))
            except Exception as e:
                logger.error("GGUF embedding failed: %s", e)
                embeddings.append(np.zeros(self._expected_dim, dtype=np.float32))

        result = np.stack(embeddings) if embeddings else np.array([], dtype=np.float32)

        # Validate dimension
        if result.size > 0 and result.shape[1] != self._expected_dim:
            logger.warning(
                "GGUF embedding dimension mismatch: expected %d, got %d",
                self._expected_dim,
                result.shape[1],
            )

        return result

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Alias for embed() - queries use same encoding as documents for KaLM."""
        return self.embed(queries)

    def is_available(self) -> bool:
        """Check if the llama-server is available."""
        try:
            resp = requests.get(
                f"{self._endpoint.rstrip('/')}/health",
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None
