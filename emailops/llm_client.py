# emailops/llm_client.py
from __future__ import annotations
# Back-compat shim exposing the new unified API
from .llm_runtime import (
    complete_text,
    complete_json,
    embed_texts,
    LLMError,
)

__all__ = ["complete_text", "complete_json", "embed_texts", "LLMError"]
