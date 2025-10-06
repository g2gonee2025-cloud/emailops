from __future__ import annotations

import os
import re
import json
import logging
import time
import random
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np

from .env_utils import (
    _init_vertex,
    reset_vertex_init,
    LLMError,
    load_validated_accounts,
    DEFAULT_ACCOUNTS,
)

# Library-safe logging: no basicConfig at module level
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Project rotation state (lazy-loaded from env_utils)
# --------------------------------------------------------------------------------------
_PROJECT_ROTATION: Dict[str, Any] = {
    "projects": [],           # [{"project_id": str, "credentials_path": str}, ...]
    "current_index": 0,
    "consecutive_errors": 0,
    "_initialized": False,
}

def _ensure_projects_loaded() -> None:
    """Lazy load projects once; prefer validated accounts from env_utils."""
    if _PROJECT_ROTATION["_initialized"]:
        return
    try:
        accounts = load_validated_accounts(default_accounts=DEFAULT_ACCOUNTS)
        _PROJECT_ROTATION["projects"] = [
            {"project_id": acc.project_id, "credentials_path": acc.credentials_path}
            for acc in accounts
        ]
        logger.info("Loaded %d projects for rotation", len(_PROJECT_ROTATION["projects"]))
    except Exception as e:
        # Fall back to DEFAULT_ACCOUNTS structure without validation
        logger.warning("Failed to load validated accounts, using default list: %s", e)
        _PROJECT_ROTATION["projects"] = [
            {"project_id": acc["project_id"], "credentials_path": acc["credentials_path"]}
            for acc in DEFAULT_ACCOUNTS
        ]
    _PROJECT_ROTATION["_initialized"] = True

def _rotate_to_next_project() -> str:
    """Switch env vars to next GCP project/service account and reset Vertex init."""
    _ensure_projects_loaded()
    if not _PROJECT_ROTATION["projects"]:
        # Nothing to rotate to; keep current env.
        logger.warning("No projects available for rotation; staying on current env.")
        return os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or "<unknown>"

    current = _PROJECT_ROTATION["current_index"]
    _PROJECT_ROTATION["current_index"] = (current + 1) % len(_PROJECT_ROTATION["projects"])
    project_config = _PROJECT_ROTATION["projects"][_PROJECT_ROTATION["current_index"]]

    creds_path = project_config["credentials_path"]
    if not os.path.isabs(creds_path):
        module_dir = Path(__file__).resolve().parent.parent
        creds_path = str(module_dir / creds_path)

    os.environ["GCP_PROJECT"] = project_config["project_id"]
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_config["project_id"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

    reset_vertex_init()
    logger.warning("🔄 Rotating to project: %s", project_config["project_id"])
    return project_config["project_id"]

# --------------------------------------------------------------------------------------
# Retry classification
# --------------------------------------------------------------------------------------
try:
    from google.api_core import exceptions as gax_exceptions
except Exception:  # pragma: no cover
    gax_exceptions = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

RETRYABLE_SUBSTRINGS = (
    "quota exceeded",
    "resource_exhausted",
    "429",
    "temporarily unavailable",
    "rate limit",
    "deadline exceeded",
    "unavailable",
    "internal error",
    "503",
)

def _is_retryable_error(err: Exception) -> bool:
    if gax_exceptions:
        retry_types = tuple(
            cls for cls in (
                getattr(gax_exceptions, "ResourceExhausted", None),
                getattr(gax_exceptions, "TooManyRequests", None),
                getattr(gax_exceptions, "ServiceUnavailable", None),
                getattr(gax_exceptions, "InternalServerError", None),
                getattr(gax_exceptions, "DeadlineExceeded", None),
            ) if cls is not None
        )
        if retry_types and isinstance(err, retry_types):
            return True
    msg = str(err).lower()
    return any(fragment in msg for fragment in RETRYABLE_SUBSTRINGS)

def _should_rotate_on(err: Exception) -> bool:
    """Heuristic for when we should rotate to the next project."""
    msg = str(err).lower()
    return ("429" in msg) or ("resource_exhausted" in msg) or ("quota" in msg) or ("rate limit" in msg)

def _sleep_with_backoff(attempt: int, base: float, max_delay: float) -> None:
    sleep_for = min(max_delay, base * (2 ** (attempt - 1)))
    jitter = random.uniform(0, sleep_for * 0.2)
    total_sleep = sleep_for + jitter
    logger.warning("Backing off for %.1fs (attempt %d)", total_sleep, attempt)
    time.sleep(total_sleep)

# --------------------------------------------------------------------------------------
# Vertex model helpers
# --------------------------------------------------------------------------------------
def _vertex_model(system_instruction: Optional[str] = None):
    from vertexai.generative_models import GenerativeModel
    model_name = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")
    return GenerativeModel(model_name, system_instruction=system_instruction)

def _vertex_embed_model():
    # Try stable import first, then preview for older SDKs
    try:
        from vertexai.language_models import TextEmbeddingModel
    except Exception:
        from vertexai.preview.language_models import TextEmbeddingModel  # type: ignore
    embed_name = os.getenv("VERTEX_EMBED_MODEL", "gemini-embedding-001")
    return TextEmbeddingModel.from_pretrained(embed_name)

def _normalize_model_alias(name: Optional[str]) -> Optional[str]:
    """Tolerate common aliases/typos for Vertex embed model names."""
    if not name:
        return name
    alias_map = {
        # observed typo elsewhere: "embedded" -> "embedding"
        "gemini-embedded-001": "gemini-embedding-001",
    }
    return alias_map.get(name, name)

# --------------------------------------------------------------------------------------
# Text completion (Vertex only)
# --------------------------------------------------------------------------------------
def complete_text(
    system: str,
    user: str,
    max_output_tokens: int = 1200,
    temperature: float = 0.2,
    stop_sequences: Optional[List[str]] = None
) -> str:
    """
    Returns text completion using Vertex Gemini with retry + project rotation.
    """
    attempts = int(os.getenv("VERTEX_MAX_RETRIES", "5"))
    base_delay = float(os.getenv("VERTEX_BACKOFF_INITIAL", "4"))
    max_delay = float(os.getenv("VERTEX_BACKOFF_MAX", "60"))

    last_err: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            _init_vertex()
            model = _vertex_model(system_instruction=system)
            gen_cfg: Dict[str, Any] = dict(max_output_tokens=max_output_tokens, temperature=temperature)
            if stop_sequences:
                gen_cfg["stop_sequences"] = stop_sequences
            resp = model.generate_content(user, generation_config=gen_cfg)
            return (getattr(resp, "text", None) or "").strip()
        except Exception as e:
            last_err = e
            if (attempt == attempts) or (not _is_retryable_error(e)):
                logger.exception("Vertex generate_content failed: %s", e)
                raise LLMError(str(e)) from e
            if _should_rotate_on(e):
                _rotate_to_next_project()
            _sleep_with_backoff(attempt, base_delay, max_delay)

    # Should never reach here
    raise LLMError(str(last_err) if last_err else "Unknown error in complete_text")

def _extract_json_from_text(s: str) -> str:
    """
    Try to coerce JSON from a model string:
    1) fenced ```json blocks
    2) first {...} block
    """
    if not s:
        return "{}"
    s = s.strip()

    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s, flags=re.IGNORECASE)
    if fence:
        return fence.group(1)

    if s.startswith("{") and s.endswith("}"):
        return s

    m = re.search(r"\{[\s\S]*\}", s)
    return m.group(0) if m else "{}"

def complete_json(
    system: str,
    user: str,
    max_output_tokens: int = 1600,
    temperature: float = 0.2,
    response_schema: Optional[dict] = None,
    stop_sequences: Optional[List[str]] = None
) -> str:
    """
    Same as complete_text, but requests JSON and robustly coerces JSON on fallback.
    Retries with project rotation on quota/429.
    """
    attempts = int(os.getenv("VERTEX_MAX_RETRIES", "5"))
    base_delay = float(os.getenv("VERTEX_BACKOFF_INITIAL", "4"))
    max_delay = float(os.getenv("VERTEX_BACKOFF_MAX", "60"))

    last_err: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            _init_vertex()
            model = _vertex_model(system_instruction=system)
            gen_cfg: Dict[str, Any] = dict(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                response_mime_type="application/json",
            )
            if response_schema:
                gen_cfg["response_schema"] = response_schema
            if stop_sequences:
                gen_cfg["stop_sequences"] = stop_sequences

            resp = model.generate_content(user, generation_config=gen_cfg)
            return (getattr(resp, "text", None) or "").strip()
        except Exception as e:
            last_err = e
            # Fallback to text + regex only after we exhaust JSON‑mode retries or non‑retryable
            if (attempt == attempts) or (not _is_retryable_error(e)):
                logger.warning("JSON completion failed (%s), falling back to text mode", e)
                out = complete_text(
                    system,
                    user,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    stop_sequences=stop_sequences,
                )
                return _extract_json_from_text(out)

            if _should_rotate_on(e):
                _rotate_to_next_project()
            _sleep_with_backoff(attempt, base_delay, max_delay)

    # Final safety
    logger.warning("complete_json exhausted retries; returning empty JSON")
    return "{}"

# --------------------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------------------
def embed_texts(
    texts: List[str],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **_: Any
) -> np.ndarray:
    """
    Return array of shape (N, D) with unit‑normalized embeddings.

    Args:
        texts: list of strings to embed
        provider: one of {'vertex','openai','azure','cohere','huggingface','qwen','local'}
        model: optional provider‑specific model/deployment name override
               (e.g., Vertex embed model, OpenAI embedding model, Azure deployment, etc.)
    """
    provider = (provider or os.getenv("EMBED_PROVIDER", "vertex")).lower()

    if not texts:
        # Graceful empty input: consistent (0, 0) return avoids shape errors upstream.
        return np.zeros((0, 0), dtype="float32")

    if provider == "vertex":
        return _embed_vertex(texts, model=model)
    elif provider == "openai":
        return _embed_openai(texts, model=model)
    elif provider == "azure":
        return _embed_azure_openai(texts, model=model)
    elif provider == "cohere":
        return _embed_cohere(texts, model=model)
    elif provider == "huggingface":
        return _embed_huggingface(texts, model=model)
    elif provider == "qwen":
        return _embed_qwen(texts, model=model)
    else:  # local
        return _embed_local(texts, model=model)

def _normalize(vectors: List[List[float]]) -> np.ndarray:
    if not vectors:
        return np.zeros((0, 0), dtype="float32")
    arr = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return (arr / norms).astype("float32")

# ---- Vertex embeddings ---------------------------------------------------------
def _embed_vertex(texts: List[str], model: Optional[str] = None) -> np.ndarray:
    """
    Vertex AI embeddings with rotation on quota exhaustion.
    Uses google-genai embed_content for Gemini embedding families, or the
    legacy Vertex TextEmbeddingModel otherwise.
    """
    _init_vertex()
    _ensure_projects_loaded()

    # Normalize alias (e.g., gemini-embedded-001 -> gemini-embedding-001)
    embed_name = _normalize_model_alias(model) or os.getenv("VERTEX_EMBED_MODEL", "gemini-embedding-001")

    # Path 1: New Gemini embeddings via google-genai (recommended)
    if embed_name.startswith(("gemini-embedding", "gemini-embedder")):
        try:
            from google import genai
            from google.genai.types import EmbedContentConfig
        except Exception as e:
            raise LLMError(f"google-genai not installed: {e}")

        out_dim = os.getenv("VERTEX_EMBED_DIM")
        cfg = EmbedContentConfig(output_dimensionality=int(out_dim)) if out_dim else None
        dim = int(out_dim) if out_dim else 3072

        vectors: List[List[float]] = []
        B = int(os.getenv("EMBED_BATCH", "256"))

        for i in range(0, len(texts), B):
            chunk = texts[i:i + B]
            embedded_batch = False

            # Try once with current env + additional attempts for rotations if available
            rotation_slots = max(1, len(_PROJECT_ROTATION["projects"]) or 1)
            for _ in range(rotation_slots):
                project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
                try:
                    location = os.getenv("GCP_REGION", "global")
                    client = genai.Client(vertexai=True, project=project, location=location)
                    # Batch embed
                    resp = client.models.embed_content(model=embed_name, contents=chunk, config=cfg)

                    if resp and getattr(resp, "embeddings", None):
                        if len(resp.embeddings) == len(chunk):
                            for emb in resp.embeddings:
                                vectors.append(emb.values if getattr(emb, "values", None) else [0.0] * dim)
                            embedded_batch = True
                            _PROJECT_ROTATION["consecutive_errors"] = 0
                            break
                        else:
                            logger.warning(
                                "Embedding batch size mismatch: expected %d, got %d; rotating project",
                                len(chunk), len(resp.embeddings)
                            )
                            _rotate_to_next_project()
                            _init_vertex()
                    else:
                        logger.warning("Empty embeddings response from project %s; rotating", project)
                        _rotate_to_next_project()
                        _init_vertex()

                except Exception as e:
                    if _should_rotate_on(e) and _PROJECT_ROTATION["projects"]:
                        logger.warning("Project %s hit quota on batch; rotating", project)
                        _rotate_to_next_project()
                        _init_vertex()
                        continue
                    logger.exception("Vertex (gemini) embedding failed with non-retryable error: %s", e)
                    break  # exit rotation loop; fill zeros

            if not embedded_batch:
                logger.error("All embedding attempts failed for batch starting at index %d; appending zero vectors", i)
                vectors.extend([[0.0] * dim] * len(chunk))

        return _normalize(vectors)

    # Path 2: Legacy Vertex TextEmbeddingModel (e.g., text-embedding-005)
    vectors: List[List[float]] = []
    B = int(os.getenv("EMBED_BATCH", "256"))

    for i in range(0, len(texts), B):
        chunk = texts[i:i + B]
        # Try current env, then rotate if needed
        rotation_slots = max(1, len(_PROJECT_ROTATION["projects"]) or 1)
        success = False

        for _ in range(rotation_slots):
            try:
                from typing import cast
                from vertexai.language_models import TextEmbeddingInput
                model_obj = _vertex_embed_model()
                inputs = [TextEmbeddingInput(text=t) if isinstance(t, str) else t for t in chunk]
                embs = model_obj.get_embeddings(cast(List[Union[str, TextEmbeddingInput]], inputs))
                for e in embs:
                    vectors.append(e.values)
                success = True
                break
            except Exception as e:
                if _should_rotate_on(e) and _PROJECT_ROTATION["projects"]:
                    _rotate_to_next_project()
                    _init_vertex()
                    continue
                logger.exception("Vertex (legacy) embedding failed on batch %d:%d: %s", i, i + B, e)
                break

        if not success:
            dim = int(os.getenv("EMBED_DIM", "768"))
            vectors.extend([[0.0] * dim for _ in chunk])

    return _normalize(vectors)

# ---- OpenAI embeddings ---------------------------------------------------------
def _embed_openai(texts: List[str], model: Optional[str] = None) -> np.ndarray:
    try:
        from openai import OpenAI
    except ImportError:
        raise LLMError("Install openai: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("Set OPENAI_API_KEY environment variable")

    client = OpenAI(api_key=api_key)
    model_name = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    vectors: List[List[float]] = []
    B = int(os.getenv("EMBED_BATCH", "100"))
    for i in range(0, len(texts), B):
        chunk = texts[i:i + B]
        try:
            resp = client.embeddings.create(input=chunk, model=model_name)
            for item in resp.data:
                vectors.append(item.embedding)
        except Exception as e:
            logger.exception("OpenAI embedding failed: %s", e)
            raise LLMError(f"OpenAI embedding failed: {e}")
    return _normalize(vectors)

# ---- Azure OpenAI embeddings ---------------------------------------------------
def _embed_azure_openai(texts: List[str], model: Optional[str] = None) -> np.ndarray:
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise LLMError("Install openai: pip install openai")

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = model or os.getenv("AZURE_OPENAI_DEPLOYMENT")

    if not all([api_key, endpoint, deployment]):
        raise LLMError("Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT")

    client = AzureOpenAI(
        api_key=api_key,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_endpoint=str(endpoint),
    )

    vectors: List[List[float]] = []
    B = int(os.getenv("EMBED_BATCH", "100"))
    for i in range(0, len(texts), B):
        chunk = texts[i:i + B]
        try:
            resp = client.embeddings.create(input=chunk, model=str(deployment))
            for item in resp.data:
                vectors.append(item.embedding)
        except Exception as e:
            logger.exception("Azure OpenAI embedding failed: %s", e)
            raise LLMError(f"Azure OpenAI embedding failed: {e}")
    return _normalize(vectors)

# ---- Cohere embeddings ---------------------------------------------------------
def _embed_cohere(texts: List[str], model: Optional[str] = None) -> np.ndarray:
    try:
        import cohere
    except ImportError:
        raise LLMError("Install cohere: pip install cohere")

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise LLMError("Set COHERE_API_KEY environment variable")

    co = cohere.Client(api_key)
    model_name = model or os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
    input_type = os.getenv("COHERE_INPUT_TYPE", "search_document")  # or "search_query"

    vectors: List[List[float]] = []
    B = int(os.getenv("EMBED_BATCH", "96"))
    for i in range(0, len(texts), B):
        chunk = texts[i:i + B]
        try:
            resp = co.embed(texts=chunk, model=model_name, input_type=input_type)
            if hasattr(resp, "embeddings") and isinstance(resp.embeddings, list):
                vectors.extend(resp.embeddings)
        except Exception as e:
            logger.exception("Cohere embedding failed: %s", e)
            raise LLMError(f"Cohere embedding failed: {e}")
    return _normalize(vectors)

# ---- Hugging Face API embeddings ----------------------------------------------
def _embed_huggingface(texts: List[str], model: Optional[str] = None) -> np.ndarray:
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        raise LLMError("Install huggingface_hub: pip install huggingface_hub")

    api_key = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise LLMError("Set HF_API_KEY or HUGGINGFACE_API_KEY environment variable")

    model_name = model or os.getenv("HF_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
    client = InferenceClient(token=api_key)

    vectors: List[List[float]] = []
    for text in texts:  # HF API typically processes one at a time
        try:
            emb = client.feature_extraction(text, model=model_name)
            vectors.append(emb if isinstance(emb, list) else emb.tolist())
        except Exception as e:
            logger.exception("HuggingFace embedding failed: %s", e)
            raise LLMError(f"HuggingFace embedding failed: {e}")
    return _normalize(vectors)

# ---- Qwen embeddings -----------------------------------------------------------
def _embed_qwen(texts: List[str], model: Optional[str] = None) -> np.ndarray:
    import requests

    api_key = os.getenv("QWEN_API_KEY")
    base_url = os.getenv("QWEN_BASE_URL")
    model_name = model or os.getenv("QWEN_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
    if not all([api_key, base_url]):
        raise LLMError("Set QWEN_API_KEY and QWEN_BASE_URL for qwen provider")
    endpoint = base_url.rstrip("/") + "/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    B = int(os.getenv("QWEN_EMBED_BATCH", os.getenv("EMBED_BATCH", "50")))
    vectors: List[List[float]] = []

    for i in range(0, len(texts), B):
        chunk = texts[i:i + B]
        payload = {"model": model_name, "input": chunk}
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=int(os.getenv("QWEN_TIMEOUT", "60")))
            if resp.status_code != 200:
                raise LLMError(f"Qwen embedding HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            entries = data.get("data") or []
            for e in entries:
                vectors.append(e.get("embedding") or [])
        except Exception as e:
            logger.exception("Qwen embedding batch failed (%d:%d): %s", i, i + B, e)
            dim = len(vectors[-1]) if vectors else int(os.getenv("QWEN_DIM", "4096"))
            vectors.extend([[0.0] * dim for _ in chunk])

    return _normalize(vectors)

# ---- Local embeddings (sentence-transformers) ---------------------------------
def _embed_local(texts: List[str], model: Optional[str] = None) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise LLMError("Install sentence-transformers to use local embeddings") from e

    model_name = model or os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    st_model = SentenceTransformer(model_name)
    arr = st_model.encode(texts, normalize_embeddings=True)
    return np.array(arr, dtype="float32")
