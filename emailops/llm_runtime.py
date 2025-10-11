#!/usr/bin/env python3
# emailops/llm_runtime.py
from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------------------------------------
# Public error type (unifies previous split across env_utils/llm_client)
# --------------------------------------------------------------------------------------
class LLMError(Exception):
    """Custom exception for LLM- and embedding-related errors."""

    pass


# --------------------------------------------------------------------------------------
# Logging (library-safe)
# --------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Accounts / validation (from env_utils, consolidated)
# --------------------------------------------------------------------------------------
_validated_accounts: list[dict[str, Any]] | None = None
_vertex_initialized = False


@dataclass
class VertexAccount:
    """Validated Vertex AI account configuration"""

    project_id: str
    credentials_path: str
    account_group: int = 0
    is_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "credentials_path": self.credentials_path,
            "account_group": self.account_group,
        }


# Same defaults you already ship
DEFAULT_ACCOUNTS = [
    {
        "project_id": "api-agent-470921",
        "credentials_path": "secrets/api-agent-470921-4e2065b2ecf9.json",
    },
    {
        "project_id": "apt-arcana-470409-i7",
        "credentials_path": "secrets/apt-arcana-470409-i7-ce42b76061bf.json",
    },
    {
        "project_id": "embed2-474114",
        "credentials_path": "secrets/embed2-474114-fca38b4d2068.json",
    },
    {
        "project_id": "crafty-airfoil-474021-s2",
        "credentials_path": "secrets/crafty-airfoil-474021-s2-34159960925b.json",
    },
    {
        "project_id": "my-project-31635v",
        "credentials_path": "secrets/my-project-31635v-8ec357ac35b2.json",
    },
    {
        "project_id": "semiotic-nexus-470620-f3",
        "credentials_path": "secrets/semiotic-nexus-470620-f3-3240cfaf6036.json",
    },
]


def load_validated_accounts(
    validated_file: str = "validated_accounts.json",
    default_accounts: list[dict[str, str]] | None = None,
) -> list[VertexAccount]:
    """Load and validate GCP accounts (merged from env_utils)."""
    global _validated_accounts
    if _validated_accounts is not None:
        return _validated_accounts

    accounts: list[VertexAccount] = []
    vf = Path(validated_file)

    # Try validated_accounts.json
    if vf.exists():
        try:
            data = json.loads(vf.read_text(encoding="utf-8"))
            account_list = data.get("accounts", [])
            logger.info(
                "Loaded %d validated accounts from %s",
                len(account_list),
                validated_file,
            )
            for idx, acc in enumerate(account_list):
                accounts.append(
                    VertexAccount(
                        project_id=acc["project_id"],
                        credentials_path=acc["credentials_path"],
                        account_group=(0 if idx < 3 else 1),
                        is_valid=True,
                    )
                )
        except Exception as e:
            logger.warning("Failed to load validated accounts: %s", e)

    # Fallback
    if not accounts and default_accounts:
        logger.info("Using default accounts (no validated_accounts.json found)")
        for idx, acc in enumerate(default_accounts):
            accounts.append(
                VertexAccount(
                    project_id=acc["project_id"],
                    credentials_path=acc["credentials_path"],
                    account_group=(0 if idx < 3 else 1),
                    is_valid=True,
                )
            )

    # Validate credential files
    valid_accounts: list[VertexAccount] = []
    for acc in accounts:
        p = Path(acc.credentials_path)
        if not p.is_absolute():
            # Try multiple possible locations for the credentials
            possible_paths = [
                Path(__file__).resolve().parent.parent
                / acc.credentials_path,  # From project root
                Path(__file__).resolve().parent.parent
                / "secrets"
                / Path(acc.credentials_path).name,  # In secrets dir
                Path(acc.credentials_path),  # As given
            ]
            found = False
            for test_path in possible_paths:
                if test_path.exists():
                    p = test_path
                    acc.credentials_path = str(test_path)  # Update to absolute path
                    found = True
                    break
            if not found:
                logger.warning(
                    "Credentials file not found for %s: %s (tried multiple locations)",
                    acc.project_id,
                    acc.credentials_path,
                )
                acc.is_valid = False
                continue
        if p.exists():
            valid_accounts.append(acc)
        else:
            logger.warning("Credentials file not found for %s: %s", acc.project_id, p)
            acc.is_valid = False
    if not valid_accounts:
        raise LLMError(
            "No valid GCP accounts found. Provide validated_accounts.json or valid files in secrets/."
        )
    _validated_accounts = valid_accounts
    return valid_accounts


def save_validated_accounts(
    accounts: list[VertexAccount], output_file: str = "validated_accounts.json"
) -> None:
    """Persist validated accounts list (merged)."""
    from datetime import datetime

    data = {
        "accounts": [a.to_dict() for a in accounts if a.is_valid],
        "timestamp": datetime.now().isoformat(),
        "total_working": len([a for a in accounts if a.is_valid]),
    }
    Path(output_file).write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved %d validated accounts to %s", data["total_working"], output_file)


def reset_vertex_init() -> None:
    """Reset init state (used by project rotation)."""
    global _vertex_initialized
    _vertex_initialized = False
    logger.debug("Vertex AI initialization state reset")


def _init_vertex(
    project_id: str | None = None,
    credentials_path: str | None = None,
    location: str | None = None,
) -> None:
    """
    Initialize Vertex AI SDK with proper credentials.
    This unifies previous env_utils._init_vertex used by llm_client.  (Compatibility preserved.)
    """
    global _vertex_initialized
    if _vertex_initialized:
        return

    import vertexai  # google-cloud-aiplatform

    project = (
        project_id
        or os.getenv("VERTEX_PROJECT")
        or os.getenv("GCP_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    location = (
        location
        or os.getenv("VERTEX_LOCATION")
        or os.getenv("GCP_REGION")
        or os.getenv("GOOGLE_CLOUD_REGION")
        or "global"
    )
    if not project:
        raise LLMError(
            "GCP project not specified. Set GCP_PROJECT/GOOGLE_CLOUD_PROJECT or pass project_id."
        )

    service_account_path = (
        credentials_path
        or os.getenv("VERTEX_SERVICE_ACCOUNT_JSON")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )
    try:
        if service_account_path:
            cp = Path(service_account_path)
            if not cp.is_absolute():
                cp = Path(__file__).resolve().parent.parent / service_account_path
            if cp.exists():
                from google.oauth2 import service_account

                credentials = service_account.Credentials.from_service_account_file(
                    str(cp)
                )
                vertexai.init(
                    project=project, location=location, credentials=credentials
                )
                logger.info("✔ Vertex AI initialized with service account: %s", cp.name)
            else:
                logger.warning(
                    "GOOGLE_APPLICATION_CREDENTIALS points to %s but file not found; using ADC.",
                    service_account_path,
                )
                vertexai.init(project=project, location=location)
        else:
            vertexai.init(project=project, location=location)
        _vertex_initialized = True
    except Exception as e:
        raise LLMError(f"Failed to initialize Vertex AI SDK: {e}") from e


def validate_account(account: VertexAccount) -> tuple[bool, str]:
    """Quick account validation (unchanged behavior)."""
    p = Path(account.credentials_path)
    if not p.exists():
        return False, f"Credentials file not found: {account.credentials_path}"
    try:
        reset_vertex_init()
        _init_vertex(
            project_id=account.project_id, credentials_path=account.credentials_path
        )
        return True, "OK"
    except Exception as e:
        return False, str(e)


# --------------------------------------------------------------------------------------
# Project rotation (previously split; now co-located)
# --------------------------------------------------------------------------------------
_PROJECT_ROTATION: dict[str, Any] = {
    "projects": [],  # [{"project_id": str, "credentials_path": str}, ...]
    "current_index": 0,
    "consecutive_errors": 0,
    "_initialized": False,
}
_PROJECT_ROTATION_LOCK = threading.Lock()


def _ensure_projects_loaded() -> None:
    if _PROJECT_ROTATION["_initialized"]:
        return
    with _PROJECT_ROTATION_LOCK:
        if _PROJECT_ROTATION["_initialized"]:
            return
        try:
            accounts = load_validated_accounts(default_accounts=DEFAULT_ACCOUNTS)
            _PROJECT_ROTATION["projects"] = [
                {"project_id": a.project_id, "credentials_path": a.credentials_path}
                for a in accounts
            ]
            logger.info(
                "Loaded %d projects for rotation", len(_PROJECT_ROTATION["projects"])
            )
        except Exception as e:
            logger.warning(
                "Failed to load validated accounts; using defaults only: %s", e
            )
            _PROJECT_ROTATION["projects"] = list(DEFAULT_ACCOUNTS)
        _PROJECT_ROTATION["_initialized"] = True


def _rotate_to_next_project() -> str:
    _ensure_projects_loaded()
    with _PROJECT_ROTATION_LOCK:
        if not _PROJECT_ROTATION["projects"]:
            logger.warning(
                "No projects available for rotation; staying on current env."
            )
            return (
                os.getenv("GCP_PROJECT")
                or os.getenv("GOOGLE_CLOUD_PROJECT")
                or "<unknown>"
            )

        idx = _PROJECT_ROTATION["current_index"]
        _PROJECT_ROTATION["current_index"] = (idx + 1) % len(
            _PROJECT_ROTATION["projects"]
        )
        conf = _PROJECT_ROTATION["projects"][_PROJECT_ROTATION["current_index"]]

        creds_path = conf["credentials_path"]
        if not os.path.isabs(creds_path):
            creds_path = str(Path(__file__).resolve().parent.parent / creds_path)

        os.environ["GCP_PROJECT"] = conf["project_id"]
        os.environ["GOOGLE_CLOUD_PROJECT"] = conf["project_id"]
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

        reset_vertex_init()
        logger.warning("🔄 Rotating to project: %s", conf["project_id"])
        return conf["project_id"]


# --------------------------------------------------------------------------------------
# Retry / transient error classification (kept as-is)
# --------------------------------------------------------------------------------------
try:
    from google.api_core import exceptions as gax_exceptions  # type: ignore
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
            c
            for c in (
                getattr(gax_exceptions, "ResourceExhausted", None),
                getattr(gax_exceptions, "TooManyRequests", None),
                getattr(gax_exceptions, "ServiceUnavailable", None),
                getattr(gax_exceptions, "InternalServerError", None),
                getattr(gax_exceptions, "DeadlineExceeded", None),
            )
            if c is not None
        )
        if retry_types and isinstance(err, retry_types):
            return True
    return any(s in str(err).lower() for s in RETRYABLE_SUBSTRINGS)


def _should_rotate_on(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        ("429" in msg)
        or ("resource_exhausted" in msg)
        or ("quota" in msg)
        or ("rate limit" in msg)
    )


def _sleep_with_backoff(attempt: int, base: float, max_delay: float) -> None:
    sleep_for = min(max_delay, base * (2 ** (attempt - 1)))
    jitter = random.uniform(0, sleep_for * 0.2)
    time.sleep(sleep_for + jitter)


# --------------------------------------------------------------------------------------
# Vertex model helpers
# --------------------------------------------------------------------------------------
def _vertex_model(system_instruction: str | None = None):
    from vertexai.generative_models import GenerativeModel

    name = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")
    return GenerativeModel(name, system_instruction=system_instruction)


def _normalize_model_alias(name: str | None) -> str | None:
    return {"gemini-embedded-001": "gemini-embedding-001"}.get(name, name)


# --------------------------------------------------------------------------------------
# Text completion (Vertex)
# --------------------------------------------------------------------------------------
def complete_text(
    system: str,
    user: str,
    max_output_tokens: int = 1200,
    temperature: float = 0.2,
    stop_sequences: list[str] | None = None,
) -> str:
    """Vertex Gemini text completion with retry + project rotation."""
    attempts = int(os.getenv("VERTEX_MAX_RETRIES", "5"))
    base_delay = float(os.getenv("VERTEX_BACKOFF_INITIAL", "4"))
    max_delay = float(os.getenv("VERTEX_BACKOFF_MAX", "60"))

    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            _init_vertex()
            model = _vertex_model(system_instruction=system)
            cfg: dict[str, Any] = dict(
                max_output_tokens=max_output_tokens, temperature=temperature
            )
            if stop_sequences:
                cfg["stop_sequences"] = stop_sequences
            resp = model.generate_content(user, generation_config=cfg)
            return (getattr(resp, "text", None) or "").strip()
        except Exception as e:
            last_err = e
            if (attempt == attempts) or (not _is_retryable_error(e)):
                logger.exception("Vertex generate_content failed: %s", e)
                raise LLMError(str(e)) from e
            if _should_rotate_on(e):
                _rotate_to_next_project()
            _sleep_with_backoff(attempt, base_delay, max_delay)
    raise LLMError(str(last_err) if last_err else "Unknown error in complete_text")


def _extract_json_from_text(s: str) -> str:
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
    response_schema: dict | None = None,
    stop_sequences: list[str] | None = None,
) -> str:
    """JSON-mode completion with robust fallback to text+regex JSON salvage."""
    attempts = int(os.getenv("VERTEX_MAX_RETRIES", "5"))
    base_delay = float(os.getenv("VERTEX_BACKOFF_INITIAL", "4"))
    max_delay = float(os.getenv("VERTEX_BACKOFF_MAX", "60"))

    for attempt in range(1, attempts + 1):
        try:
            _init_vertex()
            from vertexai.generative_models import GenerationConfig

            model = _vertex_model(system_instruction=system)
            cfg: dict[str, Any] = dict(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                response_mime_type="application/json",
            )
            if response_schema:
                cfg["response_schema"] = response_schema
            if stop_sequences:
                cfg["stop_sequences"] = stop_sequences
            resp = model.generate_content(
                user, generation_config=GenerationConfig(**cfg)
            )
            return (getattr(resp, "text", None) or "").strip()
        except Exception as e:
            if (attempt == attempts) or (not _is_retryable_error(e)):
                logger.warning(
                    "JSON completion failed (%s), falling back to text mode", e
                )
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
    logger.warning("complete_json exhausted retries; returning empty JSON")
    return "{}"


# --------------------------------------------------------------------------------------
# Embeddings (Vertex/OpenAI/Azure/Cohere/HF/Qwen/local) – merged from llm_client
# --------------------------------------------------------------------------------------
def _normalize(vectors: list[list[float]]) -> np.ndarray:
    if not vectors:
        return np.zeros((0, 0), dtype="float32")
    arr = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return (arr / norms).astype("float32")


def embed_texts(
    texts: list[str],
    provider: str | None = None,
    model: str | None = None,
    **_: Any,
) -> np.ndarray:
    """
    Return array of shape (N, D) with unit-normalized embeddings.
    Providers: vertex | openai | azure | cohere | huggingface | qwen | local
    """
    provider = (provider or os.getenv("EMBED_PROVIDER", "vertex")).lower()
    if not texts:
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
    else:
        return _embed_local(texts, model=model)


def _embed_vertex(texts: list[str], model: str | None = None) -> np.ndarray:
    """Vertex embeddings with rotation on quota exhaustion (google‑genai path + legacy fallback)."""
    _init_vertex()
    _ensure_projects_loaded()

    embed_name = _normalize_model_alias(model) or os.getenv(
        "VERTEX_EMBED_MODEL", "gemini-embedding-001"
    )
    # Path 1: google-genai embed API (recommended for gemini-embedding*)
    if embed_name.startswith(("gemini-embedding", "gemini-embedder")):
        try:
            from google import genai
            from google.genai.types import EmbedContentConfig
        except Exception as e:
            raise LLMError(f"google-genai not installed: {e}")

        out_dim = os.getenv("VERTEX_EMBED_DIM")
        cfg = (
            EmbedContentConfig(output_dimensionality=int(out_dim)) if out_dim else None
        )
        dim = int(out_dim) if out_dim else 3072

        vectors: list[list[float]] = []
        B = min(int(os.getenv("EMBED_BATCH", "64")), 250)
        for i in range(0, len(texts), B):
            chunk = texts[i : i + B]
            embedded = False
            for _ in range(max(1, len(_PROJECT_ROTATION["projects"]) or 1)):
                project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
                try:
                    location = os.getenv("GCP_REGION", "global")
                    client = genai.Client(
                        vertexai=True, project=project, location=location
                    )
                    resp = client.models.embed_content(
                        model=embed_name, contents=chunk, config=cfg
                    )
                    if (
                        resp
                        and getattr(resp, "embeddings", None)
                        and len(resp.embeddings) == len(chunk)
                    ):
                        for emb in resp.embeddings:
                            vectors.append(
                                emb.values
                                if getattr(emb, "values", None)
                                else [0.0] * dim
                            )
                        embedded = True
                        _PROJECT_ROTATION["consecutive_errors"] = 0
                        break
                    logger.warning(
                        "Empty/mismatched embedding batch from %s; rotating", project
                    )
                    _rotate_to_next_project()
                    _init_vertex()
                except Exception as e:
                    if _should_rotate_on(e) and _PROJECT_ROTATION["projects"]:
                        logger.warning("Quota on batch; rotating project")
                        _rotate_to_next_project()
                        _init_vertex()
                        continue
                    logger.exception("Vertex (gemini) embedding failed: %s", e)
                    break
            if not embedded:
                logger.error(
                    "All embedding attempts failed for batch starting at %d; zero vectors",
                    i,
                )
                vectors.extend([[0.0] * dim] * len(chunk))
        return _normalize(vectors)

    # Path 2: legacy vertexai TextEmbeddingModel
    vectors: list[list[float]] = []
    B = min(int(os.getenv("EMBED_BATCH", "64")), 250)
    for i in range(0, len(texts), B):
        chunk = texts[i : i + B]
        success = False
        for _ in range(max(1, len(_PROJECT_ROTATION["projects"]) or 1)):
            try:
                from typing import cast

                from vertexai.language_models import (
                    TextEmbeddingInput,
                    TextEmbeddingModel,
                )  # legacy path

                model_obj = TextEmbeddingModel.from_pretrained(
                    os.getenv("VERTEX_EMBED_MODEL", "text-embedding-005")
                )
                inputs = [
                    TextEmbeddingInput(text=t) if isinstance(t, str) else t
                    for t in chunk
                ]
                embs = model_obj.get_embeddings(
                    cast(list[str | TextEmbeddingInput], inputs)
                )
                for e in embs:
                    vectors.append(e.values)
                success = True
                break
            except Exception as e:
                if _should_rotate_on(e) and _PROJECT_ROTATION["projects"]:
                    _rotate_to_next_project()
                    _init_vertex()
                    continue
                logger.exception(
                    "Vertex (legacy) embedding failed on batch %d:%d: %s", i, i + B, e
                )
                break
        if not success:
            dim = int(os.getenv("EMBED_DIM", "768"))
            vectors.extend([[0.0] * dim for _ in chunk])
    return _normalize(vectors)


def _embed_openai(texts: list[str], model: str | None = None) -> np.ndarray:
    try:
        from openai import OpenAI
    except ImportError:
        raise LLMError("Install openai: pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("Set OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    model_name = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    vectors: list[list[float]] = []
    B = int(os.getenv("EMBED_BATCH", "100"))
    for i in range(0, len(texts), B):
        chunk = texts[i : i + B]
        try:
            resp = client.embeddings.create(input=chunk, model=model_name)
            for item in resp.data:
                vectors.append(item.embedding)
        except Exception as e:
            logger.exception("OpenAI embedding failed: %s", e)
            raise LLMError(f"OpenAI embedding failed: {e}")
    return _normalize(vectors)


def _embed_azure_openai(texts: list[str], model: str | None = None) -> np.ndarray:
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise LLMError("Install openai: pip install openai")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = model or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not all([api_key, endpoint, deployment]):
        raise LLMError(
            "Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT"
        )
    client = AzureOpenAI(
        api_key=api_key,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_endpoint=str(endpoint),
    )
    vectors: list[list[float]] = []
    B = int(os.getenv("EMBED_BATCH", "100"))
    for i in range(0, len(texts), B):
        chunk = texts[i : i + B]
        try:
            resp = client.embeddings.create(input=chunk, model=str(deployment))
            for item in resp.data:
                vectors.append(item.embedding)
        except Exception as e:
            logger.exception("Azure OpenAI embedding failed: %s", e)
            raise LLMError(f"Azure OpenAI embedding failed: {e}")
    return _normalize(vectors)


def _embed_cohere(texts: list[str], model: str | None = None) -> np.ndarray:
    try:
        import cohere
    except ImportError:
        raise LLMError("Install cohere: pip install cohere")
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise LLMError("Set COHERE_API_KEY")
    co = cohere.Client(api_key)
    model_name = model or os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
    input_type = os.getenv("COHERE_INPUT_TYPE", "search_document")
    vectors: list[list[float]] = []
    B = int(os.getenv("EMBED_BATCH", "96"))
    for i in range(0, len(texts), B):
        chunk = texts[i : i + B]
        try:
            resp = co.embed(texts=chunk, model=model_name, input_type=input_type)
            if hasattr(resp, "embeddings") and isinstance(resp.embeddings, list):
                vectors.extend(resp.embeddings)
        except Exception as e:
            logger.exception("Cohere embedding failed: %s", e)
            raise LLMError(f"Cohere embedding failed: {e}")
    return _normalize(vectors)


def _embed_huggingface(texts: list[str], model: str | None = None) -> np.ndarray:
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        raise LLMError("Install huggingface_hub: pip install huggingface_hub")
    api_key = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise LLMError("Set HF_API_KEY or HUGGINGFACE_API_KEY")
    model_name = model or os.getenv("HF_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
    client = InferenceClient(token=api_key)
    vectors: list[list[float]] = []
    for text in texts:  # HF API typically 1-at-a-time
        try:
            emb = client.feature_extraction(text, model=model_name)
            vectors.append(emb if isinstance(emb, list) else emb.tolist())
        except Exception as e:
            logger.exception("HuggingFace embedding failed: %s", e)
            raise LLMError(f"HuggingFace embedding failed: {e}")
    return _normalize(vectors)


def _embed_qwen(texts: list[str], model: str | None = None) -> np.ndarray:
    import requests

    api_key = os.getenv("QWEN_API_KEY")
    base_url = os.getenv("QWEN_BASE_URL")
    model_name = model or os.getenv("QWEN_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
    if not all([api_key, base_url]):
        raise LLMError("Set QWEN_API_KEY and QWEN_BASE_URL")
    endpoint = base_url.rstrip("/") + "/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    B = int(os.getenv("QWEN_EMBED_BATCH", os.getenv("EMBED_BATCH", "50")))
    vectors: list[list[float]] = []
    for i in range(0, len(texts), B):
        chunk = texts[i : i + B]
        payload = {"model": model_name, "input": chunk}
        try:
            resp = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=int(os.getenv("QWEN_TIMEOUT", "60")),
            )
            if resp.status_code != 200:
                raise LLMError(
                    f"Qwen embedding HTTP {resp.status_code}: {resp.text[:200]}"
                )
            data = resp.json()
            for e in data.get("data") or []:
                vectors.append(e.get("embedding") or [])
        except Exception as e:
            logger.exception("Qwen batch failed %d:%d: %s", i, i + B, e)
            dim = len(vectors[-1]) if vectors else int(os.getenv("QWEN_DIM", "4096"))
            vectors.extend([[0.0] * dim for _ in chunk])
    return _normalize(vectors)


def _embed_local(texts: list[str], model: str | None = None) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise LLMError("Install sentence-transformers to use local embeddings") from e
    name = model or os.getenv(
        "LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    st = SentenceTransformer(name)
    arr = st.encode(texts, normalize_embeddings=True)
    return np.array(arr, dtype="float32")


# --------------------------------------------------------------------------------------
# (Optional) Convenience re-exports from utils so callers can import from one place
# --------------------------------------------------------------------------------------
try:
    # These are used widely by indexer/search/summarizer; re-export for a single surface.
    pass
except Exception:
    # If utils isn't accessible for some reason, the runtime remains functional for LLM ops.
    pass
