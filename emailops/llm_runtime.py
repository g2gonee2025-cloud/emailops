#!/usr/bin/env python3
# emailops/llm_runtime.py
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import EmailOpsConfig, get_config

# Import utilities from utils.py
from .utils import (
    clean_email_text,
    logger,
    monitor_performance,
    read_text_file,
)

# Import for control char removal
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")

def _strip_control_chars(s: str) -> str:
    """Remove non-printable control characters and normalize newlines."""
    if not s:
        return ""
    # Normalize CRLF/CR -> LF and strip control characters
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_CHARS.sub("", s)


# --------------------------------------------------------------------------------------
# Public error type (unifies previous split across env_utils/llm_client)
# --------------------------------------------------------------------------------------
class LLMError(Exception):
    """Custom exception for LLM- and embedding-related errors."""

    pass


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
# Use centralized configuration instead of direct environment variables
_config = get_config()


# --------------------------------------------------------------------------------------
# HIGH #14: Rate limiting for API calls
# --------------------------------------------------------------------------------------
_API_CALL_TIMES = deque(maxlen=1000)
_RATE_LIMIT_PER_MINUTE = int(os.getenv("API_RATE_LIMIT", "60"))  # Keep for backward compatibility
_RATE_LIMIT_LOCK = threading.Lock()


def _check_rate_limit() -> None:
    """Enforce per-minute rate limit without holding the lock while sleeping.

    This avoids blocking other threads for the entire sleep duration and yields
    throughput that more closely matches a token-bucket.
    """
    while True:
        with _RATE_LIMIT_LOCK:
            now = time.time()
            # Remove calls older than 1 minute
            while _API_CALL_TIMES and now - _API_CALL_TIMES[0] > 60:
                _API_CALL_TIMES.popleft()

            if len(_API_CALL_TIMES) < _RATE_LIMIT_PER_MINUTE:
                _API_CALL_TIMES.append(now)
                return

            # Compute how long until the oldest timestamp expires
            sleep_time = max(0.0, 60 - (now - _API_CALL_TIMES[0]))

        if sleep_time > 0:
            logger.info("Rate limit reached; sleeping %.1f seconds", sleep_time)
            time.sleep(sleep_time)
        else:
            # Yield briefly to avoid busy-waiting if clocks are skewed
            time.sleep(0.01)


# --------------------------------------------------------------------------------------
# Accounts / validation (from env_utils, consolidated)
# --------------------------------------------------------------------------------------
_validated_accounts: list[VertexAccount] | None = None
_vertex_initialized = False
_INIT_LOCK = threading.RLock()
_VALIDATED_LOCK = threading.RLock()


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
    with _VALIDATED_LOCK:
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

    # HIGH #15: Enhanced credential validation
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
            # Validate it's actually a valid service account file
            try:
                with p.open('r') as f:
                    cred_data = json.load(f)
                # Check required fields
                required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
                missing = [f for f in required_fields if f not in cred_data]
                if missing:
                    logger.warning("Invalid credentials file for %s (missing: %s)",
                                  acc.project_id, missing)
                    acc.is_valid = False
                    continue
                if cred_data.get('type') != 'service_account':
                    logger.warning("Not a service account file: %s", p)
                    acc.is_valid = False
                    continue
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Cannot read credentials for %s: %s", acc.project_id, e)
                acc.is_valid = False
                continue
            valid_accounts.append(acc)
        else:
            logger.warning("Credentials file not found for %s: %s", acc.project_id, p)
            acc.is_valid = False
    if not valid_accounts:
        raise LLMError(
            "No valid GCP accounts found. Provide validated_accounts.json or valid files in secrets/."
        )
    with _VALIDATED_LOCK:
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
    with _INIT_LOCK:
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
        with _INIT_LOCK:
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

        # Validate paths exist and are absolute before exposing
        creds_path = conf["credentials_path"]
        if not Path(creds_path).is_absolute():
            creds_path = str(Path(__file__).resolve().parent.parent / creds_path)

        # Verify file exists and is readable
        if not Path(creds_path).exists():
            logger.error("Credentials file not found: %s", creds_path)
            return os.getenv("GCP_PROJECT") or "<unknown>"

        os.environ["GCP_PROJECT"] = conf["project_id"]
        os.environ["GOOGLE_CLOUD_PROJECT"] = conf["project_id"]
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

        reset_vertex_init()

        # MEDIUM #20: Add rotation metrics logging
        logger.debug(
            "Project rotation: %d/%d, consecutive_errors=%d",
            _PROJECT_ROTATION["current_index"] + 1,
            len(_PROJECT_ROTATION["projects"]),
            _PROJECT_ROTATION.get("consecutive_errors", 0)
        )
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
    """
    Create a Vertex AI GenerativeModel instance with configured model name.

    Args:
        system_instruction: Optional system instruction for model behavior

    Returns:
        GenerativeModel instance configured with VERTEX_MODEL env var (default: gemini-2.5-pro)

    Raises:
        ImportError: If vertexai module not available

    Example:
        >>> model = _vertex_model("You are a helpful assistant")
    """
    from vertexai.generative_models import GenerativeModel

    name = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")
    return GenerativeModel(name, system_instruction=system_instruction)


def _normalize_model_alias(name: str | None) -> str | None:
    """
    Normalize common model name aliases to canonical names.

    Args:
        name: Model name to normalize (may be None)

    Returns:
        Normalized model name, or None if input was None

    Example:
        >>> _normalize_model_alias("gemini-embedding-001")
        'gemini-embedding-001'
    """
    if name is None:
        return None
    return {"gemini-embedding-001": "gemini-embedding-001"}.get(name, name)


# --------------------------------------------------------------------------------------
# Text completion (Vertex)
# --------------------------------------------------------------------------------------
@monitor_performance
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
            from vertexai.generative_models import (
                GenerationConfig,
                HarmBlockThreshold,
                HarmCategory,
            )
            # Disable safety filters for business use
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
            cfg = GenerationConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences or None,
            )
            _check_rate_limit()
            resp = model.generate_content(user, generation_config=cfg, safety_settings=safety_settings)
            text = (getattr(resp, "text", None) or "").strip()
            if not text:
                raise LLMError("Empty completion from model")
            return text
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
    """Best-effort salvage of JSON object or array from arbitrary text (or fenced code)."""
    if not s:
        return "{}"
    s = s.strip()
    # Prefer fenced blocks if present; allow ```json, ```json5, ```hjson
    fence = re.search(r"```(?:json|json5|hjson)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    candidate = fence.group(1).strip() if fence else s
    # Extract either an object {…} or an array […]; return {} if nothing matches
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", candidate)
    return m.group(1).strip() if m else "{}"


@monitor_performance
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
            from vertexai.generative_models import (
                GenerationConfig,
                HarmBlockThreshold,
                HarmCategory,
            )

            model = _vertex_model(system_instruction=system)
            # Disable safety filters for business use
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
            cfg: dict[str, Any] = {
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "response_mime_type": "application/json",
            }
            if response_schema:
                cfg["response_schema"] = response_schema
            if stop_sequences:
                cfg["stop_sequences"] = stop_sequences
            _check_rate_limit()
            resp = model.generate_content(user, generation_config=GenerationConfig(**cfg), safety_settings=safety_settings)
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
# Embeddings (Vertex/OpenAI/Azure/Cohere/HF/Qwen/local) - merged from llm_client
# --------------------------------------------------------------------------------------
def _normalize(vectors: list[list[float]]) -> np.ndarray:
    """
    Normalize embedding vectors to unit length (L2 norm = 1).

    Args:
        vectors: List of embedding vectors as lists of floats

    Returns:
        NumPy array of shape (N, D) with unit-normalized vectors (float32)
        Returns empty (0, 0) array if input is empty

    Note:
        Adds small epsilon (1e-12) to avoid division by zero
    """
    if not vectors:
        return np.zeros((0, 0), dtype="float32")
    arr = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return (arr / norms).astype("float32")


@monitor_performance
def embed_texts(
    texts: Iterable[str],
    provider: str | None = None,
    model: str | None = None,
    **_: Any,
) -> np.ndarray:
    """
    Return array of shape (N, D) with unit-normalized embeddings.
    Providers: vertex | openai | azure | cohere | huggingface | qwen | local
    """
    seq = list(texts)
    provider = (provider or os.getenv("EMBED_PROVIDER", "vertex")).lower()
    if not seq:
        return np.zeros((0, 0), dtype="float32")

    # Run provider-specific embedding, then apply final sanity checks before returning.
    if provider == "vertex":
        arr = _embed_vertex(seq, model=model)
    elif provider == "openai":
        arr = _embed_openai(seq, model=model)
    elif provider == "azure":
        arr = _embed_azure_openai(seq, model=model)
    elif provider == "cohere":
        arr = _embed_cohere(seq, model=model)
    elif provider == "huggingface":
        arr = _embed_huggingface(seq, model=model)
    elif provider == "qwen":
        arr = _embed_qwen(seq, model=model)
    else:
        arr = _embed_local(seq, model=model)

    # -------- Final sanity checks (post-normalization) --------
    arr = np.asarray(arr, dtype="float32")
    if arr.ndim != 2:
        raise LLMError(f"Embedding array must be 2-D; got shape {arr.shape}")
    if arr.shape[0] != len(seq):
        raise LLMError(
            f"Embedding row count mismatch: got {arr.shape[0]}, expected {len(seq)}"
        )
    if not np.isfinite(arr).all():
        raise LLMError("Non-finite values found in embeddings")
    # Norms should not all be ~0 after normalization; catch broken providers.
    with np.errstate(invalid="ignore"):
        row_norms = np.linalg.norm(arr, axis=1)
    if row_norms.size and float(np.max(row_norms)) < 1e-3:
        raise LLMError("Embeddings appear degenerate (near-zero norms); aborting")
    return arr


def _embed_vertex(texts: list[str], model: str | None = None) -> np.ndarray:
    """Vertex embeddings with rotation on quota exhaustion (google-genai path + legacy fallback)."""
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
            raise LLMError(f"google-genai not installed: {e}") from e

        out_dim = os.getenv("VERTEX_EMBED_DIM")
        cfg = (
            EmbedContentConfig(output_dimensionality=int(out_dim)) if out_dim else None
        )

        vectors: list[list[float]] = []
        B = min(int(os.getenv("EMBED_BATCH", "64")), 250)
        for i in range(0, len(texts), B):
            chunk = texts[i : i + B]
            embedded = False
            # CRITICAL FIX #6: Thread-safe access to _PROJECT_ROTATION
            with _PROJECT_ROTATION_LOCK:
                max_retries = max(1, len(_PROJECT_ROTATION["projects"]) or 1)
            for _ in range(max_retries):
                project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
                try:
                    location = os.getenv("GCP_REGION", "global")
                    from typing import cast
                    client = genai.Client(
                        vertexai=True, project=project, location=location
                    )
                    _check_rate_limit()
                    # Robust embedding: prefer batch API when available, otherwise per-item.
                    embeddings_batch: list[list[float]] = []
                    # Try batch API if provided by google-genai
                    batch_fn = getattr(client.models, "batch_embed_contents", None)
                    if callable(batch_fn):
                        try:
                            requests = [{"content": t} for t in chunk]
                            _check_rate_limit()
                            resp = batch_fn(model=embed_name, requests=requests, config=cfg)
                            items = getattr(resp, "embeddings", None) or getattr(resp, "data", None) or []
                            for item in items:
                                vals = None
                                if hasattr(item, "values"):
                                    vals = item.values
                                elif hasattr(item, "embedding") and hasattr(item.embedding, "values"):
                                    vals = item.embedding.values
                                if vals is None:
                                    raise RuntimeError("Empty embedding values in batch response")
                                embeddings_batch.append(vals)  # type: ignore[arg-type]
                        except Exception:
                            embeddings_batch = []  # fall through to per-item path
                    if not embeddings_batch:
                        # Fallback: per-item embed_content; handle both 'embedding' and 'embeddings'
                        for _t in chunk:
                            _check_rate_limit()
                            res = client.models.embed_content(model=embed_name, content=_t, config=cfg)
                            vals = None
                            if hasattr(res, "values"):
                                vals = res.values
                            elif hasattr(res, "embedding") and hasattr(res.embedding, "values"):
                                vals = res.embedding.values
                            elif hasattr(res, "embeddings") and res.embeddings:
                                first = res.embeddings[0]
                                vals = getattr(first, "values", None)
                            if vals is None:
                                embeddings_batch = []
                                break
                            embeddings_batch.append(vals)  # type: ignore[arg-type]
                    if len(embeddings_batch) == len(chunk):
                        vectors.extend(embeddings_batch)
                        embedded = True
                        with _PROJECT_ROTATION_LOCK:
                            _PROJECT_ROTATION["consecutive_errors"] = 0
                        break
                    logger.warning(
                        "Empty/mismatched embedding batch from %s; rotating", project
                    )
                    with _PROJECT_ROTATION_LOCK:
                        _PROJECT_ROTATION["consecutive_errors"] = _PROJECT_ROTATION.get("consecutive_errors", 0) + 1
                    _rotate_to_next_project()
                    _init_vertex()
                except Exception as e:
                    # CRITICAL FIX #6: Thread-safe access to _PROJECT_ROTATION
                    with _PROJECT_ROTATION_LOCK:
                        has_projects = bool(_PROJECT_ROTATION["projects"])
                    if _should_rotate_on(e) and has_projects:
                        logger.warning("Quota on batch; rotating project")
                        with _PROJECT_ROTATION_LOCK:
                            _PROJECT_ROTATION["consecutive_errors"] = _PROJECT_ROTATION.get("consecutive_errors", 0) + 1
                        _rotate_to_next_project()
                        _init_vertex()
                        continue
                    logger.exception("Vertex (gemini) embedding failed: %s", e)
                    break
            if not embedded:
                # Exhausted rotations (if any). Do not substitute zero vectors.
                raise LLMError(
                    f"Vertex (gemini) embedding failed for batch starting at index {i}; "
                    "exhausted rotation attempts."
                )
        return _normalize(vectors)

    # Path 2: legacy vertexai TextEmbeddingModel
    vectors: list[list[float]] = []
    B = min(int(os.getenv("EMBED_BATCH", "64")), 250)
    for i in range(0, len(texts), B):
        chunk = texts[i : i + B]
        success = False
        # CRITICAL FIX #6: Thread-safe access to _PROJECT_ROTATION
        with _PROJECT_ROTATION_LOCK:
            max_retries = max(1, len(_PROJECT_ROTATION["projects"]) or 1)
        for _ in range(max_retries):
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
                _check_rate_limit()
                embs = model_obj.get_embeddings(
                    cast(list[str | TextEmbeddingInput], inputs)
                )
                # Fail the batch if any embedding is missing instead of returning zeros.
                batch_vals: list[list[float]] = []
                for e in embs:
                    vals = getattr(e, "values", None)
                    if vals is None:
                        raise RuntimeError(
                            "Vertex (legacy) returned empty embedding values"
                        )
                    batch_vals.append(vals)
                vectors.extend(batch_vals)
                success = True
                break
            except Exception as e:
                # CRITICAL FIX #6: Thread-safe access to _PROJECT_ROTATION
                with _PROJECT_ROTATION_LOCK:
                    has_projects = bool(_PROJECT_ROTATION["projects"])
                if _should_rotate_on(e) and has_projects:
                    _rotate_to_next_project()
                    _init_vertex()
                    continue
                logger.exception(
                    "Vertex (legacy) embedding failed on batch %d:%d: %s", i, i + B, e
                )
                break
        if not success:
            # Exhausted rotations (if any). Do not substitute zero vectors.
            raise LLMError(
                f"Vertex (legacy) embedding failed for batch {i}:{i + B}; "
                "exhausted rotation attempts."
            )
    return _normalize(vectors)


def _embed_openai(texts: list[str], model: str | None = None) -> np.ndarray:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise LLMError("Install openai: pip install openai") from e
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("Set OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    model_name = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    vectors: list[list[float]] = []
    B = int(os.getenv("EMBED_BATCH", "64"))
    for i in range(0, len(texts), B):
        chunk = texts[i : i + B]
        try:
            _check_rate_limit()
            resp = client.embeddings.create(input=chunk, model=model_name)
            for item in resp.data:
                vectors.append(item.embedding)
        except Exception as e:
            logger.exception("OpenAI embedding failed: %s", e)
            raise LLMError(f"OpenAI embedding failed: {e}") from e
    return _normalize(vectors)


def _embed_azure_openai(texts: list[str], model: str | None = None) -> np.ndarray:
    try:
        from openai import AzureOpenAI
    except ImportError as e:
        raise LLMError("Install openai: pip install openai") from e
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
    B = int(os.getenv("EMBED_BATCH", "64"))
    for i in range(0, len(texts), B):
        chunk = texts[i : i + B]
        try:
            _check_rate_limit()
            resp = client.embeddings.create(input=chunk, model=str(deployment))
            for item in resp.data:
                vectors.append(item.embedding)
        except Exception as e:
            logger.exception("Azure OpenAI embedding failed: %s", e)
            raise LLMError(f"Azure OpenAI embedding failed: {e}") from e
    return _normalize(vectors)


def _embed_cohere(texts: list[str], model: str | None = None) -> np.ndarray:
    try:
        import cohere
    except ImportError as e:
        raise LLMError("Install cohere: pip install cohere") from e
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise LLMError("Set COHERE_API_KEY")
    co = cohere.Client(api_key)
    model_name = model or os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
    input_type = os.getenv("COHERE_INPUT_TYPE", "search_document")
    vectors: list[list[float]] = []
    B = int(os.getenv("EMBED_BATCH", "64"))
    for i in range(0, len(texts), B):
        chunk = texts[i : i + B]
        try:
            _check_rate_limit()
            resp = co.embed(texts=chunk, model=model_name, input_type=input_type)
            if hasattr(resp, "embeddings") and isinstance(resp.embeddings, list):
                vectors.extend(resp.embeddings)
        except Exception as e:
            logger.exception("Cohere embedding failed: %s", e)
            raise LLMError(f"Cohere embedding failed: {e}") from e
    return _normalize(vectors)


def _embed_huggingface(texts: list[str], model: str | None = None) -> np.ndarray:
    try:
        from huggingface_hub import InferenceClient
    except ImportError as e:
        raise LLMError("Install huggingface_hub: pip install huggingface_hub") from e
    api_key = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise LLMError("Set HF_API_KEY or HUGGINGFACE_API_KEY")
    model_name = model or os.getenv("HF_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
    client = InferenceClient(token=api_key)
    vectors: list[list[float]] = []
    for text in texts:  # HF API typically 1-at-a-time
        try:
            _check_rate_limit()
            emb = client.feature_extraction(text, model=model_name)
            vectors.append(emb if isinstance(emb, list) else emb.tolist())
        except Exception as e:
            logger.exception("HuggingFace embedding failed: %s", e)
            raise LLMError(f"HuggingFace embedding failed: {e}") from e
    return _normalize(vectors)


def _embed_qwen(texts: list[str], model: str | None = None) -> np.ndarray:
    import requests

    api_key = os.getenv("QWEN_API_KEY")
    base_url = os.getenv("QWEN_BASE_URL")
    model_name = model or os.getenv("QWEN_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
    if not all([api_key, base_url]):
        raise LLMError("Set QWEN_API_KEY and QWEN_BASE_URL")
    endpoint = (base_url.rstrip("/") if base_url else "") + "/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    B = int(os.getenv("EMBED_BATCH", "64"))
    vectors: list[list[float]] = []
    for i in range(0, len(texts), B):
        chunk = texts[i : i + B]
        payload = {"model": model_name, "input": chunk}
        try:
            _check_rate_limit()
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
            items = data.get("data") or []
            if len(items) != len(chunk):
                raise LLMError("Qwen embedding: row count mismatch")
            for e in items:
                emb = e.get("embedding")
                if not emb:
                    raise LLMError("Qwen embedding: empty vector returned")
                vectors.append(emb)
        except Exception as e:
            logger.exception("Qwen batch failed %d:%d: %s", i, i + B, e)
            raise LLMError(f"Qwen embedding failed for batch {i}:{i+B}: {e}") from e
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
# Convenience re-exports from utils so callers can import from one place
# --------------------------------------------------------------------------------------
# Re-export commonly used utilities for single import surface
__all__ = [
    # Configuration
    'EmailOpsConfig',
    # Error types
    'LLMError',
    # Account management
    'VertexAccount',
    'clean_email_text',
    'complete_json',
    # LLM functions
    'complete_text',
    'embed_texts',
    'get_config',
    'load_validated_accounts',
    # Utility re-exports from utils.py
    'logger',
    'monitor_performance',
    'read_text_file',
    'save_validated_accounts',
    'validate_account',
]

# Note: logger, clean_email_text, and read_text_file are already available
# from the imports at the top of the file
