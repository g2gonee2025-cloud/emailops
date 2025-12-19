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
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    pass

import numpy as np
import requests

# Optional imports - these may not be available
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    from google.api_core import exceptions as gax_exceptions  # type: ignore
except Exception:
    gax_exceptions = None  # type: ignore[assignment]

try:
    from google.oauth2 import service_account
except ImportError:
    service_account = None  # type: ignore

try:
    import vertexai  # google-cloud-aiplatform
    from vertexai.generative_models import (
        GenerationConfig,
        GenerativeModel,
        HarmBlockThreshold,
        HarmCategory,
    )
except ImportError:
    vertexai = None  # type: ignore
    GenerativeModel = None  # type: ignore
    GenerationConfig = None  # type: ignore
    HarmBlockThreshold = None  # type: ignore
    HarmCategory = None  # type: ignore

try:
    from vertexai.language_models import (  # legacy path
        TextEmbeddingInput,
        TextEmbeddingModel,
    )
except ImportError:
    TextEmbeddingInput = None  # type: ignore
    TextEmbeddingModel = None  # type: ignore

try:
    from google import genai
    from google.genai.types import EmbedContentConfig
except ImportError:
    genai = None  # type: ignore
    EmbedContentConfig = None  # type: ignore

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None  # type: ignore

try:
    from openai import AzureOpenAI, OpenAI
except ImportError:
    OpenAI = None  # type: ignore
    AzureOpenAI = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore

from .core_config import EmailOpsConfig, get_config
from .core_email_processing import clean_email_text
from .core_exceptions import LLMError

# Import utilities from utils.py
from .utils import (
    logger,
    monitor_performance,
)

# Import resilience patterns
try:
    from .services.resilience import circuit_breaker, with_retry
except ImportError:
    # Fallback if resilience patterns not available
    def with_retry(**_kwargs):
        def decorator(func):
            return func

        return decorator

    def circuit_breaker(**_kwargs):
        def decorator(func):
            return func

        return decorator


# Note: _strip_control_chars now imported from utils.py (Issue #1 fix)
# Use normalize_newlines=True for embedding/indexing pipeline consistency


# --------------------------------------------------------------------------------------
# Public error type (now imported from centralized exceptions module)
# --------------------------------------------------------------------------------------
# MEDIUM #32: Use centralized exception - LLMError now imported from .exceptions


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
# P0-24 FIX: Defer config loading to prevent circular imports
# Config is loaded lazily when first needed instead of at module import time
_config: EmailOpsConfig | None = None


def _get_config() -> EmailOpsConfig:
    """Lazy-load config to avoid circular import at module level."""
    global _config
    if _config is None:
        _config = get_config()
    return _config


# --------------------------------------------------------------------------------------
# HIGH #14: Rate limiting for API calls
# --------------------------------------------------------------------------------------
_API_CALL_TIMES: deque = deque(maxlen=1000)
_RATE_LIMIT_PER_MINUTE = int(
    os.getenv("API_RATE_LIMIT", "60")
)  # Keep for backward compatibility
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
            logger.info("Rate limit reached, sleeping %.1f seconds", sleep_time)
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


# P0-1 FIX: Remove hardcoded credentials - must load from validated_accounts.json
# This prevents accidental credential exposure in version control
DEFAULT_ACCOUNTS: list[dict[str, str]] = []


def load_validated_accounts(
    validated_file: str = "validated_accounts.json",
    default_accounts: list[dict[str, str]] | None = None,
) -> list[VertexAccount]:
    """
    Load and validate GCP accounts from external config files only.

    P0-1 FIX: Never uses hardcoded credentials. Accounts must come from:
    1. validated_accounts.json (preferred)
    2. Environment variables (GCP_PROJECT + GOOGLE_APPLICATION_CREDENTIALS)
    3. Application Default Credentials (ADC) with explicit project

    P0-49 FIX: Thread-safe double-checked locking prevents race conditions.

    Raises:
        LLMError: If no valid accounts can be loaded
    """
    global _validated_accounts

    # P0-49 FIX: Fast path check without lock
    if _validated_accounts is not None:
        return _validated_accounts

    # P0-49 FIX: Acquire lock for initialization
    with _VALIDATED_LOCK:
        # Double-check after acquiring lock
        if _validated_accounts is not None:
            return _validated_accounts

        accounts: list[VertexAccount] = []

    # Try multiple validated account file locations
    search_paths = [
        Path(validated_file).expanduser(),
        Path.home() / ".emailops" / validated_file,
        Path(__file__).resolve().parent.parent / validated_file,
    ]

    for vf in search_paths:
        if vf.exists():
            try:
                data = json.loads(vf.read_text(encoding="utf-8"))
                account_list = data.get("accounts", [])
                logger.info(
                    "Loaded %d validated accounts from %s",
                    len(account_list),
                    vf,
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
                break  # Successfully loaded from this file
            except Exception as e:
                logger.warning("Failed to load accounts from %s: %s", vf, e)
                continue

    # Fallback to environment variables if no file found
    if not accounts:
        project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

        if project:
            logger.info("Loading single account from environment variables")
            accounts.append(
                VertexAccount(
                    project_id=project,
                    credentials_path=creds_path,
                    account_group=0,
                    is_valid=True,
                )
            )
        else:
            # Try using default_accounts parameter only as last resort
            if default_accounts:
                logger.warning(
                    "No validated accounts or env config found, using provided defaults"
                )
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
                with p.open("r") as f:
                    cred_data = json.load(f)
                # Check required fields
                required_fields = [
                    "type",
                    "project_id",
                    "private_key_id",
                    "private_key",
                    "client_email",
                ]
                missing = [f for f in required_fields if f not in cred_data]
                if missing:
                    logger.warning(
                        "Invalid credentials file for %s (missing: %s)",
                        acc.project_id,
                        missing,
                    )
                    acc.is_valid = False
                    continue
                if cred_data.get("type") != "service_account":
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
        ) from None
    with _VALIDATED_LOCK:
        _validated_accounts = valid_accounts
        return valid_accounts


def save_validated_accounts(
    accounts: list[VertexAccount], output_file: str = "validated_accounts.json"
) -> None:
    """Persist validated accounts list (merged)."""

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
        or "us-central1"
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
        if service_account_path and service_account:
            cp = Path(service_account_path)
            if not cp.is_absolute():
                cp = Path(__file__).resolve().parent.parent / service_account_path
            if cp.exists():
                credentials = service_account.Credentials.from_service_account_file(
                    str(cp)
                )
                if vertexai:
                    try:
                        vertexai.init(
                            project=project, location=location, credentials=credentials
                        )
                    except Exception as init_err:
                        logger.error("Failed to initialize Vertex AI: %s", init_err)
                        raise
                logger.info("✔ Vertex AI initialized with service account: %s", cp.name)
            else:
                logger.warning(
                    "GOOGLE_APPLICATION_CREDENTIALS points to %s but file not found, using ADC.",
                    service_account_path,
                )
                if vertexai:
                    try:
                        vertexai.init(project=project, location=location)
                    except Exception as init_err:
                        logger.error(
                            "Failed to initialize Vertex AI with ADC: %s", init_err
                        )
                        raise
        else:
            if vertexai:
                try:
                    vertexai.init(project=project, location=location)
                except Exception as init_err:
                    logger.error(
                        "Failed to initialize Vertex AI with default credentials: %s",
                        init_err,
                    )
                    raise
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
        logger.error("Account validation failed for %s: %s", account.project_id, e)
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
    """
    P0-1 FIX: Load projects without hardcoded fallback.
    P0-3 FIX: Thread-safe double-checked locking pattern.
    """
    # Fast path without lock
    if _PROJECT_ROTATION["_initialized"]:
        return

    with _PROJECT_ROTATION_LOCK:
        # Double-check after acquiring lock
        if _PROJECT_ROTATION["_initialized"]:
            return

        try:
            # P0-1 FIX: No hardcoded DEFAULT_ACCOUNTS fallback
            accounts = load_validated_accounts(default_accounts=None)
            _PROJECT_ROTATION["projects"] = [
                {"project_id": a.project_id, "credentials_path": a.credentials_path}
                for a in accounts
            ]
            logger.info(
                "Loaded %d projects for rotation", len(_PROJECT_ROTATION["projects"])
            )
        except LLMError as e:
            # P0-1 FIX: Fail fast instead of silently using empty list
            logger.error("Failed to load any validated accounts: %s", e)
            raise LLMError(
                "Cannot initialize project rotation: no valid GCP accounts. "
                "Create validated_accounts.json or set GCP_PROJECT + GOOGLE_APPLICATION_CREDENTIALS"
            ) from e

        _PROJECT_ROTATION["_initialized"] = True


def _rotate_to_next_project() -> str:
    _ensure_projects_loaded()
    with _PROJECT_ROTATION_LOCK:
        if not _PROJECT_ROTATION["projects"]:
            logger.warning(
                "No projects available for rotation, staying on current env."
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
            _PROJECT_ROTATION.get("consecutive_errors", 0),
        )
        logger.warning("🔄 Rotating to project: %s", conf["project_id"])
        return conf["project_id"]


# --------------------------------------------------------------------------------------
# Retry / transient error classification (kept as-is)
# --------------------------------------------------------------------------------------

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

    name = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")
    if not GenerativeModel:
        raise ImportError(
            "Vertex AI GenerativeModel is not available. Please install google-cloud-aiplatform."
        )
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
    # P2-7 FIX: This function is redundant, _norm_vertex_model_name in indexing_metadata.py should be used
    if name is None:
        return None
    return {"gemini-embedding-001": "gemini-embedding-001"}.get(name, name)


# --------------------------------------------------------------------------------------
# Text completion (Vertex)
# --------------------------------------------------------------------------------------
@monitor_performance
@with_retry(
    max_attempts=int(os.getenv("VERTEX_MAX_RETRIES", "5")),
    base_delay=float(os.getenv("VERTEX_BACKOFF_INITIAL", "4")),
    max_delay=float(os.getenv("VERTEX_BACKOFF_MAX", "60")),
    should_retry=_is_retryable_error,
)
@circuit_breaker(failure_threshold=5, timeout=60, error_types=(LLMError, Exception))
def complete_text(
    system: str,
    user: str,
    max_output_tokens: int = 1200,
    temperature: float = 0.2,
    stop_sequences: list[str] | None = None,
) -> str:
    """Vertex Gemini text completion with retry + project rotation."""
    try:
        _init_vertex()
    except Exception as e:
        logger.error("Failed to initialize Vertex AI: %s", e)
        raise LLMError(f"Vertex AI initialization failed: {e}") from e

    model = _vertex_model(system_instruction=system)

    # Disable safety filters for business use
    safety_settings = {}
    if HarmCategory and HarmBlockThreshold:
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

    if not GenerationConfig:
        raise ImportError(
            "Vertex AI GenerationConfig is not available. Please install google-cloud-aiplatform."
        )

    cfg = GenerationConfig(
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        stop_sequences=stop_sequences or None,
    )
    _check_rate_limit()

    try:
        resp = model.generate_content(
            user, generation_config=cfg, safety_settings=safety_settings
        )
    except Exception as e:
        if _should_rotate_on(e):
            _rotate_to_next_project()
        logger.exception("Vertex generate_content failed: %s", e)
        raise LLMError(f"Vertex API call failed: {e}") from e

    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        raise LLMError("Empty completion from model")
    return text


def _extract_json_from_text(s: str) -> str:
    """Best-effort salvage of JSON object or array from arbitrary text (or fenced code)."""
    if not s:
        return "{}"
    s = s.strip()
    # Prefer fenced blocks if present; allow ```json, ```json5, ```hjson
    fence = re.search(
        r"```(?:json|json5|hjson)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE
    )
    candidate = fence.group(1).strip() if fence else s
    # Extract either an object {…} or an array […]; return {} if nothing matches
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", candidate)
    return m.group(1).strip() if m else "{}"


@monitor_performance
@with_retry(
    max_attempts=int(os.getenv("VERTEX_MAX_RETRIES", "5")),
    base_delay=float(os.getenv("VERTEX_BACKOFF_INITIAL", "4")),
    max_delay=float(os.getenv("VERTEX_BACKOFF_MAX", "60")),
    should_retry=_is_retryable_error,
)
@circuit_breaker(failure_threshold=5, timeout=60, error_types=(LLMError, Exception))
def complete_json(
    system: str,
    user: str,
    max_output_tokens: int = 1600,
    temperature: float = 0.2,
    response_schema: dict | None = None,
    stop_sequences: list[str] | None = None,
) -> str:
    """
    JSON-mode completion with robust fallback to text+regex JSON salvage.

    Args:
        system: The system instruction or context for the model.
        user: The user's prompt.
        max_output_tokens: The maximum number of tokens to generate.
        temperature: The sampling temperature for generation.
        response_schema: An optional JSON schema to enforce on the output.
        stop_sequences: Optional list of strings to stop generation.
            **Warning**: Due to Vertex AI API limitations, this is only
            honored in the text-mode fallback and may be ignored during
            native JSON-mode generation.
    """
    try:
        _init_vertex()
    except Exception as e:
        logger.error("Failed to initialize Vertex AI: %s", e)
        raise LLMError(f"Vertex AI initialization failed: {e}") from e

    model = _vertex_model(system_instruction=system)
    # Disable safety filters for business use
    safety_settings = {}
    if HarmCategory and HarmBlockThreshold:
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

    if not GenerationConfig:
        raise ImportError(
            "Vertex AI GenerationConfig is not available. Please install google-cloud-aiplatform."
        )

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

    try:
        resp = model.generate_content(
            user,
            generation_config=GenerationConfig(**cfg),
            safety_settings=safety_settings,
        )
        return (getattr(resp, "text", None) or "").strip()
    except Exception as e:
        if _should_rotate_on(e):
            _rotate_to_next_project()
        logger.warning("JSON completion failed (%s), falling back to text mode", e)
        # HIGH #40: Preserve response_schema constraint in fallback by adding to system prompt
        enhanced_system = system
        if response_schema:
            schema_description = json.dumps(response_schema, indent=2)
            enhanced_system += f"\n\nIMPORTANT: Your response must conform to this JSON schema:\n{schema_description}"
        out = complete_text(
            enhanced_system,
            user + "\n\nOutput valid JSON matching the required schema.",
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )
        return _extract_json_from_text(out)


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

    Raises:
        LLMError: If vectors list is empty (indicates upstream failure)

    Note:
        Adds small epsilon (1e-12) to avoid division by zero
    """
    # Issue #13 FIX: Don't silently return empty array - indicates upstream failure
    if not vectors:
        raise LLMError(
            "Cannot normalize empty vector list. Check embedding provider returned results.",
            error_code="EMPTY_EMBEDDING_VECTORS",
        )
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

    Args:
        texts: Iterable of text strings to embed
        provider: Embedding provider name
        model: Optional model override

    Returns:
        NumPy array of shape (N, D) with unit-normalized embeddings

    Raises:
        LLMError: If texts is empty, provider fails, or embeddings invalid
    """
    seq = list(texts)
    provider_str = provider or os.getenv("EMBED_PROVIDER", "vertex")
    provider = provider_str.lower() if provider_str else "vertex"

    # Issue #13 FIX: Reject empty input instead of returning (0,0) array
    if not seq:
        raise LLMError(
            "Cannot embed empty text list. Provide at least one text string.",
            error_code="EMPTY_EMBEDDING_INPUT",
        )

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
        raise LLMError(f"Embedding array must be 2-D, got shape {arr.shape}")
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
        raise LLMError("Embeddings appear degenerate (near-zero norms), aborting")
    return arr


def _embed_vertex(texts: list[str], model: str | None = None) -> np.ndarray:
    """Vertex embeddings with rotation on quota exhaustion (google-genai path + legacy fallback)."""
    from .indexing_metadata import _norm_vertex_model_name

    _init_vertex()
    _ensure_projects_loaded()

    embed_name = _norm_vertex_model_name(model) or os.getenv(
        "VERTEX_EMBED_MODEL", "gemini-embedding-001"
    )
    # Path 1: google-genai embed API (recommended for gemini-embedding*)
    if embed_name and embed_name.startswith(("gemini-embedding", "gemini-embedder")):
        if not genai:
            raise LLMError("google-genai not installed") from None

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
                    location = os.getenv("GCP_REGION", "us-central1")
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
                            reqs = [{"content": t} for t in chunk]
                            _check_rate_limit()
                            resp = batch_fn(
                                model=embed_name, requests=reqs, config=cfg
                            )  # type: ignore
                            items = (
                                getattr(resp, "embeddings", None)
                                or getattr(resp, "data", None)
                                or []
                            )
                            for item in items:
                                vals = None
                                if hasattr(item, "values"):
                                    vals = item.values
                                elif hasattr(item, "embedding") and hasattr(
                                    item.embedding, "values"
                                ):
                                    vals = item.embedding.values
                                if vals is None:
                                    raise RuntimeError(
                                        "Empty embedding values in batch response"
                                    ) from None
                                embeddings_batch.append(vals)  # type: ignore[arg-type]
                        except Exception:
                            embeddings_batch = []  # fall through to per-item path
                    if not embeddings_batch:
                        # Fallback: per-item embed_content; handle both 'embedding' and 'embeddings'
                        for _t in chunk:
                            _check_rate_limit()
                            res = client.models.embed_content(
                                model=embed_name or "gemini-embedding-001",
                                contents=_t,
                                config=cfg,
                            )  # type: ignore
                            vals = None
                            if hasattr(res, "values"):
                                vals = getattr(res, "values", None)  # type: ignore[attr-defined]
                            elif hasattr(res, "embedding"):
                                embedding = getattr(res, "embedding", None)  # type: ignore[attr-defined]
                                if embedding and hasattr(embedding, "values"):
                                    vals = getattr(embedding, "values", None)  # type: ignore[attr-defined]
                            elif hasattr(res, "embeddings"):
                                embeddings = getattr(res, "embeddings", None)  # type: ignore[attr-defined]
                                if embeddings:
                                    first = embeddings[0]
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
                        "Empty/mismatched embedding batch from %s, rotating", project
                    )
                    with _PROJECT_ROTATION_LOCK:
                        _PROJECT_ROTATION["consecutive_errors"] = (
                            _PROJECT_ROTATION.get("consecutive_errors", 0) + 1
                        )
                    _rotate_to_next_project()
                    _init_vertex()
                except Exception as e:
                    # CRITICAL FIX #6: Thread-safe access to _PROJECT_ROTATION
                    with _PROJECT_ROTATION_LOCK:
                        has_projects = bool(_PROJECT_ROTATION["projects"])
                    if _should_rotate_on(e) and has_projects:
                        logger.warning("Quota on batch, rotating project")
                        with _PROJECT_ROTATION_LOCK:
                            _PROJECT_ROTATION["consecutive_errors"] = (
                                _PROJECT_ROTATION.get("consecutive_errors", 0) + 1
                            )
                        _rotate_to_next_project()
                        _init_vertex()
                        continue
                    logger.exception("Vertex (gemini) embedding failed: %s", e)
                    break
            if not embedded:
                # Exhausted rotations (if any). Do not substitute zero vectors.
                raise LLMError(
                    f"Vertex (gemini) embedding failed for batch starting at index {i}; exhausted rotation attempts."
                ) from None
        return _normalize(vectors)

    # Path 2: legacy vertexai TextEmbeddingModel
    vectors_legacy: list[list[float]] = []
    B = min(int(os.getenv("EMBED_BATCH", "64")), 250)
    for i in range(0, len(texts), B):
        chunk = texts[i : i + B]
        success = False
        # CRITICAL FIX #6: Thread-safe access to _PROJECT_ROTATION
        with _PROJECT_ROTATION_LOCK:
            max_retries = max(1, len(_PROJECT_ROTATION["projects"]) or 1)
        for _ in range(max_retries):
            try:
                if not TextEmbeddingModel or not TextEmbeddingInput:
                    raise ImportError("vertexai.language_models not available")
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
                for emb in embs:
                    vals = getattr(emb, "values", None)
                    if vals is None:
                        raise RuntimeError(
                            "Vertex (legacy) returned empty embedding values"
                        ) from None
                    batch_vals.append(vals)
                vectors_legacy.extend(batch_vals)
                success = True
                break
            except Exception as err:
                # CRITICAL FIX #6: Thread-safe access to _PROJECT_ROTATION
                with _PROJECT_ROTATION_LOCK:
                    has_projects = bool(_PROJECT_ROTATION["projects"])
                if _should_rotate_on(err) and has_projects:
                    _rotate_to_next_project()
                    _init_vertex()
                    continue
                logger.exception(
                    "Vertex (legacy) embedding failed on batch %d:%d: %s", i, i + B, err
                )
                break
        if not success:
            # Exhausted rotations (if any). Do not substitute zero vectors.
            raise LLMError(
                f"Vertex (legacy) embedding failed for batch {i}:{i + B}; "
                "exhausted rotation attempts."
            ) from None
    return _normalize(vectors_legacy)


def _embed_openai(texts: list[str], model: str | None = None) -> np.ndarray:
    if not OpenAI:
        raise LLMError("Install openai: pip install openai") from None
    cfg = _get_config()
    api_key = cfg.sensitive.openai_api_key
    if not api_key:
        raise LLMError("Set OPENAI_API_KEY") from None
    client = OpenAI(api_key=api_key)
    model_name = model or cfg.embedding.openai_embed_model
    vectors: list[list[float]] = []
    B = int(os.getenv("EMBED_BATCH", "64"))
    for i in range(0, len(texts), B):
        chunk = texts[i : i + B]
        try:
            _check_rate_limit()
            resp = client.embeddings.create(
                input=chunk, model=model_name or "text-embedding-3-small"
            )
            for item in resp.data:
                vectors.append(item.embedding)
        except Exception as e:
            logger.exception("OpenAI embedding failed: %s", e)
            raise LLMError(f"OpenAI embedding failed: {e}") from e
    return _normalize(vectors)


def _embed_azure_openai(texts: list[str], model: str | None = None) -> np.ndarray:
    if not AzureOpenAI:
        raise LLMError("Install openai: pip install openai") from None
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = model or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not all([api_key, endpoint, deployment]):
        raise LLMError(
            "Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT"
        ) from None
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
    if not cohere:
        raise LLMError("Install cohere: pip install cohere") from None
    cfg = _get_config()
    api_key = cfg.sensitive.cohere_api_key
    if not api_key:
        raise LLMError("Set COHERE_API_KEY") from None
    co = cohere.Client(api_key)
    model_name = model or cfg.embedding.cohere_embed_model
    input_type = cfg.embedding.cohere_input_type
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
    if not InferenceClient:
        raise LLMError("Install huggingface_hub: pip install huggingface_hub") from None
    cfg = _get_config()
    api_key = cfg.sensitive.hf_api_key or cfg.sensitive.huggingface_api_key
    if not api_key:
        raise LLMError("Set HF_API_KEY or HUGGINGFACE_API_KEY") from None
    model_name = model or cfg.embedding.hf_embed_model
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
    cfg = _get_config()
    api_key = cfg.sensitive.qwen_api_key
    base_url = cfg.sensitive.qwen_base_url
    model_name = model or cfg.embedding.qwen_embed_model
    if not all([api_key, base_url]):
        raise LLMError("Set QWEN_API_KEY and QWEN_BASE_URL")
    endpoint = (base_url.rstrip("/") if base_url else "") + "/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    B = cfg.processing.batch_size
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
                timeout=cfg.sensitive.qwen_timeout or 60,
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
            raise LLMError(f"Qwen embedding failed for batch {i}:{i + B}: {e}") from e
    return _normalize(vectors)


def _embed_local(texts: list[str], model: str | None = None) -> np.ndarray:
    if not SentenceTransformer:
        raise LLMError(
            "Install sentence-transformers to use local embeddings"
        ) from None
    cfg = _get_config()
    name = model or cfg.embedding.local_embed_model
    st = SentenceTransformer(name)
    arr = st.encode(texts, normalize_embeddings=True)
    return np.array(arr, dtype="float32")


# --------------------------------------------------------------------------------------
# Convenience re-exports from utils so callers can import from one place
# --------------------------------------------------------------------------------------
# Re-export commonly used utilities for single import surface
__all__ = [
    # Configuration
    "EmailOpsConfig",
    # Error types
    "LLMError",
    # Account management
    "VertexAccount",
    "clean_email_text",
    "complete_json",
    # LLM functions
    "complete_text",
    "embed_texts",
    "get_config",
    "load_validated_accounts",
    # Utility re-exports from utils.py
    "logger",
    "monitor_performance",
    "save_validated_accounts",
    "validate_account",
]

# Note: logger and clean_email_text are already available
# from the imports at the top of the file
