#!/usr/bin/env python3
"""
Index metadata management for EmailOps (multi‑provider; Vertex‑optimized by default).

Final patched drop-in replacement.

Key improvements over the previous refactor:
- Robust model-name handling: recognizes fully-qualified Vertex model resource names by
  inspecting the last path segment; expands model coverage (e.g., multilingual).
- Safer 'actual' dimension inference: never falls back to configured/metadata dims as
  a proxy for actual dims (prevents false positives).
- Consistency checks now validate FAISS vector count (ntotal) vs. mapping size and
  vs. embeddings row count when available.
- More robust atomic JSON writes:
    * Unique temp files (safe under concurrency)
    * fsync + retrying os.replace for Windows file-lock scenarios.
- Stronger memmap cleanup on Windows/NFS (closes both base and _mmap when present).
- Accepts common provider aliases ('vertex', 'vertexai', 'google-vertex', etc.).
- get_index_info improvements: clearer "unknown" display and optional FAISS vector count.
- Minor nits fixed (printing of configured dimensions when None, etc.).

Public API preserved:
- create_index_metadata
- save_index_metadata
- load_index_metadata
- validate_index_compatibility
- get_index_info
- index_paths
- read_mapping
- write_mapping
- check_index_consistency
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Atomic write retry configuration
ATOMIC_WRITE_MAX_RETRIES = 6
ATOMIC_WRITE_BASE_DELAY = 0.05
ATOMIC_WRITE_EXPONENTIAL_BASE = 2

# Default half-life for time decay
DEFAULT_HALF_LIFE_DAYS = 30

# -----------------------------------------------------------------------------
# Filenames & Paths (single source of truth)
# -----------------------------------------------------------------------------
META_FILENAME = "meta.json"
FAISS_INDEX_FILENAME = "index.faiss"
MAPPING_FILENAME = "mapping.json"
EMBEDDINGS_FILENAME = "embeddings.npy"
TIMESTAMP_FILENAME = "last_run.txt"
FILE_TIMES_FILENAME = "file_times.json"
INDEX_DIRNAME_DEFAULT = os.getenv("INDEX_DIRNAME", "_index")

__all__ = [
    "META_FILENAME",
    "FAISS_INDEX_FILENAME",
    "MAPPING_FILENAME",
    "EMBEDDINGS_FILENAME",
    "TIMESTAMP_FILENAME",
    "FILE_TIMES_FILENAME",
    "INDEX_DIRNAME_DEFAULT",
    "create_index_metadata",
    "save_index_metadata",
    "load_index_metadata",
    "validate_index_compatibility",
    "get_index_info",
    "index_paths",
    "read_mapping",
    "write_mapping",
    "check_index_consistency",
]

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class IndexPaths:
    base: Path
    meta: Path
    mapping: Path
    embeddings: Path
    faiss: Path
    timestamp: Path
    file_times: Path


def index_paths(index_dir: Union[str, Path]) -> IndexPaths:
    index_dir = Path(index_dir).resolve()
    return IndexPaths(
        base=index_dir,
        meta=index_dir / META_FILENAME,
        mapping=index_dir / MAPPING_FILENAME,
        embeddings=index_dir / EMBEDDINGS_FILENAME,
        faiss=index_dir / FAISS_INDEX_FILENAME,
        timestamp=index_dir / TIMESTAMP_FILENAME,
        file_times=index_dir / FILE_TIMES_FILENAME,
    )


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _safe_int(val: Optional[str]) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _normalize_provider(provider: str) -> str:
    """
    Normalize common provider aliases to 'vertex'.
    """
    p = (provider or "").strip().lower().replace("-", "").replace(" ", "")
    if p in {"vertex", "vertexai", "googlevertex", "googlevertexai"}:
        return "vertex"
    return p


def _norm_vertex_model_name(raw: Optional[str]) -> str:
    """
    Normalize common Vertex model name variants we want to support for storage.
    Treat 'gemini-embedded-001' as an alias of 'gemini-embedding-001', but keep
    fully-qualified resource names intact (do NOT strip paths here).
    """
    model = (raw or "").strip()
    lower = model.lower()
    if lower == "gemini-embedded-001":
        return "gemini-embedding-001"
    return model


def _vertex_dimensions_for_model(model: str) -> Optional[int]:
    """
    Heuristics for Vertex AI embedding dimensions based on the final segment
    of the model name / resource path:

    - gemini-embedding-*                => default 3072
    - text-embedding-004/005            => default 768
    - textembedding-gecko*              => default 768
    - text-multilingual-embedding-*     => default 768

    If we can't infer, return None (caller may supply explicit override).
    """
    m = (model or "").lower()
    last = m.split("/")[-1]  # support fully-qualified resource names

    if (
        last == "gemini-embedding-001"
        or last.startswith(("gemini-embedding", "gemini-embedder"))
    ):
        return 3072

    if last.startswith(
        (
            "text-embedding-004",
            "text-embedding-005",
            "textembedding-gecko",
            "text-multilingual-embedding",
        )
    ):
        return 768

    return None


def _resolve_vertex_config() -> Dict[str, Any]:
    """
    Read Vertex-specific configuration from environment and apply sensible defaults.
    Honors VERTEX_OUTPUT_DIM (preferred) or VERTEX_EMBED_DIM if set.
    """
    raw_model = os.getenv("VERTEX_EMBED_MODEL", "gemini-embedding-001")
    model = _norm_vertex_model_name(raw_model)

    # Allow the user to pin output dimensionality (Gemini supports this).
    explicit_dim = _safe_int(os.getenv("VERTEX_OUTPUT_DIM")) or _safe_int(
        os.getenv("VERTEX_EMBED_DIM")
    )
    if explicit_dim is not None:
        if explicit_dim <= 0:
            logger.warning("Ignoring non-positive embedding dimension from env: %s", explicit_dim)
            dim = None
        else:
            dim = explicit_dim
    else:
        dim = _vertex_dimensions_for_model(model)

    return {"model": model, "dimensions": dim}


def _close_memmap(arr: Any) -> None:
    """
    Best-effort close of NumPy memmap to release file handles on Windows/NFS.
    """
    if arr is None:
        return
    try:
        base = arr
        while getattr(base, "base", None) is not None:
            base = base.base
        if hasattr(base, "close"):
            base.close()
    except Exception:
        pass
    with contextlib.suppress(Exception):
        if hasattr(arr, "_mmap") and getattr(arr, "_mmap", None) is not None:
            arr._mmap.close()


def _get_all_dimensions(index_dir: Union[str, Path]) -> Dict[str, Optional[int]]:
    """
    Consolidated dimension detection from all available sources.
    Returns dict with keys: 'embeddings', 'faiss', 'detected'
    """
    dims = {
        'embeddings': _detect_actual_dimensions(index_dir),
        'faiss': _detect_faiss_dimensions(index_dir),
        'detected': None
    }
    dims['detected'] = dims['embeddings'] or dims['faiss']
    return dims


def _detect_actual_dimensions(index_dir: Union[str, Path]) -> Optional[int]:
    """
    Detect embedding width from embeddings.npy (2D) without leaving file handles open.
    Uses mmap for speed but attempts to close underlying file to avoid Windows locks.
    """
    p = index_paths(index_dir)
    npy = p.embeddings
    if not npy.exists():
        return None
    try:
        import numpy as np  # type: ignore

        arr = np.load(str(npy), mmap_mode="r")
        if arr.ndim == 2:
            width = int(arr.shape[1])
            logger.debug("Detected %d dimensions from %s", width, npy.name)
        else:
            _close_memmap(arr)
            return None
        _close_memmap(arr)
        del arr
        gc.collect()
        return width
    except Exception as e:
        logger.debug("Could not detect dimensions from %s: %s", npy.name, e)
    return None


def _detect_faiss_dimensions(index_dir: Union[str, Path]) -> Optional[int]:
    p = index_paths(index_dir)
    fi = p.faiss
    if not fi.exists():
        return None
    try:
        import faiss  # type: ignore

        idx = faiss.read_index(str(fi))
        d = int(getattr(idx, "d", None) or idx.d)
        logger.debug("Detected %d dimensions from FAISS index", d)
        del idx
        return d
    except Exception as e:
        logger.debug("Could not detect dimensions from %s: %s", fi.name, e)
        return None


def _detect_faiss_count(index_dir: Union[str, Path]) -> Optional[int]:
    """
    Detect the number of vectors stored in a FAISS index (ntotal).
    """
    p = index_paths(index_dir)
    fi = p.faiss
    if not fi.exists():
        return None
    try:
        import faiss  # type: ignore

        idx = faiss.read_index(str(fi))
        n = int(getattr(idx, "ntotal", 0))
        del idx
        return n
    except Exception as e:
        logger.debug(f"Could not read FAISS index to infer count: {e}")
        return None


def _detect_index_type(index_dir: Union[str, Path]) -> str:
    p = index_paths(index_dir)
    if p.faiss.exists():
        return "faiss"
    if p.embeddings.exists():
        return "numpy"
    return "none"


def _atomic_write_json(path: Path, data: Dict[str, Any] | List[Any]) -> None:
    """
    Cross-platform, concurrency-friendly atomic write:
      - write JSON to a unique temp file in the same directory,
      - flush and fsync,
      - atomically replace the target with retries (for Windows file locks).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)

    tmp_name = None
    last_exc: Optional[BaseException] = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as f:
            tmp_name = f.name
            f.write(text)
            f.flush()
            os.fsync(f.fileno())

        # Retry os.replace a few times to handle transient locks (Windows)
        success = False
        for attempt in range(ATOMIC_WRITE_MAX_RETRIES):
            try:
                os.replace(tmp_name, path)
                success = True
                tmp_name = None  # consumed
                break
            except (PermissionError, OSError) as e:
                last_exc = e
                time.sleep(ATOMIC_WRITE_BASE_DELAY * (ATOMIC_WRITE_EXPONENTIAL_BASE**attempt))
        if not success:
            assert last_exc is not None
            raise last_exc
    finally:
        if tmp_name and os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def create_index_metadata(
    provider: str,
    num_documents: int,
    num_folders: int,
    index_dir: Union[str, Path],
    custom_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    provider_norm = _normalize_provider(provider)
    is_vertex = provider_norm == "vertex"
    # Vertex-specific config; non-Vertex providers do not use hardcoded dims.
    model_info = _resolve_vertex_config() if is_vertex else {}

    # Determine index type and detect actual dimension (custom_metadata can override)
    index_type = _detect_index_type(index_dir)
    actual_dim: Optional[int] = None
    detected_dim = _detect_actual_dimensions(index_dir) or _detect_faiss_dimensions(index_dir)
    if custom_metadata and "actual_dimensions" in custom_metadata:
        actual_dim = custom_metadata.get("actual_dimensions")
        if detected_dim is not None and actual_dim != detected_dim:
            logger.warning(
                "custom_metadata.actual_dimensions (%s) disagrees with on-disk dimensions (%s)",
                actual_dim,
                detected_dim,
            )
    if actual_dim is None:
        actual_dim = detected_dim

    half_life_env = _safe_int(os.getenv("HALF_LIFE_DAYS"))
    half_life = DEFAULT_HALF_LIFE_DAYS if (half_life_env is None or half_life_env < 1) else half_life_env

    metadata: Dict[str, Any] = {
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        # Store exactly what the caller passed; keep a normalized copy for internal checks.
        "provider": provider,
        "provider_norm": provider_norm,
        # Vertex: prefer configured model/dims; others: advisory fields only (may be None/"unknown")
        "model": (
            model_info.get("model")
            if is_vertex
            else ((custom_metadata or {}).get("model") or "unknown")
        ),
        "dimensions": (
            model_info.get("dimensions") if is_vertex else (custom_metadata or {}).get("dimensions")
        ),
        "actual_dimensions": actual_dim,
        "num_documents": int(num_documents),
        "num_folders": int(num_folders),
        "index_type": index_type,
        "half_life_days": half_life,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }

    # Prevent clobbering computed/reserved fields (but allow explicit actual_dimensions override)
    RESERVED = {
        "version",
        "created_at",
        "provider",
        "provider_norm",
        "model",
        "dimensions",
        "actual_dimensions",
        "num_documents",
        "num_folders",
        "index_type",
        "half_life_days",
        "python_version",
    }
    if custom_metadata:
        for k, v in custom_metadata.items():
            if k in RESERVED:
                continue
            metadata[k] = v
        if "actual_dimensions" in custom_metadata:
            metadata["actual_dimensions"] = custom_metadata["actual_dimensions"]
        # For non-Vertex providers, allow custom model/dimensions to be recorded as advisory.
        if not is_vertex:
            if "model" in custom_metadata:
                metadata["model"] = custom_metadata["model"]
            if "dimensions" in custom_metadata:
                metadata["dimensions"] = custom_metadata["dimensions"]

    return metadata


def save_index_metadata(metadata: Dict[str, Any], index_dir: Union[str, Path]) -> None:
    p = index_paths(index_dir).meta
    _atomic_write_json(p, metadata)
    logger.info(f"Saved index metadata to {p}")


def load_index_metadata(index_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    p = index_paths(index_dir).meta
    if not p.exists():
        logger.warning(f"No metadata found at {p}")
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            logger.error(f"Metadata at {p} must be a JSON object; got {type(data).__name__}")
            return None
        return data
    except Exception as e:
        logger.error(f"Failed to load metadata from {p}: {e}")
        return None


def check_index_consistency(index_dir: Union[str, Path], raise_on_mismatch: bool = True) -> bool:
    """
    Ensure mapping.json entries == embeddings.npy rows (when present)
    and/or == FAISS ntotal (when present).
    Intended to be called immediately after a build/save step so drift is caught early.
    Returns True on success; raises or returns False on mismatch.
    """
    p = index_paths(index_dir)
    n_map: Optional[int] = None
    n_rows: Optional[int] = None
    n_index: Optional[int] = None

    # Read mapping strictly
    if p.mapping.exists():
        try:
            mapping = read_mapping(index_dir, strict=True)
            n_map = len(mapping)
        except Exception as e:
            if raise_on_mismatch:
                raise
            logger.error(f"Failed to read mapping.json for consistency check: {e}")
            return False

    # Read embeddings row count if present
    if p.embeddings.exists():
        try:
            import numpy as np  # type: ignore

            arr = np.load(str(p.embeddings), mmap_mode="r")
            if arr.ndim != 2:
                msg = f"Embeddings array must be 2D; got shape {arr.shape}"
                _close_memmap(arr)
                if raise_on_mismatch:
                    raise ValueError(msg)
                logger.error(msg)
                return False
            n_rows = int(arr.shape[0])
            _close_memmap(arr)
            del arr
            gc.collect()
        except Exception as e:
            if raise_on_mismatch:
                raise
            logger.error(f"Failed to read embeddings for consistency check: {e}")
            return False

    # Read FAISS vector count if present
    if p.faiss.exists():
        try:
            n_index = _detect_faiss_count(index_dir)
        except Exception as e:
            if raise_on_mismatch:
                raise
            logger.error(f"Failed to read FAISS index for consistency check: {e}")
            return False

    # Cross-validate counts
    def _fail(msg: str) -> bool:
        if raise_on_mismatch:
            raise ValueError(msg)
        logger.error(msg)
        return False

    if n_map is not None and n_rows is not None and n_map != n_rows:
        return _fail(f"mapping.json entries ({n_map}) != embeddings rows ({n_rows})")

    if n_map is not None and n_index is not None and n_map != n_index:
        return _fail(f"mapping.json entries ({n_map}) != FAISS vectors ({n_index})")

    if n_rows is not None and n_index is not None and n_rows != n_index:
        return _fail(f"embeddings rows ({n_rows}) != FAISS vectors ({n_index})")

    return True


def _validate_metadata(metadata: Optional[Dict[str, Any]], provider_norm: str, raise_on_mismatch: bool) -> bool:
    """Validate metadata existence and provider match."""
    if metadata is None:
        msg = "No index metadata found; refusing to proceed."
        if raise_on_mismatch:
            raise FileNotFoundError(msg)
        logger.error(msg)
        return False

    # Compare normalized providers
    indexed_provider = _normalize_provider(metadata.get("provider", "") or metadata.get("provider_norm", ""))
    if indexed_provider and indexed_provider != provider_norm:
        msg = f"Index was created with provider '{indexed_provider}' but trying to search with '{provider_norm}'."
        if raise_on_mismatch:
            raise ValueError(msg)
        logger.error(msg)
        return False
    
    return True


def _validate_artifacts(p: IndexPaths, raise_on_mismatch: bool) -> bool:
    """Validate that at least one index artifact exists."""
    if not p.faiss.exists() and not p.embeddings.exists():
        msg = "No index artifacts found (missing index.faiss and embeddings.npy)."
        if raise_on_mismatch:
            raise FileNotFoundError(msg)
        logger.error(msg)
        return False
    return True


def _validate_dimensions(
    index_dir: Union[str, Path],
    metadata: Dict[str, Any],
    is_vertex: bool,
    raise_on_mismatch: bool
) -> bool:
    """Validate dimension compatibility."""
    dims = _get_all_dimensions(index_dir)
    detected_dims = dims['detected']
    meta_actual = metadata.get("actual_dimensions")
    
    if is_vertex:
        # Vertex-specific: enforce configured output dimension if both sides are known.
        expected_cfg = _resolve_vertex_config()
        expected_dims = expected_cfg.get("dimensions")
        actual_dims = detected_dims if detected_dims is not None else meta_actual
        
        if meta_actual is not None and detected_dims is not None and meta_actual != detected_dims:
            logger.warning(
                "Metadata actual_dimensions (%s) disagrees with on-disk dimensions (%s).",
                meta_actual,
                detected_dims,
            )
        
        if actual_dims is not None and expected_dims is not None and actual_dims != expected_dims:
            msg = f"Dimension mismatch: index has {actual_dims} dims, but Vertex config is {expected_dims}."
            if raise_on_mismatch:
                raise ValueError(msg)
            logger.error(msg)
            return False
    else:
        # Non-Vertex: only enforce that on-disk dims match metadata.actual_dimensions
        if detected_dims is not None and meta_actual is not None and detected_dims != meta_actual:
            msg = f"Dimension drift: on-disk dims ({detected_dims}) != metadata.actual_dimensions ({meta_actual})."
            if raise_on_mismatch:
                raise ValueError(msg)
            logger.error(msg)
            return False
    
    return True


def validate_index_compatibility(
    index_dir: Union[str, Path],
    provider: str,
    raise_on_mismatch: bool = True,
    check_counts: bool = True,
) -> bool:
    """
    Validate index compatibility with the given provider.
    Refactored for clarity with helper functions.
    """
    provider_norm = _normalize_provider(provider)
    is_vertex = provider_norm == "vertex"

    # Load and validate metadata
    metadata = load_index_metadata(index_dir)
    if not _validate_metadata(metadata, provider_norm, raise_on_mismatch):
        return False

    # Validate artifacts exist
    p = index_paths(index_dir)
    if not _validate_artifacts(p, raise_on_mismatch):
        return False

    # Validate dimensions (metadata is guaranteed to be non-None after validation)
    assert metadata is not None  # For type checker
    if not _validate_dimensions(index_dir, metadata, is_vertex, raise_on_mismatch):
        return False

    # Sanity: index_type vs. files present
    meta_index_type = (metadata.get("index_type") or "unknown").lower()
    detected_type = _detect_index_type(index_dir)
    if meta_index_type != detected_type:
        logger.warning(
            f"Index type in metadata ('{meta_index_type}') does not match files present ('{detected_type}')."
        )

    # If both FAISS and NPY exist, warn about potential drift
    if p.faiss.exists() and p.embeddings.exists():
        logger.warning(
            "Both FAISS index and embeddings.npy detected; ensure they were built in the same run."
        )

    if check_counts:
        return check_index_consistency(index_dir, raise_on_mismatch=raise_on_mismatch)

    return True


def _gather_index_counts(index_dir: Union[str, Path]) -> Dict[str, Any]:
    """Gather all index counts and dimensions."""
    info: Dict[str, Any] = {
        'mapping_count': None,
        'mapping_error': None,
        'embeddings_count': None,
        'faiss_count': None,
        'inferred_dim': None
    }
    
    p = index_paths(index_dir)
    
    # mapping.json
    if p.mapping.exists():
        try:
            mapping = read_mapping(index_dir, strict=True)
            if isinstance(mapping, list):
                info['mapping_count'] = len(mapping)
        except Exception as e:
            info['mapping_error'] = str(e)
    
    # embeddings.npy
    if p.embeddings.exists():
        try:
            import numpy as np  # type: ignore
            arr = np.load(str(p.embeddings), mmap_mode="r")
            if arr.ndim == 2:
                info['embeddings_count'] = int(arr.shape[0])
                info['inferred_dim'] = int(arr.shape[1])
            _close_memmap(arr)
            del arr
            gc.collect()
        except Exception:
            pass
    
    # FAISS index
    if p.faiss.exists():
        try:
            info['faiss_count'] = _detect_faiss_count(index_dir)
            if info['inferred_dim'] is None:
                info['inferred_dim'] = _detect_faiss_dimensions(index_dir)
        except Exception:
            pass
    
    return info


def get_index_info(index_dir: Union[str, Path]) -> str:
    """
    Get formatted index information.
    Refactored for clarity with helper function.
    """
    metadata = load_index_metadata(index_dir)
    if metadata is None:
        return "No index metadata available"

    # Gather all counts and dimensions
    counts = _gather_index_counts(index_dir)
    
    actual = metadata.get("actual_dimensions") or counts['inferred_dim']
    configured_dims = metadata.get("dimensions")
    configured_dims_str = str(configured_dims) if configured_dims is not None else "unknown"

    # Build output lines
    lines = [
        "Index Information:",
        f"  Provider: {metadata.get('provider', 'unknown')}",
        f"  Model: {metadata.get('model', 'unknown')}",
        f"  Configured Dimensions: {configured_dims_str}",
    ]
    
    if actual is not None:
        lines.append(f"  Actual Dimensions: {actual}")

    # Document counts
    if counts['mapping_count'] is not None:
        lines.append(f"  Documents (mapping.json): {counts['mapping_count']}")
    elif counts['mapping_error'] is not None:
        lines.append(f"  Documents (mapping.json): error reading ({counts['mapping_error']})")
    else:
        lines.append(f"  Documents (metadata): {metadata.get('num_documents', 'unknown')}")

    # Embeddings and FAISS counts
    if counts['embeddings_count'] is not None:
        lines.append(f"  Embeddings (rows in {EMBEDDINGS_FILENAME}): {counts['embeddings_count']}")

    if counts['faiss_count'] is not None:
        lines.append(f"  FAISS vectors (ntotal): {counts['faiss_count']}")

    # Additional metadata
    lines.extend([
        f"  Folders: {metadata.get('num_folders', 'unknown')}",
        f"  Created: {metadata.get('created_at', 'unknown')}",
        f"  Index Type: {metadata.get('index_type', 'unknown')}",
        f"  Half-life (days): {metadata.get('half_life_days', 'unknown')}",
    ])
    
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Mapping helpers (BOM/JSON tolerant)
# -----------------------------------------------------------------------------

def read_mapping(index_dir: Union[str, Path], strict: bool = True) -> List[Dict[str, Any]]:
    """
    Read mapping.json.

    If strict=True (default), raises on invalid JSON or wrong schema.
    If strict=False, returns [] on any error (legacy behavior).
    """
    p = index_paths(index_dir).mapping
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, list) or (data and not all(isinstance(x, dict) for x in data)):
            raise ValueError("mapping.json must be a list of objects")
        return data  # type: ignore[return-value]
    except Exception as e:
        if strict:
            raise
        logger.warning(f"Failed to read {p.name}: {e}")
        return []


def write_mapping(index_dir: Union[str, Path], mapping: List[Dict[str, Any]]) -> Path:
    p = index_paths(index_dir).mapping
    _atomic_write_json(p, mapping)
    return p
