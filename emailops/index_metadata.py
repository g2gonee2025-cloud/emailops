#!/usr/bin/env python3
"""
Index metadata management for EmailOps (Vertex-only, Gemini-optimized).

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
    try:
        base = arr
        while getattr(base, "base", None) is not None:
            base = base.base  # unwrap views
        if hasattr(base, "close"):
            base.close()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        # numpy.memmap frequently exposes the underlying mmap handle here
        if hasattr(arr, "_mmap") and getattr(arr, "_mmap", None) is not None:
            arr._mmap.close()  # type: ignore[attr-defined]
    except Exception:
        pass


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
        else:
            _close_memmap(arr)
            return None
        _close_memmap(arr)
        del arr
        gc.collect()
        return width
    except Exception as e:
        logger.debug(f"Could not read {EMBEDDINGS_FILENAME} to infer dimensions: {e}")
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
        del idx
        return d
    except Exception as e:
        logger.debug(f"Could not read FAISS index to infer dimensions: {e}")
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
        for attempt in range(6):
            try:
                os.replace(tmp_name, path)
                success = True
                tmp_name = None  # consumed
                break
            except (PermissionError, OSError) as e:
                last_exc = e
                time.sleep(0.05 * (2**attempt))
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
    if provider_norm != "vertex":
        raise ValueError(
            f"Only 'vertex' provider is supported in this build; got '{provider}'."
        )

    model_info = _resolve_vertex_config()

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
    half_life = 30 if (half_life_env is None or half_life_env < 1) else half_life_env

    metadata: Dict[str, Any] = {
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider_norm,
        "model": model_info["model"],  # keep as provided (may be a full resource name)
        "dimensions": model_info.get("dimensions"),
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


def validate_index_compatibility(
    index_dir: Union[str, Path],
    provider: str,
    raise_on_mismatch: bool = True,
    check_counts: bool = True,
) -> bool:
    provider_norm = _normalize_provider(provider)
    if provider_norm != "vertex":
        msg = f"Only 'vertex' provider is supported; got '{provider}'."
        if raise_on_mismatch:
            raise ValueError(msg)
        logger.error(msg)
        return False

    metadata = load_index_metadata(index_dir)
    if metadata is None:
        msg = "No index metadata found; refusing to proceed."
        if raise_on_mismatch:
            raise FileNotFoundError(msg)
        logger.error(msg)
        return False

    indexed_provider = _normalize_provider(metadata.get("provider", ""))
    if indexed_provider and indexed_provider != provider_norm:
        msg = (
            f"Index was created with provider '{indexed_provider}' but trying to search with '{provider_norm}'."
        )
        if raise_on_mismatch:
            raise ValueError(msg)
        logger.error(msg)
        return False

    # Ensure at least one index artifact exists
    p = index_paths(index_dir)
    if not p.faiss.exists() and not p.embeddings.exists():
        msg = "No index artifacts found (missing index.faiss and embeddings.npy)."
        if raise_on_mismatch:
            raise FileNotFoundError(msg)
        logger.error(msg)
        return False

    # Compare dimensions
    expected_cfg = _resolve_vertex_config()
    expected_dims = expected_cfg.get("dimensions")

    # Prefer on-disk detection; fall back to metadata.actual_dimensions only if needed.
    detected_dims = _detect_actual_dimensions(index_dir) or _detect_faiss_dimensions(index_dir)
    meta_actual = metadata.get("actual_dimensions")
    if detected_dims is None:
        actual_dims = meta_actual
    else:
        actual_dims = detected_dims
        if meta_actual is not None and meta_actual != detected_dims:
            logger.warning(
                "Metadata actual_dimensions (%s) disagrees with on-disk dimensions (%s).",
                meta_actual,
                detected_dims,
            )

    if actual_dims is not None and expected_dims is not None and actual_dims != expected_dims:
        msg = (
            f"Dimension mismatch: index has {actual_dims} dims, but Vertex config is {expected_dims}."
        )
        if raise_on_mismatch:
            raise ValueError(msg)
        logger.error(msg)
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


def get_index_info(index_dir: Union[str, Path]) -> str:
    metadata = load_index_metadata(index_dir)
    if metadata is None:
        return "No index metadata available"

    mapping_count: Optional[int] = None
    embeddings_count: Optional[int] = None
    inferred_dim: Optional[int] = None
    faiss_count: Optional[int] = None

    p = index_paths(index_dir)

    # mapping.json
    mapping_error = None
    if p.mapping.exists():
        try:
            mapping = read_mapping(index_dir, strict=True)
            if isinstance(mapping, list):
                mapping_count = len(mapping)
        except Exception as e:
            mapping_error = str(e)

    # embeddings.npy
    if p.embeddings.exists():
        try:
            import numpy as np  # type: ignore

            arr = np.load(str(p.embeddings), mmap_mode="r")
            if arr.ndim == 2:
                embeddings_count = int(arr.shape[0])
                inferred_dim = int(arr.shape[1])
            _close_memmap(arr)
            del arr
            gc.collect()
        except Exception:
            pass

    # FAISS index
    if p.faiss.exists():
        try:
            faiss_count = _detect_faiss_count(index_dir)
            if inferred_dim is None:
                inferred_dim = _detect_faiss_dimensions(index_dir)
        except Exception:
            pass

    actual = metadata.get("actual_dimensions") or inferred_dim

    configured_dims = metadata.get("dimensions")
    configured_dims_str = (
        str(configured_dims) if configured_dims is not None else "unknown"
    )

    lines = [
        "Index Information:",
        f"  Provider: {metadata.get('provider', 'unknown')}",
        f"  Model: {metadata.get('model', 'unknown')}",
        f"  Configured Dimensions: {configured_dims_str}",
    ]
    if actual is not None:
        lines.append(f"  Actual Dimensions: {actual}")

    if mapping_count is not None:
        lines.append(f"  Documents (mapping.json): {mapping_count}")
    elif mapping_error is not None:
        lines.append(f"  Documents (mapping.json): error reading ({mapping_error})")
    else:
        lines.append(f"  Documents (metadata): {metadata.get('num_documents', 'unknown')}")

    if embeddings_count is not None:
        lines.append(f"  Embeddings (rows in {EMBEDDINGS_FILENAME}): {embeddings_count}")

    if faiss_count is not None:
        lines.append(f"  FAISS vectors (ntotal): {faiss_count}")

    lines.extend(
        [
            f"  Folders: {metadata.get('num_folders', 'unknown')}",
            f"  Created: {metadata.get('created_at', 'unknown')}",
            f"  Index Type: {metadata.get('index_type', 'unknown')}",
            f"  Half-life (days): {metadata.get('half_life_days', 'unknown')}",
        ]
    )
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
