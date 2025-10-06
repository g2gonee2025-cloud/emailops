#!/usr/bin/env python3
"""
Index metadata management for EmailOps.
Handles creation and validation of index metadata files.
"""
from __future__ import annotations

import json
import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Filenames used throughout the index directory (keep in sync with indexer/search)
META_FILENAME = "meta.json"
FAISS_INDEX_FILENAME = "index.faiss"
MAPPING_FILENAME = "mapping.json"
EMBEDDINGS_FILENAME = "embeddings.npy"


def _safe_int(val: Optional[str]) -> Optional[int]:
    """Parse int from env string, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _resolve_provider_config(provider: str) -> Dict[str, Any]:
    """
    Resolve model name and expected dimensions dynamically from environment,
    per provider. Dimensions may be None when unknown (deployment-dependent).
    """
    p = (provider or "").strip().lower()

    # Vertex AI (Google)
    if p == "vertex":
        model = os.getenv("VERTEX_EMBED_MODEL", "gemini-embedding-001")
        # If explicitly set, trust it. Otherwise, infer by model family.
        explicit_dim = _safe_int(os.getenv("VERTEX_EMBED_DIM"))
        if explicit_dim is not None:
            dim = explicit_dim
        else:
            # Gemini embedding models default to 3072; legacy text-embedding-* often 768.
            dim = 3072 if model.startswith(("gemini-embedding", "gemini-embedder")) else 768
        return {"model": model, "dimensions": dim, "reranker": None}

    # OpenAI
    if p == "openai":
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        explicit_dim = _safe_int(os.getenv("OPENAI_EMBED_DIM"))
        if explicit_dim is not None:
            dim = explicit_dim
        else:
            # Known defaults for current OpenAI embedding models
            if model in ("text-embedding-3-large",) or model.endswith("3-large"):
                dim = 3072
            elif model in ("text-embedding-3-small", "text-embedding-ada-002"):
                dim = 1536
            else:
                # Unknown model; allow mismatch to be handled via actual_dimensions
                dim = None
        return {"model": model, "dimensions": dim, "reranker": None}

    # Azure OpenAI (deployment name is the "model" in code paths)
    if p == "azure":
        model = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("OPENAI_EMBED_MODEL", "unknown")
        # Dimensions depend on how the deployment is configured; default unknown.
        dim = _safe_int(os.getenv("AZURE_OPENAI_EMBED_DIM")) or _safe_int(os.getenv("OPENAI_EMBED_DIM"))
        return {"model": model, "dimensions": dim, "reranker": None}

    # Cohere
    if p == "cohere":
        model = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
        dim = _safe_int(os.getenv("COHERE_EMBED_DIM")) or 1024  # embed-english-v3.* typically 1024
        return {"model": model, "dimensions": dim, "reranker": os.getenv("COHERE_RERANKER_MODEL")}

    # HuggingFace Inference API
    if p == "huggingface":
        model = os.getenv("HF_EMBED_MODEL") or os.getenv("HUGGINGFACE_EMBED_MODEL") or "BAAI/bge-large-en-v1.5"
        # Different HF models have different dims; unless explicitly set, leave unknown.
        dim = _safe_int(os.getenv("HF_EMBED_DIM")) or _safe_int(os.getenv("HUGGINGFACE_EMBED_DIM"))
        return {"model": model, "dimensions": dim, "reranker": None}

    # Local (sentence-transformers)
    if p == "local":
        model = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        dim = _safe_int(os.getenv("LOCAL_EMBED_DIM")) or 384  # all-MiniLM-L6-v2 has 384 dims
        return {"model": model, "dimensions": dim, "reranker": None}

    # Qwen
    if p == "qwen":
        model = os.getenv("QWEN_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
        dim = _safe_int(os.getenv("QWEN_DIM")) or 4096  # default, may differ by model
        return {"model": model, "dimensions": dim, "reranker": os.getenv("QWEN_RERANKER_MODEL")}

    # Unknown provider: fall back to generic envs if present
    model = os.getenv("EMBED_MODEL", "unknown")
    dim = _safe_int(os.getenv("EMBED_DIM"))
    return {"model": model, "dimensions": dim, "reranker": None}


def _detect_actual_dimensions(index_dir: Path) -> Optional[int]:
    """
    Try to infer the actual embedding dimensionality from embeddings.npy without
    loading the whole array into memory.
    """
    npy = index_dir / EMBEDDINGS_FILENAME
    if not npy.exists():
        return None
    try:
        import numpy as np  # local import to avoid hard dependency at module import
        arr = np.load(str(npy), mmap_mode="r")
        if arr.ndim == 2:
            return int(arr.shape[1])
    except Exception as e:
        logger.debug(f"Could not read {EMBEDDINGS_FILENAME} to infer dimensions: {e}")
    return None


def _detect_index_type(index_dir: Path) -> str:
    if (index_dir / FAISS_INDEX_FILENAME).exists():
        return "faiss"
    if (index_dir / EMBEDDINGS_FILENAME).exists():
        return "numpy"
    return "none"


def create_index_metadata(
    provider: str,
    num_documents: int,
    num_folders: int,
    index_dir: Path,
    custom_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create metadata for the index.

    Args:
        provider: Embedding provider used (case-insensitive)
        num_documents: Total number of indexed items (rows in mapping)
        num_folders: Number of conversation folders processed
        index_dir: Directory where the index is stored
        custom_metadata: Optional additional metadata to include

    Returns:
        Dictionary containing the metadata
    """
    provider_norm = (provider or "").strip().lower()
    model_info = _resolve_provider_config(provider_norm)

    # Determine index type and try to detect actual dimension if not provided via custom_metadata
    index_type = _detect_index_type(index_dir)
    actual_dim = None
    if custom_metadata:
        actual_dim = custom_metadata.get("actual_dimensions")
    if actual_dim is None:
        actual_dim = _detect_actual_dimensions(index_dir)

    metadata: Dict[str, Any] = {
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider_norm,
        "model": model_info["model"],
        "dimensions": model_info.get("dimensions"),     # configured / expected
        "actual_dimensions": actual_dim,               # observed from embeddings, when known
        "reranker": model_info.get("reranker"),
        "num_documents": int(num_documents),
        "num_folders": int(num_folders),
        "index_type": index_type,
        "half_life_days": int(os.getenv("HALF_LIFE_DAYS", "30")),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }

    if custom_metadata:
        # Do not clobber the computed fields if we have them already.
        for k, v in custom_metadata.items():
            if k not in ("actual_dimensions",):
                metadata[k] = v
        # If caller provided actual_dimensions explicitly, prefer that value.
        if "actual_dimensions" in custom_metadata:
            metadata["actual_dimensions"] = custom_metadata["actual_dimensions"]

    return metadata


def save_index_metadata(metadata: Dict[str, Any], index_dir: Path) -> None:
    """
    Save metadata to the index directory.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path = index_dir / META_FILENAME
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved index metadata to {meta_path}")


def load_index_metadata(index_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load metadata from the index directory.

    Returns:
        Metadata dictionary if found, None otherwise
    """
    meta_path = index_dir / META_FILENAME
    if not meta_path.exists():
        logger.warning(f"No metadata found at {meta_path}")
        return None

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        logger.info(f"Loaded index metadata from {meta_path}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to load metadata from {meta_path}: {e}")
        return None


def validate_index_compatibility(
    index_dir: Path,
    provider: str,
    raise_on_mismatch: bool = False
) -> bool:
    """
    Validate that the index is compatible with the requested provider.

    Args:
        index_dir: Directory where index is stored
        provider: Embedding provider to validate against (case-insensitive)
        raise_on_mismatch: If True, raise an exception on mismatch

    Returns:
        True if compatible (or unknown), False if a hard mismatch was detected
    """
    provider_norm = (provider or "").strip().lower()
    metadata = load_index_metadata(index_dir)

    # If no metadata, warn but allow (backward compatibility).
    if metadata is None:
        logger.warning(
            "No index metadata found. Cannot validate provider compatibility. "
            "Proceeding with caution - results may be incorrect if provider mismatch."
        )
        return True

    indexed_provider = (metadata.get("provider") or "").strip().lower()
    if indexed_provider and indexed_provider != provider_norm:
        msg = (
            f"Index was created with provider '{indexed_provider}' "
            f"but trying to search with provider '{provider_norm}'. "
            f"This will likely produce incorrect results!"
        )
        if raise_on_mismatch:
            raise ValueError(msg)
        logger.error(msg)
        return False

    # Try to compare dimensions if both sides are known
    expected_cfg = _resolve_provider_config(provider_norm)
    expected_dims = expected_cfg.get("dimensions")
    actual_dims = metadata.get("actual_dimensions") or metadata.get("dimensions") or _detect_actual_dimensions(index_dir)

    if actual_dims is not None and expected_dims is not None and actual_dims != expected_dims:
        logger.warning(
            f"Dimension mismatch: index has {actual_dims} dimensions, "
            f"but provider '{provider_norm}' is configured for {expected_dims}. "
            "Proceeding with indexed dimensions."
        )

    # Sanity: check that index_type matches files on disk
    meta_index_type = (metadata.get("index_type") or "unknown").lower()
    detected_type = _detect_index_type(index_dir)
    if meta_index_type != detected_type:
        logger.warning(
            f"Index type in metadata ('{meta_index_type}') does not match files present ('{detected_type}'). "
            "Search will still attempt to proceed."
        )

    return True


def get_index_info(index_dir: Path) -> str:
    """
    Get a human-readable summary of the index metadata and on-disk state.
    """
    metadata = load_index_metadata(index_dir)
    if metadata is None:
        return "No index metadata available"

    # Try to read mapping.json and embeddings.npy to report truthful counts/shapes
    mapping_count: Optional[int] = None
    embeddings_count: Optional[int] = None
    inferred_dim: Optional[int] = None

    mapping_path = index_dir / MAPPING_FILENAME
    if mapping_path.exists():
        try:
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            if isinstance(mapping, list):
                mapping_count = len(mapping)
        except Exception:
            pass

    inferred_dim = _detect_actual_dimensions(index_dir)
    if inferred_dim is not None:
        embeddings_count = None  # count still unknown without loading header fully
        # We can still read row count cheaply via npy header
        try:
            import numpy as np  # local import
            arr = np.load(str(index_dir / EMBEDDINGS_FILENAME), mmap_mode="r")
            embeddings_count = int(arr.shape[0]) if arr.ndim == 2 else None
        except Exception:
            pass

    lines = [
        "Index Information:",
        f"  Provider: {metadata.get('provider', 'unknown')}",
        f"  Model: {metadata.get('model', 'unknown')}",
        f"  Configured Dimensions: {metadata.get('dimensions', 'unknown')}",
    ]
    if metadata.get("actual_dimensions") is not None or inferred_dim is not None:
        lines.append(f"  Actual Dimensions: {metadata.get('actual_dimensions', inferred_dim)}")

    if mapping_count is not None:
        lines.append(f"  Documents (mapping.json): {mapping_count}")
    else:
        lines.append(f"  Documents (metadata): {metadata.get('num_documents', 'unknown')}")

    if embeddings_count is not None:
        lines.append(f"  Embeddings: {embeddings_count}")

    lines.extend([
        f"  Folders: {metadata.get('num_folders', 'unknown')}",
        f"  Created: {metadata.get('created_at', 'unknown')}",
        f"  Index Type: {metadata.get('index_type', 'unknown')}",
        f"  Half-life (days): {metadata.get('half_life_days', 'unknown')}",
    ])

    if reranker := metadata.get("reranker"):
        lines.append(f"  Reranker: {reranker}")

    return "\n".join(lines)
