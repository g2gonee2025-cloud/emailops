#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Library-safe logger
logger = logging.getLogger("emailops.doctor")

INDEX_DIRNAME_DEFAULT = os.getenv("INDEX_DIRNAME", "_index")
REPO_ROOT = Path(__file__).resolve().parents[1]

# -------------------------
# Dependency Management
# -------------------------

_PKG_IMPORT_MAP: Dict[str, str] = {
    # Always
    "numpy": "numpy",

    # Vertex
    "google-genai": "google.genai",                # used by Vertex embeddings path
    "google-cloud-aiplatform": "vertexai",         # provides vertexai.init

    # Other providers used by llm_client
    "openai": "openai",
    "cohere": "cohere",
    "huggingface_hub": "huggingface_hub",
    "requests": "requests",                        # qwen

    # Local embeddings
    "sentence-transformers": "sentence_transformers",

    # Optional extractors used by indexer (warn-only)
    "pypdf": "pypdf",
    "python-docx": "docx",
    "docx2txt": "docx2txt",
    "pandas": "pandas",
    "openpyxl": "openpyxl",

    # Optional FAISS (if present we can report ntotal)
    "faiss-cpu": "faiss",
}

def _try_import(import_name: str) -> bool:
    try:
        __import__(import_name)
        return True
    except Exception:
        return False

def _requirements_file_candidates() -> List[Path]:
    return [
        REPO_ROOT / "requirements.txt",
        Path.cwd() / "requirements.txt",
        Path(__file__).resolve().parent / "requirements.txt",
    ]

def _find_requirements_file() -> Optional[Path]:
    for p in _requirements_file_candidates():
        if p.exists():
            return p
    return None

def _install_packages(packages: List[str], requirements_hint: Optional[Path]) -> bool:
    try:
        if requirements_hint and requirements_hint.exists():
            logger.info("Installing dependencies from %s ...", requirements_hint)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_hint)])
            return True
        if packages:
            logger.info("Installing missing packages: %s", " ".join(packages))
            subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
            return True
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Failed to install dependencies: %s", e)
        return False

def _packages_for_provider(provider: str) -> Tuple[List[str], List[str]]:
    """Return (critical, optional) packages for the chosen provider."""
    provider = (provider or "vertex").lower()

    critical = ["numpy"]
    if provider == "vertex":
        critical += ["google-genai", "google-cloud-aiplatform"]
    elif provider == "openai":
        critical += ["openai"]
    elif provider == "azure":
        # AzureOpenAI comes from the openai SDK
        critical += ["openai"]
    elif provider == "cohere":
        critical += ["cohere"]
    elif provider == "huggingface":
        critical += ["huggingface_hub"]
    elif provider == "qwen":
        critical += ["requests"]
    elif provider == "local":
        critical += ["sentence-transformers"]

    optional = ["pypdf", "python-docx", "docx2txt", "pandas", "openpyxl", "faiss-cpu"]
    return critical, optional

def check_and_install_dependencies(provider: str, auto_install: bool = False) -> None:
    """Non-interactive dependency check with optional auto-install."""
    critical, optional = _packages_for_provider(provider)

    missing_critical = [pkg for pkg in critical if not _try_import(_PKG_IMPORT_MAP[pkg])]
    missing_optional = [pkg for pkg in optional if not _try_import(_PKG_IMPORT_MAP[pkg])]

    if missing_critical:
        logger.error("Missing critical packages for provider '%s': %s", provider, ", ".join(missing_critical))
        if auto_install:
            req = _find_requirements_file()
            if not _install_packages(missing_critical, req):
                sys.exit(1)
        else:
            req_hint = _find_requirements_file()
            if req_hint:
                logger.info("Install with: pip install -r %s", req_hint)
            else:
                logger.info("Install with: pip install %s", " ".join(missing_critical))
            sys.exit(1)

    if missing_optional:
        logger.warning("Optional packages not found (some attachments may be skipped by the indexer): %s",
                       ", ".join(missing_optional))

# -------------------------
# Index & Environment Checks
# -------------------------

def _load_mapping(index_dir: Path) -> List[Dict[str, Any]]:
    mapping_path = index_dir / "mapping.json"
    if not mapping_path.exists():
        return []
    try:
        return json.loads(mapping_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to read mapping.json: %s", e)
        return []

def _get_index_statistics(root: Path, index_dir: Path) -> Dict[str, Any]:
    """Report index/document stats based on mapping.json / embeddings / faiss index."""
    stats: Dict[str, Any] = {
        "conversations_total": 0,
        "conversations_indexed": 0,
        "attachments_indexed": 0,
        "documents_indexed": 0,
        "embeddings_count": 0,
        "embedding_dim": None,
        "faiss_vectors": None,
    }

    # Count total conversations in the export
    conv_dirs = [p.parent for p in root.rglob("Conversation.txt")]
    stats["conversations_total"] = len(conv_dirs)

    # Mapping-driven stats
    mapping = _load_mapping(index_dir)
    stats["documents_indexed"] = len(mapping)

    if mapping:
        conv_ids = {m.get("conv_id") for m in mapping if m.get("conv_id")}
        stats["conversations_indexed"] = len(conv_ids)
        stats["attachments_indexed"] = sum(1 for m in mapping if m.get("doc_type") == "attachment")

    # Embeddings (mmap to avoid loading into RAM)
    embeddings_path = index_dir / "embeddings.npy"
    if embeddings_path.exists():
        try:
            import numpy as _np
            arr = _np.load(str(embeddings_path), mmap_mode="r")
            stats["embeddings_count"] = int(arr.shape[0])
            if arr.ndim == 2:
                stats["embedding_dim"] = int(arr.shape[1])
        except Exception as e:
            logger.warning("Failed to read embeddings.npy: %s", e)

    # FAISS (optional)
    index_path = index_dir / "index.faiss"
    if index_path.exists() and _try_import("faiss"):
        try:
            import faiss  # type: ignore
            fidx = faiss.read_index(str(index_path))
            stats["faiss_vectors"] = int(getattr(fidx, "ntotal", 0))
        except Exception as e:
            logger.warning("Failed to load FAISS index: %s", e)

    return stats

def _summarize_index_compat(index_dir: Path, provider: str) -> bool:
    from emailops.index_metadata import validate_index_compatibility, get_index_info  # lazy import
    logger.info("\n Index Information")
    logger.info(get_index_info(index_dir))
    logger.info("\n Index Compatibility Check")
    ok = validate_index_compatibility(index_dir, provider)
    if not ok:
        logger.error("Provider/index mismatch detected.")
    return bool(ok)

def _probe_embeddings(provider: str) -> Tuple[bool, Optional[int]]:
    """Attempt a 1-shot embed call to verify connectivity; returns (ok, dim)."""
    from emailops.llm_client import embed_texts, LLMError  # lazy import
    try:
        v = embed_texts(["EmailOps Doctor connectivity probe"], provider=provider)
        dim = int(v.shape[1]) if v.ndim == 2 else None
        logger.info("Connected to embedding provider '%s' (dimension: %s).", provider, dim or "unknown")
        return True, dim
    except LLMError as e:
        logger.error("Embedding provider check failed: %s", e)
        return False, None
    except Exception as e:
        logger.error("Unexpected error while probing embeddings: %s", e)
        return False, None

# -------------------------
# CLI
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose and verify the EmailOps environment.")
    ap.add_argument("--root", required=True, help="Export root containing conversations and the index directory")
    ap.add_argument("--provider",
                    choices=["vertex", "openai", "azure", "cohere", "huggingface", "local", "qwen"],
                    help="Embedding provider to check; defaults to the provider recorded in the index (if present) or EMBED_PROVIDER.")
    ap.add_argument("--skip-install-check", action="store_true", help="Skip dependency checks")
    ap.add_argument("--install-missing", action="store_true", help="Auto-install missing dependencies (non-interactive)")
    ap.add_argument("--skip-embed-check", action="store_true", help="Skip the live embedding connectivity probe")
    ap.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    root = Path(args.root).expanduser().resolve()
    index_dir = root / INDEX_DIRNAME_DEFAULT

    # Load index metadata to infer provider when possible
    meta_provider: Optional[str] = None
    try:
        from emailops.index_metadata import load_index_metadata  # lazy import
        meta = load_index_metadata(index_dir)
        if meta and isinstance(meta, dict):
            meta_provider = meta.get("provider")
    except Exception:
        meta = None
        meta_provider = None

    provider = args.provider or meta_provider or os.getenv("EMBED_PROVIDER", "vertex")
    logger.info("=" * 80)
    logger.info("EMAILOPS DOCTOR - System Diagnostics")
    logger.info("=" * 80)

    # 1) Dependencies
    logger.info("\n Dependency Check ")
    if not args.skip_install_check:
        check_and_install_dependencies(provider, auto_install=args.install_missing)
    else:
        logger.info("Skipped dependency checks per flag.")

    # 2) GCP config (shown only for Vertex users)
    if provider == "vertex":
        logger.info("\n GCP Configuration ")
        gcp_project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        gcp_region = os.getenv("GCP_REGION", "global")
        creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        logger.info("Project: %s", gcp_project or "Not set")
        logger.info("Region:  %s", gcp_region)
        logger.info("Creds:   %s", creds or "Not set (ADC will be used)")

        if not gcp_project:
            logger.error("GCP project is not set. Set GCP_PROJECT or GOOGLE_CLOUD_PROJECT.")
        else:
            logger.info("GCP env vars present.")

    # 3) Provider connectivity (real embed call)
    logger.info("\n Embedding Provider Connectivity ")
    if args.skip_embed_check:
        logger.info("Skipped embedding connectivity probe per flag.")
        ok_dim = (True, None)
    else:
        ok_dim = _probe_embeddings(provider)

    # 4) Index & compatibility
    logger.info("\n Index Directory ")
    logger.info("Index dir: %s", index_dir)
    if index_dir.exists():
        _summarize_index_compat(index_dir, provider)

        stats = _get_index_statistics(root, index_dir)
        logger.info("\n Index Stats ")
        logger.info("Conversations indexed: %s / %s",
                    stats["conversations_indexed"], stats["conversations_total"])
        logger.info("Documents indexed:     %s", stats["documents_indexed"])
        logger.info("Attachments indexed:   %s", stats["attachments_indexed"])
        logger.info("Embeddings vectors:    %s", stats["embeddings_count"])
        logger.info("Embedding dimension:   %s", stats["embedding_dim"] or "unknown")
        if stats["faiss_vectors"] is not None:
            logger.info("FAISS vectors:         %s", stats["faiss_vectors"])

        # Light consistency checks
        if stats["documents_indexed"] and stats["embeddings_count"]:
            if stats["documents_indexed"] != stats["embeddings_count"]:
                logger.warning("Mapping vs embeddings count mismatch: %s docs vs %s vectors",
                               stats["documents_indexed"], stats["embeddings_count"])

        # If we successfully probed a dimension and meta recorded dimensions, compare
        if ok_dim[0] and meta and meta.get("actual_dimensions"):
            try:
                detected = ok_dim[1]
                recorded = int(meta["actual_dimensions"])
                if detected and detected != recorded:
                    logger.warning("Embedding dimension mismatch: detected %s vs recorded %s", detected, recorded)
            except Exception:
                pass
    else:
        logger.warning("Index directory not found. Expected at: %s", index_dir)

    # 5) Summary & next steps
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY & NEXT STEPS")
    logger.info("=" * 80)

    next_steps: List[str] = []

    # Project guidance for Vertex
    if provider == "vertex":
        gcp_project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not gcp_project:
            next_steps.append("1) Set GCP_PROJECT (or GOOGLE_CLOUD_PROJECT) and (optionally) GOOGLE_APPLICATION_CREDENTIALS.")

    # Index creation guidance
    if not index_dir.exists():
        next_steps.append("2) Build the index: python -m emailops.email_indexer --root <root>")
    else:
        # If no mapping or empty
        mapping = _load_mapping(index_dir)
        if not mapping:
            next_steps.append("2) Build/rebuild the index: python -m emailops.email_indexer --root <root>")
        else:
            # Compare counts
            stats = _get_index_statistics(root, index_dir)
            if stats["documents_indexed"] != stats["embeddings_count"]:
                next_steps.append("2) Rebuild index to fix vector count mismatch: python -m emailops.email_indexer --root <root> --force-reindex")
            elif stats["conversations_total"] and stats["conversations_indexed"] < stats["conversations_total"]:
                next_steps.append("2) Incrementally update index: python -m emailops.email_indexer --root <root>")

    # Final step if everything looks good
    if not next_steps:
        logger.info("System looks ready for search and drafting.")
        logger.info("Run: python -m emailops.search_and_draft --root <root> --query 'your question'")
    else:
        logger.info("Recommended actions:")
        for s in next_steps:
            logger.info("  %s", s)

    logger.info("\n" + "=" * 80)
    logger.info("Doctor check complete")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
