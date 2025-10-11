#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

# Library-safe logger
logger = logging.getLogger("emailops.doctor")

INDEX_DIRNAME_DEFAULT = os.getenv("INDEX_DIRNAME", "_index")
REPO_ROOT = Path(__file__).resolve().parents[1]

# -------------------------
# Dependency Management
# -------------------------

_PKG_IMPORT_MAP: dict[str, str] = {
    # Always
    "numpy": "numpy",
    # Vertex
    "google-genai": "google.genai",  # used by Vertex embeddings path
    "google-cloud-aiplatform": "vertexai",  # provides vertexai.init
    # Other providers used by llm_client
    "openai": "openai",
    "cohere": "cohere",
    "huggingface_hub": "huggingface_hub",
    "requests": "requests",  # qwen
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
    """
    Returns True only if the *module or submodule* can be imported.
    Using importlib avoids false positives from __import__ on dotted names.
    """
    try:
        import importlib
        importlib.import_module(import_name)
        return True
    except Exception:
        return False


def _requirements_file_candidates() -> list[Path]:
    return [
        REPO_ROOT / "requirements.txt",
        Path.cwd() / "requirements.txt",
        Path(__file__).resolve().parent / "requirements.txt",
    ]


def _find_requirements_file() -> Path | None:
    for p in _requirements_file_candidates():
        if p.exists():
            return p
    return None


def _install_packages(packages: list[str], requirements_hint: Path | None) -> bool:
    try:
        import subprocess
        cmd = [sys.executable, "-m", "pip", "install"] + packages
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info("Successfully installed packages: %s", packages)
            return True
        else:
            logger.error("Failed to install packages: %s", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        logger.error("Package installation timed out")
        return False
    except Exception as e:
        logger.error("Error installing packages: %s", e)
        return False


def _packages_for_provider(provider: str) -> tuple[list[str], list[str]]:
    """Return (critical, optional) packages for the chosen provider."""
    provider = (provider or "vertex").lower()
    critical = []
    optional = []

    if provider == "vertex":
        critical = ["google-genai", "google-cloud-aiplatform"]
        optional = ["google-auth", "google-cloud-storage"]
    elif provider == "openai":
        critical = ["openai"]
        optional = ["tiktoken"]
    elif provider == "azure":
        critical = ["openai"]  # Azure uses OpenAI client
        optional = ["azure-identity"]
    elif provider == "cohere":
        critical = ["cohere"]
    elif provider == "huggingface":
        critical = ["huggingface_hub"]
    elif provider == "qwen":
        critical = ["requests"]
    elif provider == "local":
        critical = ["sentence-transformers"]
        optional = ["torch", "transformers"]

    # Common optional packages
    optional.extend([
        "numpy", "faiss-cpu", "pypdf", "python-docx", "pandas", "openpyxl"
    ])

    return critical, optional


def check_and_install_dependencies(provider: str, auto_install: bool = False) -> None:
    critical, optional = _packages_for_provider(provider)
    all_packages = critical + optional

    missing_critical = [pkg for pkg in critical if not _try_import(_PKG_IMPORT_MAP.get(pkg, pkg))]
    missing_optional = [pkg for pkg in optional if not _try_import(_PKG_IMPORT_MAP.get(pkg, pkg))]

    if missing_critical:
        logger.error("Missing critical packages for %s: %s", provider, missing_critical)
        if auto_install:
            requirements_file = _find_requirements_file()
            if _install_packages(missing_critical, requirements_file):
                logger.info("Critical packages installed successfully")
            else:
                logger.error("Failed to install critical packages")
        else:
            logger.info("Run 'pip install %s' to install missing packages", " ".join(missing_critical))
    else:
        logger.info("All critical packages for %s are available", provider)

    if missing_optional:
        logger.warning("Missing optional packages: %s", missing_optional)
        if auto_install:
            if _install_packages(missing_optional, None):
                logger.info("Optional packages installed successfully")
            else:
                logger.warning("Failed to install some optional packages")
    else:
        logger.info("All optional packages are available")


# -------------------------
# Index & Environment Checks
# -------------------------


def _load_mapping(index_dir: Path) -> list[dict[str, Any]]:
    from .index_metadata import read_mapping
    return read_mapping(index_dir)


def _get_index_statistics(root: Path, index_dir: Path) -> dict[str, Any]:
    stats = {}
    try:
        mapping = _load_mapping(index_dir)
        stats["num_documents"] = len(mapping)
        stats["num_conversations"] = len(set(m.get("conv_id") for m in mapping if m.get("conv_id")))
        stats["total_chars"] = sum(len(m.get("snippet", "")) for m in mapping)
    except Exception as e:
        logger.warning("Failed to load mapping for statistics: %s", e)
        stats["error"] = str(e)
    return stats


def _summarize_index_compat(index_dir: Path, provider: str) -> bool:
    try:
        from .index_metadata import load_index_metadata
        meta = load_index_metadata(index_dir)
        if meta:
            indexed_provider = meta.get("provider", "").lower()
            if indexed_provider and indexed_provider != provider.lower():
                logger.warning("Index built with %s, but using %s", indexed_provider, provider)
                return False
        return True
    except Exception as e:
        logger.warning("Failed to check index compatibility: %s", e)
        return False


def _probe_embeddings(provider: str) -> tuple[bool, int | None]:
    try:
        from .llm_client import embed_texts
        # Try a small embedding to test
        result = embed_texts(["test"], provider=provider)
        if result is not None and len(result) > 0:
            dim = result.shape[1] if hasattr(result, 'shape') else len(result[0])
            return True, dim
        return False, None
    except Exception as e:
        logger.warning("Embedding probe failed: %s", e)
        return False, None


# -------------------------
# CLI
# -------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="EmailOps system diagnostics and setup")
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--provider", default="vertex", help="Embedding provider to check")
    parser.add_argument("--auto-install", action="store_true", help="Automatically install missing packages")
    parser.add_argument("--check-index", action="store_true", help="Check index health")
    parser.add_argument("--check-embeddings", action="store_true", help="Test embedding functionality")

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    index_dir = root / INDEX_DIRNAME_DEFAULT

    print("EmailOps Doctor - System Diagnostics")
    print("=" * 50)

    # Check dependencies
    check_and_install_dependencies(args.provider, args.auto_install)

    if args.check_index and index_dir.exists():
        print("\nChecking index health...")
        stats = _get_index_statistics(root, index_dir)
        print(f"Documents: {stats.get('num_documents', 'unknown')}")
        print(f"Conversations: {stats.get('num_conversations', 'unknown')}")
        print(f"Total snippet chars: {stats.get('total_chars', 'unknown')}")

        compat = _summarize_index_compat(index_dir, args.provider)
        print(f"Index compatibility: {'OK' if compat else 'FAIL'}")

    if args.check_embeddings:
        print("\nTesting embeddings...")
        success, dim = _probe_embeddings(args.provider)
        if success:
            print(f"Embeddings working, dimension: {dim}")
        else:
            print("Embeddings failed")

    print("\nDiagnostics complete.")


if __name__ == "__main__":
    main()
