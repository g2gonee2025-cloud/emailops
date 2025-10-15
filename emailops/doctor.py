#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Library-safe logger
logger = logging.getLogger("emailops.doctor")


INDEX_DIRNAME_DEFAULT = os.getenv("INDEX_DIRNAME", "_index")
REPO_ROOT = Path(__file__).resolve().parents[1]

# -------------------------
# Provider normalization
# -------------------------

_PROVIDER_ALIASES: dict[str, str] = {
    "gcp": "vertex",
    "vertexai": "vertex",
    "hf": "huggingface",
}


def _normalize_provider(provider: str) -> str:
    p = (provider or "vertex").lower()
    return _PROVIDER_ALIASES.get(p, p)


# -------------------------
# Dependency Management
# -------------------------

_PKG_IMPORT_MAP: dict[str, str] = {
    # Always
    "numpy": "numpy",
    # Vertex
    "google-genai": "google.genai",  # used by Vertex embeddings path
    "google-cloud-aiplatform": "vertexai",  # provides vertexai.init
    # Optional Google libraries
    "google-auth": "google.auth",
    "google-cloud-storage": "google.cloud.storage",
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


def _try_import(import_name: str) -> tuple[bool, str]:
    """
    Check if a module can be imported without side effects.
    
    HIGH #12: Returns (success, error_type) to distinguish:
    - (True, 'ok'): Module imported successfully
    - (False, 'not_installed'): Module not found (ImportError/ModuleNotFoundError)
    - (False, 'broken'): Module installed but has runtime errors
    
    Returns:
        Tuple of (success: bool, error_type: str)
    """
    try:
        import importlib
        importlib.import_module(import_name)
        return True, 'ok'
    except (ImportError, ModuleNotFoundError):
        # Module not installed
        return False, 'not_installed'
    except Exception as e:
        # Module installed but broken (import-time error)
        logger.warning("Module '%s' is installed but broken: %s", import_name, e)
        return False, 'broken'


def _requirements_file_candidates() -> list[Path]:
    """Generate list of possible requirements.txt file locations."""
    return [
        REPO_ROOT / "requirements.txt",
        Path.cwd() / "requirements.txt",
        Path(__file__).resolve().parent / "requirements.txt",
    ]


def _find_requirements_file() -> Path | None:
    """Find the first existing requirements.txt file (if any)."""
    for p in _requirements_file_candidates():
        if p.exists():
            return p
    return None


def _install_packages(packages: list[str], *, timeout: int) -> bool:
    """Safely install Python packages using pip.

    Validates package names for safety, then runs `pip install` with a timeout.
    Returns True on success.
    """
    try:
        # Validate package names (alphanumeric, hyphens, underscores, dots only)
        for pkg in packages:
            if not re.match(r"^[a-zA-Z0-9_\-.]+$", pkg):
                logger.error("Invalid package name: %s", pkg)
                return False

        cmd = [sys.executable, "-m", "pip", "install", *packages]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=False)
        if result.returncode == 0:
            logger.info("Successfully installed packages: %s", packages)
            return True
        else:
            logger.error("Failed to install packages: %s", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        logger.error("Package installation timed out after %ss", timeout)
        return False
    except Exception as e:
        logger.error("Error installing packages: %s", e)
        return False


def _packages_for_provider(provider: str) -> tuple[list[str], list[str]]:
    """Return (critical, optional) packages for the chosen provider."""
    provider = _normalize_provider(provider)
    critical: list[str] = []
    optional: list[str] = []

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
    optional.extend(
        [
            "numpy",
            "faiss-cpu",
            "pypdf",
            "python-docx",
            "docx2txt",
            "pandas",
            "openpyxl",
        ]
    )

    return critical, optional


@dataclass(frozen=True)
class DepReport:
    provider: str
    missing_critical: list[str]
    missing_optional: list[str]
    installed: list[str]


def check_and_install_dependencies(provider: str, auto_install: bool = False, *, pip_timeout: int = 300) -> DepReport:
    provider_n = _normalize_provider(provider)
    critical, optional = _packages_for_provider(provider_n)

    # Determine present/missing using import map
    def present(pkgs: list[str]) -> list[str]:
        result = []
        for pkg in pkgs:
            success, error_type = _try_import(_PKG_IMPORT_MAP.get(pkg, pkg))
            if success:
                result.append(pkg)
            elif error_type == 'broken':
                # HIGH #12: Report broken packages separately
                logger.error("Package '%s' is installed but broken - may need reinstall", pkg)
        return result

    def missing(pkgs: list[str]) -> list[str]:
        result = []
        for pkg in pkgs:
            success, error_type = _try_import(_PKG_IMPORT_MAP.get(pkg, pkg))
            if not success and error_type == 'not_installed':
                result.append(pkg)
            # Note: 'broken' packages are NOT in missing (they're installed but broken)
        return result

    missing_critical = missing(critical)
    missing_optional = missing(optional)
    installed = present(critical + optional)

    if missing_critical:
        logger.error("Missing critical packages for %s: %s", provider_n, missing_critical)
        if auto_install:
            if _install_packages(missing_critical, timeout=pip_timeout):
                # Recompute after install
                missing_critical = missing(critical)
                installed = present(critical + optional)
                if missing_critical:
                    logger.error("Some critical packages still missing after install: %s", missing_critical)
            else:
                logger.error("Failed to install critical packages: %s", missing_critical)
        else:
            logger.info("Run 'pip install %s' to install missing packages", " ".join(missing_critical))
    else:
        logger.info("All critical packages for %s are available", provider_n)

    if missing_optional:
        logger.warning("Missing optional packages: %s", missing_optional)
        if auto_install:
            if _install_packages(missing_optional, timeout=pip_timeout):
                missing_optional = missing(optional)
                installed = present(critical + optional)
    else:
        logger.info("All optional packages are available")

    return DepReport(
        provider=provider_n, missing_critical=missing_critical, missing_optional=missing_optional, installed=installed
    )


# -------------------------
# Index & Environment Checks
# -------------------------


def _load_mapping(index_dir: Path) -> list[dict[str, Any]]:
    from .index_metadata import read_mapping

    return read_mapping(index_dir)


def _get_index_statistics(index_dir: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    try:
        mapping = _load_mapping(index_dir)
        stats["num_documents"] = len(mapping)
        stats["num_conversations"] = len({m.get("conv_id") for m in mapping if m.get("conv_id")})
        stats["total_chars"] = sum(len(m.get("snippet", "")) for m in mapping)
    except Exception as e:
        logger.warning("Failed to load mapping for statistics: %s", e)
        stats["error"] = str(e)
    return stats


def _summarize_index_compat(index_dir: Path, provider: str) -> tuple[bool, str | None]:
    """Return (compat_ok, indexed_provider)."""
    try:
        from .index_metadata import load_index_metadata

        meta = load_index_metadata(index_dir)
        indexed_provider = None
        if meta:
            indexed_provider = str(meta.get("provider", "")).lower() or None
            if indexed_provider and indexed_provider != _normalize_provider(provider):
                logger.warning("Index built with %s, but using %s", indexed_provider, provider)
                return False, indexed_provider
        return True, indexed_provider
    except Exception as e:
        logger.warning("Failed to check index compatibility: %s", e)
        return False, None


def _probe_embeddings(provider: str) -> tuple[bool, int | None]:
    try:
        from .llm_client import embed_texts

        # Try a small embedding to test
        result = embed_texts(["test"], provider=_normalize_provider(provider))
        if result is not None and len(result) > 0:
            dim = result.shape[1] if hasattr(result, "shape") else len(result[0])
            return True, dim
        return False, None
    except Exception as e:
        logger.warning("Embedding probe failed: %s", e)
        return False, None


# -------------------------
# CLI
# -------------------------


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="EmailOps system diagnostics and setup")
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--provider", default="vertex", help="Embedding provider to check (aliases: gcp, vertexai, hf)")
    parser.add_argument("--auto-install", action="store_true", help="Automatically install missing packages")
    parser.add_argument("--check-index", action="store_true", help="Check index health")
    parser.add_argument("--check-embeddings", action="store_true", help="Test embedding functionality")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging (DEBUG)")
    parser.add_argument(
        "--pip-timeout", type=int, default=None, help="pip install timeout in seconds (default 300 or $PIP_TIMEOUT)"
    )

    args = parser.parse_args()
    _configure_logging(args.verbose)

    root = Path(args.root).expanduser().resolve()
    index_dir = root / INDEX_DIRNAME_DEFAULT
    provider = _normalize_provider(args.provider)
    pip_timeout = args.pip_timeout if args.pip_timeout is not None else int(os.getenv("PIP_TIMEOUT", "300"))

    # Detect requirements.txt when present
    req_file = _find_requirements_file()
    if not args.json and req_file:
        print(f"requirements.txt detected at: {req_file}")

    if not args.json:
        print("EmailOps Doctor - System Diagnostics")
        print("=" * 50)

    # Dependency checks
    dep_report = check_and_install_dependencies(provider, args.auto_install, pip_timeout=pip_timeout)

    dep_error = bool(dep_report.missing_critical)

    # Index checks
    index_info: dict[str, Any] = {
        "root": str(index_dir),
        "exists": index_dir.exists(),
        "stats": None,
        "compatibility_ok": None,
        "indexed_provider": None,
        "error": None,
    }
    index_error = False
    if args.check_index:
        if not index_dir.exists():
            index_error = True
            index_info["exists"] = False
            msg = f"Index directory not found: {index_dir}"
            if args.json:
                index_info["error"] = msg
            else:
                print("\nChecking index health...")
                print(msg)
        else:
            if not args.json:
                print("\nChecking index health...")
            stats = _get_index_statistics(index_dir)
            index_info["stats"] = stats if "error" not in stats else {k: v for k, v in stats.items() if k != "error"}
            if "error" in stats:
                index_error = True
                index_info["error"] = stats["error"]
                if not args.json:
                    print(f"Index stats error: {stats['error']}")
            else:
                if not args.json:
                    print(f"Documents: {stats.get('num_documents', 'unknown')}")
                    print(f"Conversations: {stats.get('num_conversations', 'unknown')}")
                    print(f"Total snippet chars: {stats.get('total_chars', 'unknown')}")

            compat_ok, indexed_provider = _summarize_index_compat(index_dir, provider)
            index_info["compatibility_ok"] = compat_ok
            index_info["indexed_provider"] = indexed_provider
            if not args.json:
                if compat_ok:
                    print("Index compatibility: OK")
                else:
                    index_error = True
                    if indexed_provider:
                        print(f"Index compatibility: FAIL (indexed={indexed_provider}, using={provider})")
                    else:
                        print("Index compatibility: FAIL")

    # Embeddings probe
    embeddings_success = None
    embeddings_dim = None
    embeddings_error = False
    if args.check_embeddings:
        if not args.json:
            print("\nTesting embeddings...")
        success, dim = _probe_embeddings(provider)
        embeddings_success, embeddings_dim = success, dim
        if not success:
            embeddings_error = True
            if not args.json:
                print(f"Embeddings failed for provider '{provider}'")
        else:
            if not args.json:
                print(f"Embeddings working, dimension: {dim}")

    # JSON output
    if args.json:
        payload = {
            "provider": provider,
            "dependencies": {
                "missing_critical": dep_report.missing_critical,
                "missing_optional": dep_report.missing_optional,
                "installed": dep_report.installed,
            },
            "index": index_info if args.check_index else None,
            "embeddings": (
                {
                    "success": embeddings_success,
                    "dimension": embeddings_dim,
                    "provider": provider,
                }
                if args.check_embeddings
                else None
            ),
        }
        print(json.dumps(payload, indent=2))
    else:
        print("\nDiagnostics complete.")

    # Exit codes: 2 (deps), 3 (index), 4 (embeddings). Highest wins if multiple.
    exit_code = 0
    if dep_error:
        exit_code = max(exit_code, 2)
    if index_error:
        exit_code = max(exit_code, 3)
    if embeddings_error:
        exit_code = max(exit_code, 4)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
