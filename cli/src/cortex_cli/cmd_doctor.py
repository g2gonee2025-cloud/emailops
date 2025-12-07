#!/usr/bin/env python3
"""
EmailOps Doctor - System Diagnostics Tool.

Implements §13.3 of the Canonical Blueprint.

Provides comprehensive system health checks including:
  • Dependency verification per provider
  • Index health and compatibility checks
  • Embedding API connectivity tests
  • Configuration validation
  • Export root validation (B1 folders)
  • Database connectivity and migrations
  • Dry-run ingest checks

Exit Codes (Canonical per Blueprint §13.3):
  0 - All checks passed
  1 - Warnings (non-critical issues detected)
  2 - Failures (critical issues detected)
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cortex.config.loader import get_config
from cortex.llm.client import embed_texts
from sqlalchemy import create_engine, text

# Library-safe logger
logger = logging.getLogger("cortex.doctor")

# ANSI color codes
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "red": "\033[31m",
}


def _c(text: str, color: str) -> str:
    """Apply ANSI color to text."""
    if not sys.stdout.isatty():
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


INDEX_DIRNAME_DEFAULT = os.getenv("INDEX_DIRNAME", "_index")
REPO_ROOT = Path(__file__).resolve().parents[3]  # Adjusted for new path

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

    Returns:
        Tuple of (success: bool, error_type: str)
    """

    try:
        importlib.import_module(import_name)
        return True, "ok"
    except (ImportError, ModuleNotFoundError):
        # Module not installed
        return False, "not_installed"
    except Exception as e:
        # Module installed but broken (import-time error)
        logger.warning("Module '%s' is installed but broken: %s", import_name, e)
        return False, "broken"


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
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, shell=False
        )
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


def check_and_install_dependencies(
    provider: str, auto_install: bool = False, *, pip_timeout: int = 300
) -> DepReport:
    provider_n = _normalize_provider(provider)
    critical, optional = _packages_for_provider(provider_n)

    # Determine present/missing using import map
    def present(pkgs: list[str]) -> list[str]:
        result = []
        for pkg in pkgs:
            success, error_type = _try_import(_PKG_IMPORT_MAP.get(pkg, pkg))
            if success:
                result.append(pkg)
            elif error_type == "broken":
                logger.error(
                    "Package '%s' is installed but broken - may need reinstall", pkg
                )
        return result

    def missing(pkgs: list[str]) -> list[str]:
        result = []
        for pkg in pkgs:
            success, error_type = _try_import(_PKG_IMPORT_MAP.get(pkg, pkg))
            if not success and error_type == "not_installed":
                result.append(pkg)
            # Note: 'broken' packages are NOT in missing (they're installed but broken)
        return result

    missing_critical = missing(critical)
    missing_optional = missing(optional)
    installed = present(critical + optional)

    if missing_critical:
        logger.error(
            "Missing critical packages for %s: %s", provider_n, missing_critical
        )
        if auto_install:
            if _install_packages(missing_critical, timeout=pip_timeout):
                # Recompute after install
                missing_critical = missing(critical)
                installed = present(critical + optional)
                if missing_critical:
                    logger.error(
                        "Some critical packages still missing after install: %s",
                        missing_critical,
                    )
            else:
                logger.error(
                    "Failed to install critical packages: %s", missing_critical
                )
        else:
            logger.info(
                "Run 'pip install %s' to install missing packages",
                " ".join(missing_critical),
            )
    else:
        logger.info("All critical packages for %s are available", provider_n)

    if missing_optional:
        logger.warning("Missing optional packages: %s", missing_optional)
        if auto_install and _install_packages(missing_optional, timeout=pip_timeout):
            missing_optional = missing(optional)
            installed = present(critical + optional)
    else:
        logger.info("All optional packages are available")

    return DepReport(
        provider=provider_n,
        missing_critical=missing_critical,
        missing_optional=missing_optional,
        installed=installed,
    )


# -------------------------
# Index & Environment Checks
# -------------------------


def _probe_embeddings(_provider: str) -> tuple[bool, int | None]:
    """Test embedding functionality with the configured provider."""
    try:
        # The runtime uses the configured provider from config, not a parameter
        result = embed_texts(["test"])
        if result is not None and len(result) > 0:
            dim = result.shape[1] if hasattr(result, "shape") else len(result[0])
            return True, dim
        return False, None
    except Exception as e:
        logger.warning("Embedding probe failed: %s", e)
        return False, None


# -------------------------
# Database & Cache Checks
# -------------------------


def check_postgres(config: Any) -> tuple[bool, str | None]:
    """Check PostgreSQL connectivity."""
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(config.database.url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, None
    except Exception as e:
        return False, str(e)


def check_redis(_config: Any) -> tuple[bool, str | None]:
    """Check Redis connectivity."""
    try:
        import redis

        # Assuming Redis URL is in env or default
        redis_url = os.getenv("OUTLOOKCORTEX_REDIS_URL", "redis://localhost:6379")
        r = redis.from_url(redis_url)
        r.ping()
        return True, None
    except Exception as e:
        return False, str(e)


# -------------------------
# Export & DB Checks (Blueprint §13.3)
# -------------------------


def check_exports(config: Any, root: Path) -> tuple[bool, list[str], str | None]:
    """
    Check export root and list export folders (B1 validation).

    Blueprint §13.3: Verify export root, list export folders.

    Returns:
        Tuple of (success, list of export folders, error message if any)
    """
    try:
        export_root = root / config.directories.export_root
        if not export_root.exists():
            return False, [], f"Export root does not exist: {export_root}"

        # List B1 folders (conversation export folders)
        folders = []
        for item in export_root.iterdir():
            if item.is_dir():
                # Check if it looks like a B1 folder (has manifest or messages)
                manifest = item / "manifest.json"
                messages_dir = item / "messages"
                if manifest.exists() or messages_dir.exists():
                    folders.append(item.name)

        return True, folders, None
    except Exception as e:
        return False, [], str(e)


def check_db(config: Any) -> tuple[bool, dict[str, Any], str | None]:
    """
    Check database connectivity and migrations status.

    Blueprint §13.3: Check DB connectivity and migrations.

    Returns:
        Tuple of (success, status dict with migration info, error message if any)
    """
    status: dict[str, Any] = {
        "connected": False,
        "migrations_current": None,
        "latest_migration": None,
    }

    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(config.database.url)

        # Test connectivity
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            status["connected"] = True

            # Check alembic_version table for migrations
            try:
                result = conn.execute(
                    text(
                        "SELECT version_num FROM alembic_version ORDER BY version_num DESC LIMIT 1"
                    )
                )
                row = result.fetchone()
                if row:
                    status["latest_migration"] = row[0]
                    status["migrations_current"] = True
            except Exception:
                # alembic_version table may not exist yet
                status["migrations_current"] = False
                status["latest_migration"] = None

        return True, status, None
    except Exception as e:
        return False, status, str(e)


def check_ingest(config: Any, root: Path) -> tuple[bool, dict[str, Any], str | None]:
    """
    Run a dry-run ingest check on a small sample.

    Blueprint §13.3: Run a dry-run ingest of a small sample.

    Returns:
        Tuple of (success, check details, error message if any)
    """
    details: dict[str, Any] = {
        "sample_found": False,
        "parser_ok": False,
        "preprocessor_ok": False,
    }

    try:
        # Find a sample message file to test parsing
        export_root = root / config.directories.export_root
        sample_file = None

        if export_root.exists():
            for folder in export_root.iterdir():
                if folder.is_dir():
                    messages_dir = folder / "messages"
                    if messages_dir.exists():
                        for msg_file in messages_dir.glob("*.json"):
                            sample_file = msg_file
                            break
                if sample_file:
                    break

        if not sample_file:
            return True, details, "No sample messages found (export may be empty)"

        details["sample_found"] = True

        # Test parser import and execution
        try:
            from cortex.ingestion.parser_email import parse_eml_file

            # Actually try to parse the sample file
            parsed = parse_eml_file(sample_file)
            if parsed and parsed.message_id:
                details["parser_ok"] = True
                details["parsed_subject"] = parsed.subject
            else:
                details["parser_ok"] = False
                return False, details, "Parser returned empty result"

        except ImportError:
            details["parser_ok"] = False
            return False, details, "Failed to import email parser"
        except Exception as e:
            details["parser_ok"] = False
            return False, details, f"Parser failed on sample: {e}"

        # Test preprocessor import
        try:
            from cortex.ingestion.text_preprocessor import TextPreprocessor

            # Instantiate to check for model loading issues (spacy etc)
            _ = TextPreprocessor()
            details["preprocessor_ok"] = True
        except ImportError:
            details["preprocessor_ok"] = False
            return False, details, "Failed to import text preprocessor"
        except Exception as e:
            details["preprocessor_ok"] = False
            # This might fail if spacy model missing, which is a valid check failure
            return False, details, f"Preprocessor init failed: {e}"

        return True, details, None
    except Exception as e:
        return False, details, str(e)


def check_index_health(
    config: Any, root: Path
) -> tuple[bool, dict[str, Any], str | None]:
    """Validate index directory presence and DB embedding compatibility."""
    index_dir = root / config.directories.index_dirname
    status: dict[str, Any] = {
        "path": str(index_dir),
        "exists": index_dir.exists(),
        "file_count": 0,
        "config_embedding_dim": config.embedding.output_dimensionality,
        "db_embedding_dim": None,
    }

    if not index_dir.exists():
        return False, status, f"Index directory not found: {index_dir}"

    try:
        status["file_count"] = sum(1 for _ in index_dir.rglob("*"))
    except Exception:
        status["file_count"] = 0

    try:
        engine = create_engine(config.database.url)
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT vector_dims(embedding) AS dim FROM chunks WHERE embedding IS NOT NULL LIMIT 1"
                )
            ).fetchone()
            if result and result.dim:
                status["db_embedding_dim"] = int(result.dim)
    except Exception as exc:
        return False, status, f"Database index check failed: {exc}"

    db_dim = status["db_embedding_dim"]
    cfg_dim = status["config_embedding_dim"]
    if db_dim is not None and db_dim != cfg_dim:
        return (
            False,
            status,
            f"Embedding dimension mismatch (db={db_dim}, config={cfg_dim})",
        )

    return True, status, None


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
    parser = argparse.ArgumentParser(
        description="EmailOps Doctor - System Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes (Canonical per Blueprint §13.3):
  0 - All checks passed
  1 - Warnings (non-critical issues detected)
  2 - Failures (critical issues detected)

Examples:
  cortex doctor                      Basic dependency check
  cortex doctor --check-embeddings   Test embedding API
  cortex doctor --check-exports      Verify export folders
  cortex doctor --check-ingest       Dry-run ingest test
  cortex doctor --auto-install       Fix missing packages
  cortex doctor --json               Machine-readable output
        """,
    )
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument(
        "--provider",
        default="vertex",
        help="Embedding provider to check (aliases: gcp, vertexai, hf)",
    )
    parser.add_argument(
        "--auto-install",
        action="store_true",
        help="Automatically install missing packages",
    )
    parser.add_argument("--check-index", action="store_true", help="Check index health")
    parser.add_argument(
        "--check-db",
        action="store_true",
        help="Check database connectivity and migrations",
    )
    parser.add_argument(
        "--check-redis", action="store_true", help="Check Redis connectivity"
    )
    parser.add_argument(
        "--check-exports",
        action="store_true",
        help="Verify export root and list B1 folders",
    )
    parser.add_argument(
        "--check-ingest", action="store_true", help="Dry-run ingest of sample data"
    )
    parser.add_argument(
        "--check-embeddings", action="store_true", help="Test embedding functionality"
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON only"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose logging (DEBUG)"
    )
    parser.add_argument(
        "--pip-timeout",
        type=int,
        default=None,
        help="pip install timeout in seconds (default 300 or $PIP_TIMEOUT)",
    )

    args = parser.parse_args()
    _configure_logging(args.verbose)

    root = Path(args.root).expanduser().resolve()
    config = get_config()
    provider = _normalize_provider(args.provider)
    pip_timeout = (
        args.pip_timeout if args.pip_timeout is not None else config.system.pip_timeout
    )

    # Detect requirements.txt when present
    req_file = _find_requirements_file()

    if not args.json:
        print()
        print(
            f"{_c('╔═══════════════════════════════════════════════════════════╗', 'cyan')}"
        )
        print(
            f"{_c('║', 'cyan')}  {_c('EmailOps Doctor', 'bold')} - System Diagnostics                    {_c('║', 'cyan')}"
        )
        print(
            f"{_c('╚═══════════════════════════════════════════════════════════╝', 'cyan')}"
        )
        print()
        print(f"  {_c('Provider:', 'dim')} {_c(provider, 'bold')}")
        print(f"  {_c('Root:', 'dim')}     {root}")
        if req_file:
            print(f"  {_c('Deps:', 'dim')}     {req_file}")
        print()

    # Dependency checks
    if not args.json:
        print(f"{_c('▶ Checking dependencies...', 'cyan')}")

    dep_report = check_and_install_dependencies(
        provider, args.auto_install, pip_timeout=pip_timeout
    )

    dep_error = bool(dep_report.missing_critical)

    if not args.json:
        # Show installed packages
        if dep_report.installed:
            print(f"\n  {_c('Installed:', 'green')}")
            for pkg in dep_report.installed[:10]:  # Show first 10
                print(f"    {_c('✓', 'green')} {pkg}")
            if len(dep_report.installed) > 10:
                print(
                    f"    {_c(f'... and {len(dep_report.installed) - 10} more', 'dim')}"
                )

        # Show missing critical
        if dep_report.missing_critical:
            print(f"\n  {_c('Missing (critical):', 'red')}")
            for pkg in dep_report.missing_critical:
                print(f"    {_c('✗', 'red')} {pkg}")
            print(
                f"\n  {_c('TIP:', 'yellow')} Run {_c('cortex doctor --auto-install', 'cyan')} to fix"
            )

        # Show missing optional
        if dep_report.missing_optional:
            print(f"\n  {_c('Missing (optional):', 'yellow')}")
            for pkg in dep_report.missing_optional[:5]:
                print(f"    {_c('○', 'yellow')} {pkg}")
            if len(dep_report.missing_optional) > 5:
                print(
                    f"    {_c(f'... and {len(dep_report.missing_optional) - 5} more', 'dim')}"
                )

    # Index checks
    index_info: dict[str, Any] = {}
    index_error = False
    if args.check_index:
        if not args.json:
            print(f"\n{_c('▶ Checking index health...', 'cyan')}")

        success, status, error = check_index_health(config, root)
        index_info = status
        if not success:
            index_error = True
            if not args.json:
                print(f"  {_c('✗', 'red')} {error}")
        else:
            if not args.json:
                print(f"  {_c('✓', 'green')} Index directory: {status['path']}")
                print(
                    f"    Files: {_c(str(status.get('file_count', 0)), 'dim')} | Embedding dim: {_c(str(status.get('config_embedding_dim')), 'dim')}"
                )

    # Database check
    db_success = None
    db_error_msg = None
    db_error = False
    if args.check_db:
        if not args.json:
            print(f"\n{_c('▶ Checking database...', 'cyan')}")

        success, error = check_postgres(config)
        db_success = success
        db_error_msg = error

        if not success:
            db_error = True
            if not args.json:
                print(f"  {_c('✗', 'red')} Database check failed: {error}")
        else:
            if not args.json:
                print(f"  {_c('✓', 'green')} Database connected")

    # Redis check
    redis_success = None
    redis_error_msg = None
    redis_error = False
    if args.check_redis:
        if not args.json:
            print(f"\n{_c('▶ Checking Redis...', 'cyan')}")

        success, error = check_redis(config)
        redis_success = success
        redis_error_msg = error

        if not success:
            redis_error = True
            if not args.json:
                print(f"  {_c('✗', 'red')} Redis check failed: {error}")
        else:
            if not args.json:
                print(f"  {_c('✓', 'green')} Redis connected")

    # Embeddings probe
    embeddings_success = None
    embeddings_dim = None
    embeddings_error = False
    if args.check_embeddings:
        if not args.json:
            print(f"\n{_c('▶ Testing embeddings...', 'cyan')}")

        success, dim = _probe_embeddings(provider)
        embeddings_success, embeddings_dim = success, dim
        if not success:
            embeddings_error = True
            if not args.json:
                print(
                    f"  {_c('✗', 'red')} Embedding test failed for provider '{provider}'"
                )
                print(f"\n  {_c('TROUBLESHOOTING:', 'yellow')}")
                print("    • Check GOOGLE_APPLICATION_CREDENTIALS is set")
                print(
                    "    • Verify API access with: gcloud auth application-default login"
                )
                print("    • Ensure Vertex AI API is enabled in your GCP project")
        else:
            if not args.json:
                print(f"  {_c('✓', 'green')} Embeddings working")
                print(f"    Dimension: {_c(str(dim), 'bold')}")
                print(f"    Provider:  {_c(provider, 'bold')}")

    # Exports check (Blueprint §13.3)
    exports_success = None
    exports_folders: list[str] = []
    exports_error_msg = None
    exports_warning = False
    if args.check_exports:
        if not args.json:
            print(f"\n{_c('▶ Checking exports...', 'cyan')}")

        success, folders, error = check_exports(config, root)
        exports_success = success
        exports_folders = folders
        exports_error_msg = error

        if not success:
            exports_warning = True  # Warning, not critical failure
            if not args.json:
                print(f"  {_c('⚠', 'yellow')} Export check: {error}")
        else:
            if not args.json:
                print(f"  {_c('✓', 'green')} Export root valid")
                if folders:
                    print(f"    Found {len(folders)} B1 folder(s):")
                    for f in folders[:5]:
                        print(f"      • {f}")
                    if len(folders) > 5:
                        print(f"      {_c(f'... and {len(folders) - 5} more', 'dim')}")
                else:
                    print(
                        f"    {_c('No B1 folders found (export may be empty)', 'dim')}"
                    )

    # Ingest dry-run check (Blueprint §13.3)
    ingest_success = None
    ingest_details: dict[str, Any] = {}
    ingest_error_msg = None
    ingest_warning = False
    if args.check_ingest:
        if not args.json:
            print(f"\n{_c('▶ Checking ingest capability...', 'cyan')}")

        success, details, error = check_ingest(config, root)
        ingest_success = success
        ingest_details = details
        ingest_error_msg = error

        if not success:
            ingest_warning = True
            if not args.json:
                print(f"  {_c('⚠', 'yellow')} Ingest check: {error}")
        else:
            if not args.json:
                print(f"  {_c('✓', 'green')} Ingest capability OK")
                if details.get("sample_found"):
                    print(f"    Sample message found: {_c('✓', 'green')}")
                if details.get("parser_ok"):
                    print(f"    Email parser:         {_c('✓', 'green')}")
                    if details.get("parsed_subject"):
                        print(
                            f"      Subject: {_c(details['parsed_subject'][:40] + '...', 'dim')}"
                        )
                if details.get("preprocessor_ok"):
                    print(f"    Text preprocessor:    {_c('✓', 'green')}")
                if error:
                    print(f"    {_c(error, 'dim')}")

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
            "database": (
                {
                    "success": db_success,
                    "error": db_error_msg,
                }
                if args.check_db
                else None
            ),
            "redis": (
                {
                    "success": redis_success,
                    "error": redis_error_msg,
                }
                if args.check_redis
                else None
            ),
            "embeddings": (
                {
                    "success": embeddings_success,
                    "dimension": embeddings_dim,
                    "provider": provider,
                }
                if args.check_embeddings
                else None
            ),
            "exports": (
                {
                    "success": exports_success,
                    "folders": exports_folders,
                    "error": exports_error_msg,
                }
                if args.check_exports
                else None
            ),
            "ingest": (
                {
                    "success": ingest_success,
                    "details": ingest_details,
                    "error": ingest_error_msg,
                }
                if args.check_ingest
                else None
            ),
        }
        print(json.dumps(payload, indent=2))
    else:
        # Summary
        print()
        print(f"{_c('═' * 60, 'cyan')}")

        # Determine if we have failures (exit 2), warnings (exit 1), or all ok (exit 0)
        has_failures = (
            dep_error or index_error or embeddings_error or db_error or redis_error
        )
        has_warnings = (
            exports_warning or ingest_warning or bool(dep_report.missing_optional)
        )

        if not has_failures and not has_warnings:
            print(f"\n  {_c('✓ All checks passed!', 'green')}")
        elif has_failures:
            print(f"\n  {_c('Failures detected:', 'red')}")
            if dep_error:
                print(f"    {_c('✗', 'red')} Missing critical dependencies")
            if index_error:
                print(f"    {_c('✗', 'red')} Index health issues")
            if embeddings_error:
                print(f"    {_c('✗', 'red')} Embedding connectivity failed")
            if db_error:
                print(f"    {_c('✗', 'red')} Database connectivity failed")
            if redis_error:
                print(f"    {_c('✗', 'red')} Redis connectivity failed")

        if has_warnings:
            print(f"\n  {_c('Warnings:', 'yellow')}")
            if exports_warning:
                print(f"    {_c('⚠', 'yellow')} Export root issues")
            if ingest_warning:
                print(f"    {_c('⚠', 'yellow')} Ingest capability issues")
            if dep_report.missing_optional:
                print(f"    {_c('⚠', 'yellow')} Missing optional packages")

        print()

    # Canonical exit codes per Blueprint §13.3:
    # 0 = all checks passed
    # 1 = warnings (non-critical)
    # 2 = failures (critical)
    has_failures = (
        dep_error or index_error or embeddings_error or db_error or redis_error
    )
    has_warnings = (
        exports_warning or ingest_warning or bool(dep_report.missing_optional)
    )

    if has_failures:
        sys.exit(2)
    elif has_warnings:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
