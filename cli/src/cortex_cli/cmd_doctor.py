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

# Lazy loaded inside functions:
# from cortex.llm.client import embed_texts
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
DEFAULT_PIP_TIMEOUT = 300
DEFAULT_REDIS_URL = "redis://localhost:6379"
REPO_ROOT = Path(__file__).resolve().parents[3]  # Adjusted for new path

# -------------------------
# Provider normalization
# -------------------------

_PROVIDER_ALIASES: dict[str, str] = {
    "hf": "huggingface",
    "do": "digitalocean",
    "gcp": "vertex",
    "vertexai": "vertex",
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
    "pandas": "pandas",
    "openpyxl": "openpyxl",
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

    if provider == "openai" or provider == "digitalocean":
        critical = ["openai"]
        optional = ["tiktoken"]
    elif provider == "vertex":
        critical = ["google-genai"]
    elif provider == "cohere":
        critical = ["cohere"]
    elif provider == "huggingface":
        critical = ["huggingface_hub"]
    elif provider == "qwen":
        critical = ["requests"]
    elif provider == "local":
        critical = ["sentence-transformers"]
        optional = ["torch", "transformers"]
    elif provider == "vertex":
        critical = ["google-genai"]

    # Common optional packages
    optional.extend(
        [
            "numpy",
            "pypdf",
            "python-docx",
            "pandas",
            "openpyxl",
        ]
    )

    return critical, optional


dependencies = {
    "required": [
        "fastapi",
        "sqlalchemy",
        "pydantic",
        # "google-cloud-aiplatform",  # Removed
        "openai",
    ],
    "optional": ["uvicorn", "alembic", "psycopg2"],
}


@dataclass(frozen=True)
class DepReport:
    provider: str
    missing_critical: list[str]
    missing_optional: list[str]
    installed: list[str]


def check_and_install_dependencies(
    provider: str, auto_install: bool = False, *, pip_timeout: int = DEFAULT_PIP_TIMEOUT
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
        from cortex.llm.client import embed_texts

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
        redis_url = os.getenv("OUTLOOKCORTEX_REDIS_URL", DEFAULT_REDIS_URL)
        r = redis.from_url(redis_url)
        r.ping()
        return True, None
    except Exception as e:
        return False, str(e)


def check_reranker(config: Any) -> tuple[bool, str | None]:
    """Check reranker endpoint connectivity."""
    try:
        import httpx

        reranker_endpoint = getattr(config.search, "reranker_endpoint", None)
        if not reranker_endpoint:
            return (
                False,
                "No reranker endpoint configured (OUTLOOKCORTEX_RERANKER_ENDPOINT)",
            )

        # Test health endpoint
        health_url = f"{reranker_endpoint.rstrip('/')}/health"
        try:
            resp = httpx.get(health_url, timeout=5.0)
            if resp.status_code == 200:
                return True, None
            return False, f"Reranker returned status {resp.status_code}"
        except httpx.ConnectError:
            return False, f"Cannot connect to reranker at {reranker_endpoint}"
        except httpx.TimeoutException:
            return False, f"Reranker timeout at {reranker_endpoint}"
    except ImportError:
        return False, "httpx not installed (pip install httpx)"
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
        export_root = root / config.directories.export_root
        sample_file = _find_sample_file(export_root)

        if not sample_file:
            return (
                True,
                details,
                "No sample messages found (checked *.eml, *.json in messages/)",
            )

        details["sample_found"] = True

        # Test parser
        parser_ok, parsed_subject, parser_error = _test_parser_on_file(sample_file)
        if parser_error:
            details["parser_ok"] = False
            return False, details, parser_error

        details["parser_ok"] = parser_ok
        if parsed_subject:
            details["parsed_subject"] = parsed_subject

        if not parser_ok:
            return False, details, "Parser returned empty result"

        # Test preprocessor
        preproc_ok, preproc_error = _test_preprocessor_import()
        details["preprocessor_ok"] = preproc_ok
        if preproc_error:
            return False, details, preproc_error

        return True, details, None
    except Exception as e:
        return False, details, str(e)


def _find_sample_file(export_root: Path) -> Path | None:
    """Find a sample .eml or .json message file."""
    if not export_root.exists():
        return None

    for folder in export_root.iterdir():
        if not folder.is_dir():
            continue

        messages_dir = folder / "messages"
        if not messages_dir.exists():
            continue

        # Try .eml first
        for msg_file in messages_dir.glob("*.eml"):
            return msg_file

        # Fallback to .json
        for msg_file in messages_dir.glob("*.json"):
            return msg_file

    return None


def _test_parser_on_file(sample_file: Path) -> tuple[bool, str | None, str | None]:
    """
    Test parsing the sample file.
    Returns: (success, parsed_subject, error_message)
    """
    try:
        from cortex.archive.parser_email import parse_eml_file

        if sample_file.suffix.lower() == ".json":
            import json

            with sample_file.open() as f:
                try:
                    json.load(f)
                    return True, "N/A (JSON export)", None
                except json.JSONDecodeError:
                    return False, None, "Invalid JSON sample"
        else:
            parsed = parse_eml_file(sample_file)
            if parsed and parsed.message_id:
                return True, parsed.subject, None
            return False, None, None

    except ImportError:
        return False, None, "Failed to import email parser"
    except Exception as e:
        return False, None, f"Parser failed on sample: {e}"


def _test_preprocessor_import() -> tuple[bool, str | None]:
    """Test if text preprocessor can be imported."""
    try:
        from cortex.ingestion.text_preprocessor import TextPreprocessor

        if TextPreprocessor:
            return True, None
        return (
            False,
            "TextPreprocessor class not found",
        )  # Should not happen if import works
    except ImportError:
        return False, "Failed to import text preprocessor"
    except Exception as e:
        return False, f"Preprocessor import failed: {e}"


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
# CLI & Helpers
# -------------------------


def _run_dep_check(
    args: Any, provider: str, pip_timeout: int
) -> tuple[DepReport, bool]:
    if not args.json:
        print(f"{_c('▶ Checking dependencies...', 'cyan')}")

    dep_report = check_and_install_dependencies(
        provider, args.auto_install, pip_timeout=pip_timeout
    )
    dep_error = bool(dep_report.missing_critical)

    if not args.json:
        _print_dep_report(dep_report)

    return dep_report, dep_error


def _print_dep_report(dep_report: DepReport) -> None:
    """Print dependency report to console."""
    if dep_report.installed:
        print(f"\n  {_c('Installed:', 'green')}")
        for pkg in dep_report.installed[:10]:
            print(f"    {_c('✓', 'green')} {pkg}")
        if len(dep_report.installed) > 10:
            print(f"    {_c(f'... and {len(dep_report.installed) - 10} more', 'dim')}")

    if dep_report.missing_critical:
        print(f"\n  {_c('Missing (critical):', 'red')}")
        for pkg in dep_report.missing_critical:
            print(f"    {_c('✗', 'red')} {pkg}")
        print(
            f"\n  {_c('TIP:', 'yellow')} Run {_c('cortex doctor --auto-install', 'cyan')} to fix"
        )

    if dep_report.missing_optional:
        print(f"\n  {_c('Missing (optional):', 'yellow')}")
        for pkg in dep_report.missing_optional[:5]:
            print(f"    {_c('○', 'yellow')} {pkg}")
        if len(dep_report.missing_optional) > 5:
            print(
                f"    {_c(f'... and {len(dep_report.missing_optional) - 5} more', 'dim')}"
            )


def _run_index_check(args: Any, config: Any, root: Path) -> tuple[dict[str, Any], bool]:
    info: dict[str, Any] = {}
    error_flag = False
    if args.check_index:
        if not args.json:
            print(f"\n{_c('▶ Checking index health...', 'cyan')}")

        success, status, error = check_index_health(config, root)
        info = status
        if not success:
            error_flag = True
            if not args.json:
                print(f"  {_c('✗', 'red')} {error}")
        else:
            if not args.json:
                print(f"  {_c('✓', 'green')} Index directory: {status['path']}")
                print(
                    f"    Files: {_c(str(status.get('file_count', 0)), 'dim')} | Embedding dim: {_c(str(status.get('config_embedding_dim')), 'dim')}"
                )
    return info, error_flag


def _run_db_check(args: Any, config: Any) -> tuple[bool | None, str | None, bool]:
    success_ret = None
    error_msg = None
    error_flag = False
    if args.check_db:
        if not args.json:
            print(f"\n{_c('▶ Checking database...', 'cyan')}")
            if getattr(args, "verbose", False):
                print(
                    f"  DEBUG: OUTLOOKCORTEX_DB_URL env: {os.environ.get('OUTLOOKCORTEX_DB_URL')}"
                )
                print(f"  DEBUG: Config DB URL: {config.database.url}")

        success, error = check_postgres(config)
        success_ret = success
        error_msg = error

        if not success:
            error_flag = True
            if not args.json:
                print(f"  {_c('✗', 'red')} Database check failed: {error}")
        else:
            if not args.json:
                print(f"  {_c('✓', 'green')} Database connected")
    return success_ret, error_msg, error_flag


def _run_redis_check(args: Any, config: Any) -> tuple[bool | None, str | None, bool]:
    success_ret = None
    error_msg = None
    error_flag = False
    if args.check_redis:
        if not args.json:
            print(f"\n{_c('▶ Checking Redis...', 'cyan')}")

        success, error = check_redis(config)
        success_ret = success
        error_msg = error

        if not success:
            error_flag = True
            if not args.json:
                print(f"  {_c('✗', 'red')} Redis check failed: {error}")
        else:
            if not args.json:
                print(f"  {_c('✓', 'green')} Redis connected")
    return success_ret, error_msg, error_flag


def _run_embed_check(args: Any, provider: str) -> tuple[bool | None, int | None, bool]:
    success_ret = None
    dim_ret = None
    error_flag = False
    if args.check_embeddings:
        if not args.json:
            print(f"\n{_c('▶ Testing embeddings...', 'cyan')}")

        success, dim = _probe_embeddings(provider)
        success_ret, dim_ret = success, dim

        embed_endpoint = (
            os.environ.get("EMBED_ENDPOINT")
            or os.environ.get("DO_LLM_BASE_URL")
            or "http://embeddings-api.emailops.svc.cluster.local"
        )
        if not embed_endpoint.endswith("/v1"):
            embed_endpoint = f"{embed_endpoint.rstrip('/')}/v1"

        if not success:
            error_flag = True
            if not args.json:
                print(f"  {_c('✗', 'red')} Embedding test failed")
                print(f"    Endpoint: {_c(embed_endpoint, 'dim')}")
                print(f"\n  {_c('TROUBLESHOOTING:', 'yellow')}")
                print(f"    • Ensure embedding server is running at: {embed_endpoint}")
                print("    • Check EMBED_ENDPOINT or DO_LLM_BASE_URL env var")
                print(
                    "    • For local dev, start vLLM: python -m vllm.entrypoints.openai.api_server"
                )
        else:
            if not args.json:
                print(f"  {_c('✓', 'green')} Embeddings working")
                print(f"    Endpoint:  {_c(embed_endpoint, 'dim')}")
                print(f"    Dimension: {_c(str(dim), 'bold')}")
    return success_ret, dim_ret, error_flag


def _run_rerank_check(args: Any, config: Any) -> tuple[bool | None, str | None, bool]:
    success_ret = None
    error_msg = None
    error_flag = False
    if args.check_reranker:
        if not args.json:
            print(f"\n{_c('▶ Testing reranker...', 'cyan')}")

        success, error = check_reranker(config)
        success_ret = success
        error_msg = error

        if not success:
            error_flag = True
            if not args.json:
                print(f"  {_c('✗', 'red')} Reranker check failed: {error}")
        else:
            if not args.json:
                print(f"  {_c('✓', 'green')} Reranker endpoint OK")
    return success_ret, error_msg, error_flag


def _run_export_check(
    args: Any, config: Any, root: Path
) -> tuple[bool | None, list[str], str | None, bool]:
    if not args.check_exports:
        return None, [], None, False

    if not args.json:
        print(f"\n{_c('▶ Checking exports...', 'cyan')}")

    success, folders, error = check_exports(config, root)
    warning_flag = not success

    if not args.json:
        _print_export_result(success, folders, error)

    return success, folders, error, warning_flag


def _print_export_result(success: bool, folders: list[str], error: str | None) -> None:
    """Print export check result to console."""
    if not success:
        print(f"  {_c('⚠', 'yellow')} Export check: {error}")
        return

    print(f"  {_c('✓', 'green')} Export root valid")
    if folders:
        print(f"    Found {len(folders)} B1 folder(s):")
        for f in folders[:5]:
            print(f"      • {f}")
        if len(folders) > 5:
            print(f"      {_c(f'... and {len(folders) - 5} more', 'dim')}")
    else:
        print(f"    {_c('No B1 folders found (export may be empty)', 'dim')}")


def _run_ingest_check(
    args: Any, config: Any, root: Path
) -> tuple[bool | None, dict[str, Any], str | None, bool]:
    if not args.check_ingest:
        return None, {}, None, False

    if not args.json:
        print(f"\n{_c('▶ Checking ingest capability...', 'cyan')}")

    success, details, error = check_ingest(config, root)
    warning_flag = not success

    if not args.json:
        _print_ingest_result(success, details, error)

    return success, details, error, warning_flag


def _print_ingest_result(
    success: bool, details: dict[str, Any], error: str | None
) -> None:
    """Print ingest check result to console."""
    if not success:
        print(f"  {_c('⚠', 'yellow')} Ingest check: {error}")
        return

    print(f"  {_c('✓', 'green')} Ingest capability OK")
    if details.get("sample_found"):
        print(f"    Sample message found: {_c('✓', 'green')}")
    if details.get("parser_ok"):
        print(f"    Email parser:         {_c('✓', 'green')}")
        if details.get("parsed_subject"):
            print(f"      Subject: {_c(details['parsed_subject'][:40] + '...', 'dim')}")
    if details.get("preprocessor_ok"):
        print(f"    Text preprocessor:    {_c('✓', 'green')}")
    if error:
        print(f"    {_c(error, 'dim')}")


def _print_json_output(
    args: Any,
    provider: str,
    dep_report: DepReport,
    index_info: dict[str, Any],
    db_res: tuple[bool | None, str | None],
    redis_res: tuple[bool | None, str | None],
    embed_res: tuple[bool | None, int | None, str],
    rerank_res: tuple[bool | None, str | None],
    export_res: tuple[bool | None, list[str], str | None],
    ingest_res: tuple[bool | None, dict[str, Any], str | None],
) -> None:
    payload = {
        "provider": provider,
        "dependencies": {
            "missing_critical": dep_report.missing_critical,
            "missing_optional": dep_report.missing_optional,
            "installed": dep_report.installed,
        },
        "index": index_info if args.check_index else None,
        "database": (
            {"success": db_res[0], "error": db_res[1]} if args.check_db else None
        ),
        "redis": (
            {"success": redis_res[0], "error": redis_res[1]}
            if args.check_redis
            else None
        ),
        "embeddings": (
            {
                "success": embed_res[0],
                "dimension": embed_res[1],
                "provider": embed_res[2],
            }
            if args.check_embeddings
            else None
        ),
        "exports": (
            {
                "success": export_res[0],
                "folders": export_res[1],
                "error": export_res[2],
            }
            if args.check_exports
            else None
        ),
        "ingest": (
            {
                "success": ingest_res[0],
                "details": ingest_res[1],
                "error": ingest_res[2],
            }
            if args.check_ingest
            else None
        ),
        "reranker": (
            {"success": rerank_res[0], "error": rerank_res[1]}
            if args.check_reranker
            else None
        ),
    }
    print(json.dumps(payload, indent=2))


def _print_summary_and_exit(failures: list[str], warnings: list[str]) -> None:
    print()
    print(f"{_c('═' * 60, 'cyan')}")

    if not failures and not warnings:
        print(f"\n  {_c('✓ All checks passed!', 'green')}")
    elif failures:
        print(f"\n  {_c('Failures detected:', 'red')}")
        for f in failures:
            print(f"    {_c('✗', 'red')} {f}")

    if warnings:
        print(f"\n  {_c('Warnings:', 'yellow')}")
        for w in warnings:
            print(f"    {_c('⚠', 'yellow')} {w}")

    print()

    if failures:
        sys.exit(2)
    elif warnings:
        sys.exit(1)
    else:
        sys.exit(0)


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
        "--check-reranker",
        action="store_true",
        help="Test reranker endpoint connectivity",
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
    dep_report, dep_error = _run_dep_check(args, provider, pip_timeout)

    # Component checks
    index_info, index_error = _run_index_check(args, config, root)
    db_success, db_err_msg, db_error = _run_db_check(args, config)
    redis_success, redis_err_msg, redis_error = _run_redis_check(args, config)
    embed_success, embed_dim, embed_error = _run_embed_check(args, provider)
    rerank_success, rerank_err_msg, rerank_error = _run_rerank_check(args, config)
    exp_success, exp_folders, exp_err_msg, exp_warning = _run_export_check(
        args, config, root
    )
    ing_success, ing_details, ing_err_msg, ing_warning = _run_ingest_check(
        args, config, root
    )

    # Output
    if args.json:
        _print_json_output(
            args,
            provider,
            dep_report,
            index_info,
            (db_success, db_err_msg),
            (redis_success, redis_err_msg),
            (embed_success, embed_dim, provider),
            (rerank_success, rerank_err_msg),
            (exp_success, exp_folders, exp_err_msg),
            (ing_success, ing_details, ing_err_msg),
        )
    else:
        failures, warnings = _collect_failures_and_warnings(
            dep_error=dep_error,
            index_error=index_error,
            embed_error=embed_error,
            db_error=db_error,
            redis_error=redis_error,
            rerank_error=rerank_error,
            exp_warning=exp_warning,
            ing_warning=ing_warning,
            missing_optional=bool(dep_report.missing_optional),
        )
        _print_summary_and_exit(failures, warnings)


def _collect_failures_and_warnings(
    *,
    dep_error: bool,
    index_error: bool,
    embed_error: bool,
    db_error: bool,
    redis_error: bool,
    rerank_error: bool,
    exp_warning: bool,
    ing_warning: bool,
    missing_optional: bool,
) -> tuple[list[str], list[str]]:
    """Collect failures and warnings from check results."""
    failures = []
    if dep_error:
        failures.append("Missing critical dependencies")
    if index_error:
        failures.append("Index health issues")
    if embed_error:
        failures.append("Embedding connectivity failed")
    if db_error:
        failures.append("Database connectivity failed")
    if redis_error:
        failures.append("Redis connectivity failed")
    if rerank_error:
        failures.append("Reranker connectivity failed")

    warnings = []
    if exp_warning:
        warnings.append("Export root issues")
    if ing_warning:
        warnings.append("Ingest capability issues")
    if missing_optional:
        warnings.append("Missing optional packages")

    return failures, warnings


if __name__ == "__main__":
    main()
