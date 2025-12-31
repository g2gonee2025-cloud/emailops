#!/usr/bin/env python3
"""
Cortex Doctor - System Diagnostics Tool.

This command connects to the Cortex backend's API to run a series of
health checks on the system's components, or runs local diagnostics.

Exit Codes:
  0 - All checks passed (healthy)
  1 - Warnings detected (degraded)
  2 - Failures detected (unhealthy)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import httpx
from cortex.config.loader import get_config
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
    "red": "\033[31m",
    "cyan": "\033[36m",
}


def _c(text: str, color: str) -> str:
    """Apply ANSI color to text."""
    if not sys.stdout.isatty():
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def _get_api_url(config: Any) -> str:
    """Constructs the API URL from configuration."""
    api_host = "127.0.0.1"
    api_port = "8000"
    if hasattr(config, "api") and config.api.host:
        api_host = config.api.host
    if hasattr(config, "api") and config.api.port:
        api_port = config.api.port
    return f"http://{api_host}:{api_port}"


def _print_report_human(report: dict[str, Any]) -> None:
    """Prints the doctor report in a human-readable format."""
    status_colors = {
        "healthy": "green",
        "degraded": "yellow",
        "unhealthy": "red",
    }
    status = report.get("overall_status", "unknown")
    color = status_colors.get(status, "reset")

    print()
    print(f"{_c('Cortex System Health Report', 'bold')}")
    print(f"{_c('═' * 40, 'cyan')}")
    print(f"Overall Status: {_c(status.upper(), color)}")
    print()

    for check in report.get("checks", []):
        name = check.get("name")
        status = check.get("status")
        message = check.get("message")

        if status == "pass":
            symbol = _c("✓", "green")
            color = "green"
        elif status == "warn":
            symbol = _c("⚠", "yellow")
            color = "yellow"
        else:
            symbol = _c("✗", "red")
            color = "red"

        print(f"{symbol} {_c(name, 'bold')}: {_c(message, color)}")
        details = check.get("details")
        if details:
            for key, value in details.items():
                print(f"    {_c(key, 'dim')}: {value}")


# -----------------------------------------------------------------------------
# Local Check Functions (from HEAD)
# -----------------------------------------------------------------------------


def check_postgres(config: Any) -> tuple[bool, str | None]:
    """Check database connectivity."""
    try:
        engine = create_engine(config.database.url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, None
    except Exception as e:
        return False, str(e)


def check_redis(config: Any) -> tuple[bool, str | None]:
    """Check Redis connectivity."""
    # Placeholder - implement real check if needed or rely on backend
    return True, None


def check_reranker(config: Any) -> tuple[bool, str | None]:
    """Check reranker connectivity."""
    # Placeholder - implement real check if needed
    return True, None


def check_exports(config: Any, root: Path) -> tuple[bool, list[str], str | None]:
    """Check export directory."""
    try:
        export_root = root / config.directories.export_root
        if not export_root.exists():
            return False, [], f"Export root not found: {export_root}"

        folders = [f.name for f in export_root.iterdir() if f.is_dir()]
        return True, folders, None
    except Exception as e:
        return False, [], str(e)


def check_and_install_dependencies(
    provider: str, auto_install: bool, pip_timeout: int
) -> Any:
    """Stub for dependency check."""

    class DepReport:
        installed: list[str] = []
        missing_critical: list[str] = []
        missing_optional: list[str] = []

    return DepReport()


def _probe_embeddings(provider: str) -> tuple[bool, int | None]:
    """Stub to probe embeddings."""
    # Logic to probe embeddings would go here
    return True, 768


def check_ingest(config: Any, root: Path) -> tuple[bool, dict[str, Any], str | None]:
    """
    Run a dry-run ingest check on a small sample.
    """
    details: dict[str, Any] = {
        "sample_found": False,
        "loader_ok": False,
        "preprocessor_ok": False,
    }

    try:
        export_root = root / config.directories.export_root
        sample_dir = _find_sample_conversation_dir(export_root)

        if not sample_dir:
            return (
                True,
                details,
                "No sample conversation directories found in export root",
            )

        details["sample_found"] = True

        # Test loader
        loader_ok, loaded_subject, loader_error = _test_loader_on_dir(sample_dir)
        if loader_error:
            details["loader_ok"] = False
            return False, details, loader_error

        details["loader_ok"] = loader_ok
        if loaded_subject:
            details["loaded_subject"] = loaded_subject

        if not loader_ok:
            return False, details, "Loader returned empty result"

        # Test preprocessor
        preproc_ok, preproc_error = _test_preprocessor_import()
        details["preprocessor_ok"] = preproc_ok
        if preproc_error:
            return False, details, preproc_error

        return True, details, None
    except Exception as e:
        return False, details, str(e)


def _find_sample_conversation_dir(export_root: Path) -> Path | None:
    """Find a sample conversation directory."""
    if not export_root.exists():
        return None

    for folder in export_root.iterdir():
        if not folder.is_dir():
            continue

        # A valid conversation folder has a manifest.json
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            return folder

    return None


def _test_loader_on_dir(
    sample_dir: Path,
) -> tuple[bool, str | None, str | None]:
    """
    Test loading the sample conversation directory.
    Returns: (success, parsed_subject, error_message)
    """
    try:
        from cortex.ingestion.conv_loader import load_conversation
        from cortex.ingestion.core_manifest import resolve_subject

        convo_data = load_conversation(sample_dir)
        if convo_data:
            manifest = convo_data.get("manifest", {})
            summary_json = convo_data.get("summary", {})
            subject, _ = resolve_subject(manifest, summary_json, sample_dir.name)
            return True, subject, None
        return False, None, "load_conversation returned no data"

    except ImportError:
        return False, None, "Failed to import conversation loader"
    except Exception as e:
        return False, None, f"Loader failed on sample: {e}"


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
# Runners (Wrappers)
# -------------------------


def _run_db_check(args: Any, config: Any) -> tuple[bool | None, str | None, bool]:
    if not args.check_db:
        return None, None, False
    if not args.json:
        print(f"\n{_c('▶ Checking database...', 'cyan')}")
        if getattr(args, "verbose", False):
            print(f"  DEBUG: Config DB URL: {config.database.url}")

    success, error = check_postgres(config)
    if not success:
        if not args.json:
            print(f"  {_c('✗', 'red')} Database check failed: {error}")
        return False, error, True

    if not args.json:
        print(f"  {_c('✓', 'green')} Database connected")
    return True, None, False


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
        if not success:
            print(f"  {_c('⚠', 'yellow')} Ingest check: {error}")
        else:
            print(f"  {_c('✓', 'green')} Ingest capability OK")
            if details.get("sample_found"):
                print(f"    Sample conversation found: {_c('✓', 'green')}")
            if details.get("loader_ok"):
                print(f"    Conversation loader:       {_c('✓', 'green')}")
                if details.get("loaded_subject"):
                    print(
                        f"      Subject: {_c(details['loaded_subject'][:40] + '...', 'dim')}"
                    )
            if details.get("preprocessor_ok"):
                print(f"    Text preprocessor:         {_c('✓', 'green')}")

    return success, details, error, warning_flag


def _get_exit_code(status: str) -> int:
    """Maps the overall status to a process exit code."""
    if status == "unhealthy":
        return 2
    if status == "degraded":
        return 1
    return 0


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
        description="Cortex Doctor - System Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON only"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose logging (DEBUG)"
    )
    # Add flags supported by main.py
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--provider", default="vertex", help="Provider")
    parser.add_argument("--auto-install", action="store_true", help="Auto install deps")
    parser.add_argument("--check-index", action="store_true", help="Check index")
    parser.add_argument(
        "--check-embeddings", action="store_true", help="Check embeddings"
    )
    parser.add_argument("--check-db", action="store_true", help="Check database")
    parser.add_argument("--check-redis", action="store_true", help="Check redis")
    parser.add_argument("--check-exports", action="store_true", help="Check exports")
    parser.add_argument("--check-ingest", action="store_true", help="Check ingest")
    parser.add_argument("--check-reranker", action="store_true", help="Check reranker")
    parser.add_argument("--pip-timeout", type=int, default=300, help="Pip timeout")

    args = parser.parse_args()
    _configure_logging(args.verbose)

    config = get_config()

    # If specific checks are requested, run in local mode
    any_local_check = (
        args.check_index or args.check_db or args.check_redis or args.check_ingest
    )

    if any_local_check:
        if not args.json:
            print(f"{_c('Cortex Doctor (Local Mode)', 'bold')}")

        root = Path(args.root).resolve()

        # Run requested checks
        # DB
        db_res = _run_db_check(args, config)

        # Index
        _idx_info, idx_err = _run_index_check(args, config, root)

        # Ingest
        ing_res = _run_ingest_check(args, config, root)

        # Summarize (simplified)
        if not args.json:
            print("\nLocal checks completed.")

        if db_res[2] or idx_err or ing_res[3]:
            sys.exit(2)
        sys.exit(0)

    # Default: API Mode
    api_url = _get_api_url(config)
    doctor_endpoint = f"{api_url}/api/v1/admin/doctor"

    if not args.json:
        print(f"Contacting Cortex backend at {_c(api_url, 'cyan')}...")

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(doctor_endpoint)
            response.raise_for_status()
            report = response.json()

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_report_human(report)

        exit_code = _get_exit_code(report.get("overall_status", "unknown"))
        sys.exit(exit_code)

    except httpx.RequestError as e:
        logger.error(f"API request failed: {e}")
        if not args.json:
            print(
                f"\n{_c('Error:', 'red')} Could not connect to the Cortex backend at {_c(doctor_endpoint, 'bold')}."
            )
            print("Please ensure the backend service is running.")
        sys.exit(2)
    except httpx.HTTPStatusError as e:
        logger.error(
            f"API returned an error: {e.response.status_code} {e.response.text}"
        )
        if not args.json:
            print(
                f"\n{_c('Error:', 'red')} The API returned a status code of {e.response.status_code}."
            )
            print("Response:", e.response.text)
        sys.exit(2)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        if not args.json:
            print(f"\n{_c('An unexpected error occurred.', 'red')}")
        sys.exit(2)


if __name__ == "__main__":
    main()
