"""
Cortex CLI Entrypoint.

Implements §13.3 of the Canonical Blueprint.

Usage:
    cortex <command> [options]

Commands:
    ingest      Process and ingest email exports
    index       Build/rebuild search index with embeddings
    search      Search indexed emails
    doctor      Run system diagnostics
    status      Show current environment status
    config      View or validate configuration
    version     Show version information

Examples:
    cortex ingest ./exports/my_emails
    cortex index --provider digitalocean
    cortex search "contract renewal"
    cortex doctor --check-embeddings
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

# Ensure backend package is importable when running CLI from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
if BACKEND_SRC.exists() and str(BACKEND_SRC) not in sys.path:
    sys.path.append(str(BACKEND_SRC))

# Ensure CLI package is importable when running directly
CLI_SRC = PROJECT_ROOT / "cli" / "src"
if CLI_SRC.exists() and str(CLI_SRC) not in sys.path:
    sys.path.insert(0, str(CLI_SRC))

# Ensure Workers package is importable
WORKERS_SRC = PROJECT_ROOT / "workers" / "src"
if WORKERS_SRC.exists() and str(WORKERS_SRC) not in sys.path:
    sys.path.append(str(WORKERS_SRC))

from dotenv import load_dotenv  # noqa: E402

# Load .env file BEFORE any environment variable access
# This ensures OUTLOOKCORTEX_* vars are available for status/config commands
_dotenv_path = PROJECT_ROOT / ".env"
if _dotenv_path.exists():
    load_dotenv(_dotenv_path)
else:
    load_dotenv()  # Fall back to default dotenv discovery

from cortex_cli import cmd_search
from cortex_cli._config_helpers import (  # noqa: E402
    _print_human_config,
    _print_json_config,
)
from cortex_cli.style import colorize as _colorize  # noqa: E402


# Minimal protocol for the config object to satisfy static analysis when imports fail
class CoreConfig(Protocol):
    env: str
    provider: str
    persona: str


class SystemConfig(Protocol):
    log_level: str


class EmbeddingConfig(Protocol):
    model_name: str
    output_dimensionality: int
    batch_size: int


class ProcessingConfig(Protocol):
    chunk_size: int
    chunk_overlap: int


class SearchConfig(Protocol):
    fusion_strategy: str
    k: int
    recency_boost_strength: float
    mmr_lambda: float
    reranker_endpoint: str | None


class EmailOpsConfigProto(Protocol):
    core: CoreConfig
    system: SystemConfig
    embedding: EmbeddingConfig
    search: SearchConfig
    processing: ProcessingConfig

    def model_dump(self) -> dict[str, Any]: ...


# Import config model for typing if available; otherwise fall back to protocol
if TYPE_CHECKING:
    # Use the concrete config class for type checking; fallback to the protocol at runtime.
    from cortex.config.loader import EmailOpsConfig as EmailOpsConfig
else:
    EmailOpsConfig = EmailOpsConfigProto

from cortex.observability import init_observability

# Lazy import for heavy dependencies
from cortex_cli.cmd_doctor import main as doctor_main

# ANSI color codes for terminal output


# -----------------------------------------------------------------------------
# Help Text Constants
# -----------------------------------------------------------------------------

NOT_SET = "not set"

CORE_COMMANDS = [
    ("pipeline", "Run the unified ingestion pipeline"),
    ("ingest", "Process and ingest email exports into the system"),
    ("index", "Build/rebuild search index with embeddings"),
    ("search", "Search indexed emails with natural language"),
    ("validate", "Validate export folder structure (B1)"),
]

RAG_COMMANDS = [
    ("answer", "Ask questions about your emails"),
    ("draft", "Draft email replies based on context"),
    ("summarize", "Summarize email threads"),
]

UTILITY_COMMANDS = [
    ("doctor", "Run system diagnostics and health checks"),
    ("status", "Show current environment and configuration"),
    ("config", "View, validate, or export configuration"),
    ("version", "Display version information"),
    ("autofix", "Automatically fix common code issues"),
]

DATA_COMMANDS = [
    ("db", "Database management (stats, migrate)"),
    ("embeddings", "Embedding management (stats, backfill)"),
    ("graph", "Knowledge Graph commands (discover-schema)"),
    ("s3", "S3/Spaces storage (list, ingest, check-structure)"),
    ("maintenance", "System maintenance (resolve-entities)"),
    ("queue", "Job queue management (stats)"),
    ("fix-issues", "Generate patches for SonarQube issues"),
    ("patch", "Fix malformed patch files"),
    ("schema", "Graph schema analysis tools"),
]

COMMON_OPTIONS = [
    ("--help, -h", "Show help for a command"),
    ("--verbose, -v", "Enable verbose output (where supported)"),
    ("--json", "Output in JSON format (where supported)"),
]

EXAMPLES = [
    ("cortex pipeline --source Outlook/ --tenant acme", "Run full pipeline"),
    ("cortex pipeline --dry-run --limit 5", "Preview 5 folders"),
    ('cortex search "contract terms"', "Search for contract terms"),
    ("cortex doctor --check-embeddings", "Test embedding connectivity"),
]

WORKFLOW_STEPS = [
    ("cortex doctor", "Check system health"),
    ("cortex pipeline --source <path>", "Run full E2E pipeline"),
    ("cortex search <query>", "Query your emails"),
    ("cortex answer <question>", "Get AI answers"),
]


def _print_banner() -> None:
    """Print the CLI banner."""
    banner = f"""
{_colorize("╔═══════════════════════════════════════════════════════════╗", "cyan")}
{_colorize("║", "cyan")}  {_colorize("EmailOps Cortex CLI", "bold")} - Email Intelligence Platform       {_colorize("║", "cyan")}
{_colorize("║", "cyan")}  Powered by LangGraph                                      {_colorize("║", "cyan")}
{_colorize("╚═══════════════════════════════════════════════════════════╝", "cyan")}
"""
    print(banner)


def _print_section(
    title: str, items: list[tuple[str, str]], item_color: str = "green", width: int = 12
) -> None:
    """Print a section of the help output."""
    print(f"\n{_colorize(title, 'bold')}")
    for left, right in items:
        print(f"    {_colorize(left, item_color):{width}} {right}")


def _print_usage() -> None:
    """Print user-friendly usage information."""
    _print_banner()

    print(f"{_colorize('USAGE:', 'bold')}")
    print(f"    cortex {_colorize('<command>', 'cyan')} [options]\n")

    print(
        f"{_colorize('CORE COMMANDS:', 'bold')}  {_colorize('(Email Processing)', 'dim')}"
    )
    for cmd, desc in CORE_COMMANDS:
        print(f"    {_colorize(cmd, 'green'):12} {desc}")

    print(
        f"\n{_colorize('RAG COMMANDS:', 'bold')}   {_colorize('(AI Capabilities)', 'dim')}"
    )
    for cmd, desc in RAG_COMMANDS:
        print(f"    {_colorize(cmd, 'green'):12} {desc}")

    _print_section("UTILITY COMMANDS:", UTILITY_COMMANDS)
    _print_section("DATA COMMANDS:", DATA_COMMANDS)

    print(f"\n{_colorize('COMMON OPTIONS:', 'bold')}")
    for opt, desc in COMMON_OPTIONS:
        print(f"    {_colorize(opt, 'yellow'):16} {desc}")

    print(f"\n{_colorize('EXAMPLES:', 'bold')}")
    for example, desc in EXAMPLES:
        print(f"    {_colorize(example, 'dim'):44} # {desc}")

    print(f"\n{_colorize('WORKFLOW:', 'bold')}")
    for i, (cmd, desc) in enumerate(WORKFLOW_STEPS, 1):
        # We need to manually construct the string to colorize parts of it
        # The stored constant is just the command string, we colorize it here
        # Actually in constants we stored "cortex doctor", let's reconstruct logic or simplified it
        # The original code did: print(f"    {i}. {_colorize('cortex doctor', 'cyan')} → Check system health")
        # To match exact output:
        # We stored ("cortex doctor", "Check system health") in WORKFLOW_STEPS
        print(f"    {i}. {_colorize(cmd, 'cyan')} → {desc}")

    print(f"\n{_colorize('DOCUMENTATION:', 'bold')}")
    print(
        f"    See {_colorize('docs/CANONICAL_BLUEPRINT.md', 'blue')} for full specifications"
    )
    print()


def _print_version() -> None:
    """Print version information."""
    try:
        from importlib.metadata import version as get_version

        ver = get_version("cortex-cli")
    except Exception:
        ver = "0.1.0-dev"

    print(f"{_colorize('EmailOps Cortex CLI', 'bold')}")
    print(f"  Version:  {_colorize(ver, 'green')}")
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  Platform: {sys.platform}")


def _show_status(json_output: bool = False) -> None:
    """Show current environment status."""
    import json
    import os

    cwd = Path.cwd()

    # Collect data first
    status_data = {
        "environment": {
            "OUTLOOKCORTEX_ENV": os.getenv("OUTLOOKCORTEX_ENV", NOT_SET),
            "OUTLOOKCORTEX_DB_URL": (
                "***" if os.getenv("OUTLOOKCORTEX_DB_URL") else NOT_SET
            ),
            "GOOGLE_APPLICATION_CREDENTIALS": os.getenv(
                "GOOGLE_APPLICATION_CREDENTIALS", NOT_SET
            ),
        },
        "directories": {},
        "config_files": {},
    }

    dirs_to_check = [
        ("backend/src/cortex", "Backend module"),
        ("cli/src/cortex_cli", "CLI module"),
        ("workers/src", "Workers module"),
        ("docs", "Documentation"),
        ("secrets", "Secrets directory"),
    ]

    for dir_path, desc in dirs_to_check:
        full_path = cwd / dir_path
        status_data["directories"][dir_path] = {
            "description": desc,
            "exists": full_path.exists(),
        }

    config_files = [
        ("pyproject.toml", "Project configuration"),
        ("environment.yml", "Conda environment"),
        ("requirements.txt", "Python dependencies"),
        ("docker-compose.yml", "Docker configuration"),
    ]

    for file_path, desc in config_files:
        full_path = cwd / file_path
        status_data["config_files"][file_path] = {
            "description": desc,
            "exists": full_path.exists(),
        }

    if json_output:
        print(json.dumps(status_data, indent=2))
        return

    _print_banner()
    print(f"{_colorize('ENVIRONMENT STATUS:', 'bold')}\n")

    print(f"  {_colorize('Environment Variables:', 'cyan')}")
    for var, val in status_data["environment"].items():
        # Re-apply masking logic for display validity check (already masked in data but checking presence)
        is_set = val != NOT_SET
        status_icon = _colorize("✓", "green") if is_set else _colorize("○", "yellow")
        display_val = val if len(val) < 50 else val[:47] + "..."
        print(f"    {status_icon} {var}: {display_val}")

    # Check directories
    print(f"\n  {_colorize('Directory Structure:', 'cyan')}")
    for dir_path, info in status_data["directories"].items():
        status_icon = (
            _colorize("✓", "green") if info["exists"] else _colorize("✗", "red")
        )
        print(f"    {status_icon} {info['description']}: {dir_path}")

    # Check config files
    print(f"\n  {_colorize('Configuration Files:', 'cyan')}")
    for file_path, info in status_data["config_files"].items():
        status_icon = (
            _colorize("✓", "green") if info["exists"] else _colorize("○", "dim")
        )
        print(f"    {status_icon} {info['description']}: {file_path}")

    print(
        f"\n{_colorize('TIP:', 'yellow')} Run {_colorize('cortex doctor', 'cyan')} for detailed diagnostics\n"
    )


def _show_config(
    validate: bool = False, export_format: str | None = None, section: str | None = None
) -> None:
    """Show or validate configuration."""
    _print_banner()

    try:
        from cortex.config.loader import (  # type: ignore[import]
            get_config,
        )

        config = cast(EmailOpsConfigProto, get_config())

        if validate:
            print(f"{_colorize('Configuration Validation:', 'bold')}\n")
            print(f"  {_colorize('✓', 'green')} Configuration loaded successfully")
            print(f"  {_colorize('✓', 'green')} All required fields present")
            print(f"  {_colorize('✓', 'green')} Pydantic validation passed\n")

            print(f"  {_colorize('Summary:', 'cyan')}")
            print(f"    Environment: {config.core.env}")
            print(f"    Provider:    {config.core.provider}")
            print(f"    Log Level:   {config.system.log_level}")
            print()
        elif export_format == "json":
            _print_json_config(config, section)
        else:
            _print_human_config(config, section)

    except ImportError as e:
        print(f"{_colorize('ERROR:', 'red')} Could not load configuration module")
        print(f"  {e}")
        print(
            f"\n  Run {_colorize('cortex doctor --auto-install', 'cyan')} to fix dependencies"
        )
        sys.exit(1)
    except Exception as e:
        print(f"{_colorize('ERROR:', 'red')} Configuration validation failed")
        print(f"  {e}")
        sys.exit(1)


# =============================================================================
# CORE COMMANDS: ingest, index, search, validate
# =============================================================================


def _run_validate(
    path: str,
    json_output: bool = False,
) -> None:
    """
    Validate export folder structure (B1).
    """
    import json

    target_path = Path(path).resolve()

    if not json_output:
        _print_banner()
        print(f"{_colorize('▶ EXPORT VALIDATION (B1)', 'bold')}\n")
        print(f"  Target:    {_colorize(str(target_path), 'cyan')}")
        print()

    if not target_path.exists():
        msg = f"Path does not exist: {target_path}"
        if json_output:
            print(json.dumps({"error": msg, "success": False}))
        else:
            print(f"  {_colorize('✗', 'red')} {msg}")
        sys.exit(1)

    try:
        from cortex.ingestion.conv_manifest.validation import (
            scan_and_refresh as _scan_and_refresh,  # type: ignore[import]; type: ignore[reportUnknownVariableType]
        )

        _scan_and_refresh = cast(Any, _scan_and_refresh)
        scan_and_refresh: Callable[[Path], Any]
        scan_and_refresh = cast(Callable[[Path], Any], _scan_and_refresh)

        if not json_output:
            print(f"  {_colorize('⏳', 'yellow')} Scanning and validating...")

        report: Any = scan_and_refresh(target_path)

        if json_output:
            print(report.model_dump_json())
        else:
            print(f"\n  {_colorize('Results:', 'cyan')}")
            print(f"    Folders Scanned:   {report.folders_scanned}")
            print(f"    Manifests Created: {report.manifests_created}")
            print(f"    Manifests Updated: {report.manifests_updated}")

            if report.problems:
                print(f"\n  {_colorize('Problems Found:', 'red')}")
                for p in report.problems:
                    print(f"    • {p.folder}: {p.issue}")
            else:
                print(f"\n  {_colorize('✓', 'green')} No problems found.")
            print()

    except ImportError as e:
        msg = f"Could not load validation module: {e}"
        if json_output:
            print(json.dumps({"error": msg, "success": False}))
        else:
            print(f"\n  {_colorize('ERROR:', 'red')} {msg}")
        sys.exit(1)
    except Exception as e:
        if json_output:
            print(json.dumps({"error": str(e), "success": False}))
        else:
            print(f"\n  {_colorize('ERROR:', 'red')} {e}")
        sys.exit(1)


def _run_pipeline(args: argparse.Namespace) -> None:
    """Wrapper for pipeline command."""
    from cortex_cli.cmd_pipeline import cmd_pipeline_run

    cmd_pipeline_run(
        source_prefix=args.source,
        tenant_id=args.tenant,
        limit=args.limit,
        concurrency=args.concurrency,
        auto_embed=args.auto_embed,
        dry_run=args.dry_run,
        verbose=args.verbose,
        json_output=args.json,
    )


def _run_ingest(
    source_path: str,
    tenant_id: str = "default",
    dry_run: bool = False,
    verbose: bool = False,
    json_output: bool = False,
) -> None:
    """
    Ingest email exports into the system.

    Processes conversation folders containing:
    - Conversation.txt (email transcript)
    - manifest.json (metadata)
    - attachments/ (extracted files)
    """
    import json
    import uuid

    source = Path(source_path).resolve()

    if not json_output:
        _print_banner()
        print(f"{_colorize('▶ EMAIL INGESTION', 'bold')}\n")
        print(f"  Source:    {_colorize(str(source), 'cyan')}")
        print(f"  Tenant:    {_colorize(tenant_id, 'cyan')}")
        print(f"  Dry Run:   {_colorize(str(dry_run), 'yellow' if dry_run else 'dim')}")
        print()

    if not source.exists():
        msg = f"Source path does not exist: {source}"
        if json_output:
            print(json.dumps({"error": msg, "success": False}))
        else:
            print(f"  {_colorize('✗', 'red')} {msg}")
        sys.exit(1)

    # Discover conversation folders
    conversations: list[Path] = []
    if source.is_dir():
        # Check if source IS a conversation folder
        if (source / "Conversation.txt").exists() or (
            source / "manifest.json"
        ).exists():
            conversations = [source]
        else:
            # Scan for conversation subfolders
            for item in source.iterdir():
                if item.is_dir() and (
                    (item / "Conversation.txt").exists()
                    or (item / "manifest.json").exists()
                ):
                    conversations.append(item)

    if not conversations:
        msg = f"No conversation folders found in: {source}"
        if json_output:
            print(
                json.dumps({"error": msg, "success": False, "conversations_found": 0})
            )
        else:
            print(f"  {_colorize('⚠', 'yellow')} {msg}")
            print(f"\n  {_colorize('Expected structure:', 'dim')}")
            print(f"    {source}/")
            print("      ├── Conversation.txt")
            print("      ├── manifest.json")
            print("      └── attachments/")
        sys.exit(1)

    if not json_output:
        print(
            f"  {_colorize('✓', 'green')} Found {len(conversations)} conversation(s)\n"
        )

    if dry_run:
        if not json_output:
            print(f"{_colorize('DRY RUN - No changes will be made', 'yellow')}\n")
            for i, conv in enumerate(conversations[:10], 1):
                print(f"    {i}. {conv.name}")
            if len(conversations) > 10:
                print(f"    ... and {len(conversations) - 10} more")
            print()
        else:
            print(
                json.dumps(
                    {
                        "success": True,
                        "dry_run": True,
                        "conversations_found": len(conversations),
                        "conversations": [str(c) for c in conversations[:20]],
                    }
                )
            )
        return

    # Actually run ingestion
    try:
        from cortex.ingestion.mailroom import (
            IngestJob as _IngestJob,  # type: ignore[import]; type: ignore[reportUnknownVariableType]
        )
        from cortex.ingestion.mailroom import (
            process_job as _process_job,  # type: ignore[reportUnknownVariableType]
        )

        _IngestJob = cast(Any, _IngestJob)
        _process_job = cast(Any, _process_job)
        IngestJob: type[Any]
        process_job: Callable[[Any], Any]
        IngestJob = cast(type[Any], _IngestJob)
        process_job = cast(Callable[[Any], Any], _process_job)

        results: list[Any] = []
        success_count = 0
        fail_count = 0

        for i, conv in enumerate(conversations, 1):
            if not json_output:
                print(
                    f"  [{i}/{len(conversations)}] Processing: {conv.name}...",
                    end=" ",
                    flush=True,
                )

            job: Any = IngestJob(
                job_id=uuid.uuid4(),
                tenant_id=tenant_id,
                source_type="local_upload",
                source_uri=str(conv),
            )

            try:
                summary: Any = process_job(job)
                if summary.aborted_reason:
                    fail_count += 1
                    if not json_output:
                        print(f"{_colorize('FAILED', 'red')}")
                        if verbose:
                            print(f"       Reason: {summary.aborted_reason}")
                else:
                    success_count += 1
                    if not json_output:
                        print(
                            f"{_colorize('OK', 'green')} ({summary.messages_ingested} msg, {summary.attachments_parsed} att)"
                        )

                results.append(
                    {
                        "path": str(conv),
                        "success": not summary.aborted_reason,
                        "messages": summary.messages_ingested,
                        "attachments": summary.attachments_parsed,
                        "error": summary.aborted_reason,
                    }
                )
            except Exception as e:
                fail_count += 1
                if not json_output:
                    print(f"{_colorize('ERROR', 'red')}")
                    if verbose:
                        print(f"       {e}")
                results.append(
                    {
                        "path": str(conv),
                        "success": False,
                        "error": str(e),
                    }
                )

        if json_output:
            print(
                json.dumps(
                    {
                        "success": fail_count == 0,
                        "total": len(conversations),
                        "succeeded": success_count,
                        "failed": fail_count,
                        "results": results,
                    }
                )
            )
        else:
            print()
            print(f"{_colorize('═' * 50, 'cyan')}")
            if fail_count == 0:
                print(
                    f"\n  {_colorize('✓', 'green')} All {success_count} conversation(s) ingested successfully!"
                )
            else:
                print(f"\n  {_colorize('⚠', 'yellow')} Completed with errors:")
                print(f"    Succeeded: {_colorize(str(success_count), 'green')}")
                print(f"    Failed:    {_colorize(str(fail_count), 'red')}")
            print()

    except ImportError as e:
        msg = f"Could not load ingestion module: {e}"
        if json_output:
            print(json.dumps({"error": msg, "success": False}))
        else:
            print(f"\n  {_colorize('ERROR:', 'red')} {msg}")
            print(f"  Run {_colorize('cortex doctor --auto-install', 'cyan')} first")
        sys.exit(1)


def _run_index(
    root: str = ".",
    provider: str = "digitalocean",
    workers: int = 4,
    limit: int | None = None,
    force: bool = False,
    json_output: bool = False,
) -> None:
    """
    Build or rebuild the search index with embeddings.
    """
    import json

    root_path = Path(root).resolve()

    if not json_output:
        _print_banner()
        print(f"{_colorize('▶ INDEX BUILDER', 'bold')}\n")
        print(f"  Root:      {_colorize(str(root_path), 'cyan')}")
        print(f"  Provider:  {_colorize(provider, 'cyan')}")
        print(f"  Workers:   {_colorize(str(workers), 'cyan')}")
        if limit:
            print(f"  Limit:     {_colorize(str(limit), 'yellow')}")
        if force:
            print(
                f"  {_colorize('Force: recomputing index regardless of cache state', 'yellow')}"
            )
        print()

    try:
        from cortex_workers.reindex_jobs.parallel_indexer import (
            parallel_index_conversations as _parallel_index_conversations,  # type: ignore[import]; type: ignore[reportUnknownVariableType]
        )

        _parallel_index_conversations = cast(Any, _parallel_index_conversations)
        parallel_index_conversations: Callable[..., tuple[Any, Any]]
        parallel_index_conversations = cast(
            Callable[..., tuple[Any, Any]], _parallel_index_conversations
        )

        if not json_output:
            print(f"  {_colorize('⏳', 'yellow')} Starting parallel indexing...")
            print()

        embeddings: Any
        mappings: Any

        embeddings, mappings = parallel_index_conversations(
            root=root_path,
            provider=provider,
            num_workers=workers,
            limit=limit,
            force=force,
        )

        num_chunks = len(mappings)
        embedding_dim = embeddings.shape[1] if embeddings.size > 0 else 0

        if json_output:
            print(
                json.dumps(
                    {
                        "success": True,
                        "chunks_indexed": num_chunks,
                        "embedding_dimension": embedding_dim,
                        "provider": provider,
                    }
                )
            )
        else:
            print(f"{_colorize('═' * 50, 'cyan')}")
            if num_chunks > 0:
                print(f"\n  {_colorize('✓', 'green')} Indexing complete!")
                print(f"    Chunks indexed:  {_colorize(str(num_chunks), 'bold')}")
                print(f"    Embedding dim:   {_colorize(str(embedding_dim), 'dim')}")
            else:
                print(f"\n  {_colorize('⚠', 'yellow')} No conversations found to index")
            print()

    except ImportError as e:
        msg = f"Could not load indexer module: {e}"
        if json_output:
            print(json.dumps({"error": msg, "success": False}))
        else:
            print(f"\n  {_colorize('ERROR:', 'red')} {msg}")
        sys.exit(1)
    except Exception as e:
        if json_output:
            print(json.dumps({"error": str(e), "success": False}))
        else:
            print(f"\n  {_colorize('ERROR:', 'red')} {e}")
        sys.exit(1)


# =============================================================================
# RAG COMMANDS: answer, draft, summarize
# =============================================================================


def _run_answer(
    query: str,
    tenant_id: str = "default",
    user_id: str = "cli-user",
    json_output: bool = False,
) -> None:
    """
    Ask questions about your emails using RAG.
    """
    import json
    from cortex_cli.api_client import get_api_client

    if not json_output:
        _print_banner()
        print(f"{_colorize('▶ ASK CORTEX', 'bold')}\n")
        print(f"  Query:   {_colorize(query, 'cyan')}")
        print()

    try:
        if not json_output:
            print(f"  {_colorize('⏳', 'yellow')} Thinking...")

        api_client = get_api_client()
        response = api_client.answer(query=query, tenant_id=tenant_id, user_id=user_id)

        answer = response.get("answer")

        if json_output:
            print(json.dumps(answer if answer else {}, indent=2, default=str))
        else:
            if answer:
                print(f"\n{_colorize('ANSWER:', 'bold')}")
                print(f"{answer.get('answer_markdown', '')}\n")

                evidence = answer.get("evidence", [])
                if evidence:
                    print(f"{_colorize('SOURCES:', 'dim')}")
                    for i, ev in enumerate(evidence, 1):
                        snippet = ev.get("snippet") or ev.get("text") or ""
                        print(f"  {i}. {snippet}")
            else:
                print(f"  {_colorize('⚠', 'yellow')} No answer generated.")
            print()

    except ImportError as e:
        msg = f"Could not load RAG module: {e}"
        if json_output:
            print(json.dumps({"error": msg, "success": False}))
        else:
            print(f"\n  {_colorize('ERROR:', 'red')} {msg}")
        sys.exit(1)
    except Exception as e:
        if json_output:
            print(json.dumps({"error": str(e), "success": False}))
        else:
            print(f"\n  {_colorize('ERROR:', 'red')} {e}")
        sys.exit(1)


def _run_summarize(
    thread_id: str,
    tenant_id: str = "default",
    user_id: str = "cli-user",
    json_output: bool = False,
) -> None:
    """
    Summarize an email thread.
    """
    import asyncio
    import json

    if not json_output:
        _print_banner()
        print(f"{_colorize('▶ SUMMARIZE THREAD', 'bold')}\n")
        print(f"  Thread ID: {_colorize(thread_id, 'cyan')}")
        print()

    try:
        from cortex.orchestration.graphs import (
            build_summarize_graph as _build_summarize_graph,  # type: ignore[import]; type: ignore[reportUnknownVariableType]
        )

        _build_summarize_graph = cast(Any, _build_summarize_graph)
        build_summarize_graph: Callable[[], Any]
        build_summarize_graph = cast(Callable[[], Any], _build_summarize_graph)

        if not json_output:
            print(f"  {_colorize('⏳', 'yellow')} Summarizing...")

        async def _execute() -> Any:
            graph = build_summarize_graph().compile()
            initial_state: dict[str, Any] = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "thread_id": thread_id,
                "thread_context": None,
                "facts_ledger": None,
                "critique": None,
                "iteration_count": 0,
                "summary": None,
                "error": None,
            }
            result = await graph.ainvoke(initial_state)
            return result

        final_state: Any = asyncio.run(_execute())

        # Handle both dict and object-like state access
        error = (
            final_state.get("error")
            if isinstance(final_state, dict)
            else getattr(final_state, "error", None)
        )
        if error:
            raise Exception(error)

        summary: Any | None = (
            final_state.get("summary")
            if isinstance(final_state, dict)
            else getattr(final_state, "summary", None)
        )

        if json_output:
            print(
                json.dumps(
                    summary.model_dump() if summary else {}, indent=2, default=str
                )
            )
        else:
            if summary:
                print(f"\n{_colorize('SUMMARY:', 'bold')}")
                print(f"{summary.summary_markdown}\n")

                if summary.facts_ledger:
                    print(f"{_colorize('FACTS LEDGER:', 'dim')}")
                    for key, value in summary.facts_ledger.items():
                        print(f"  • {key}: {value}")
            else:
                print(f"  {_colorize('⚠', 'yellow')} No summary generated.")
            print()

    except ImportError as e:
        msg = f"Could not load RAG module: {e}"
        if json_output:
            print(json.dumps({"error": msg, "success": False}))
        else:
            print(f"\n  {_colorize('ERROR:', 'red')} {msg}")
        sys.exit(1)
    except Exception as e:
        if json_output:
            print(json.dumps({"error": str(e), "success": False}))
        else:
            print(f"\n  {_colorize('ERROR:', 'red')} {e}")
        sys.exit(1)


def main(args: list[str] | None = None) -> None:
    """
    Main entry point for the Cortex CLI.

    A user-friendly command-line interface for managing EmailOps Cortex.
    """
    # Initialize observability §12.3
    # This should be one of the first things to run
    try:
        from cortex.observability import init_observability

        init_observability(service_name="cortex-cli")
    except ImportError:
        # If cortex backend is not installed, CLI should still function
        pass
    if args is None:
        args = sys.argv[1:]

    # Launch interactive mode if no arguments provided
    if not args:
        try:
            from cortex_cli.tui import interactive_loop

            interactive_loop()
            return
        except ImportError:
            # If rich/questionary not installed, fall back to help
            pass

    parser = argparse.ArgumentParser(
        prog="cortex",
        description="EmailOps Cortex CLI - Email Intelligence Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cortex doctor                    Run basic system diagnostics
  cortex doctor --check-embeddings Test embedding connectivity
  cortex doctor --auto-install     Auto-install missing dependencies
  cortex status                    View current environment status
  cortex config --validate         Validate configuration files

For more information, see docs/CANONICAL_BLUEPRINT.md
        """,
        add_help=False,
    )

    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show this help message",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Show version information",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="Commands",
        description="Available commands (use 'cortex <command> --help' for details)",
        metavar="<command>",
    )

    # Setup command groups
    _setup_core_commands(subparsers)
    _setup_rag_commands(subparsers)
    _setup_utility_commands(subparsers)

    # Register plugin subcommand groups
    import typer
    from cortex_cli.cmd_backfill import setup_backfill_parser
    from cortex_cli.cmd_db import setup_db_parser
    from cortex_cli.cmd_embeddings import setup_embeddings_parser
    from cortex_cli.cmd_fix import setup_fix_parser
    from cortex_cli.cmd_graph import app as graph_app
    from cortex_cli.cmd_grounding import setup_grounding_parser
    from cortex_cli.cmd_login import setup_login_parser
    from cortex_cli.cmd_maintenance import setup_maintenance_parser
    from cortex_cli.cmd_queue import setup_queue_parser
    from cortex_cli.cmd_s3 import setup_s3_parser
    from cortex_cli.cmd_safety import setup_safety_parser
    from cortex_cli.cmd_test import setup_test_parser
    from cortex_cli.cmd_search import setup_search_parser
    from rich.console import Console
    from typer.core import TyperGroup
    from typer.main import get_command_from_info

    # A bit of a hack to integrate Typer apps with argparse
    def setup_typer_command(subparsers, name, app, help_text=""):
        parser = subparsers.add_parser(name, help=help_text, add_help=False)
        command_info = typer.main.get_command_info(
            app,
            name=name,
            pretty_exceptions_short=False,
            pretty_exceptions_show_locals=False,
            rich_markup_mode="rich",
        )
        command = get_command_from_info(
            command_info,
            pretty_exceptions_short=False,
            pretty_exceptions_show_locals=False,
            rich_markup_mode="rich",
        )

        def _run_typer(args):
            try:
                if isinstance(command, TyperGroup):
                    command(args.typer_args, standalone_mode=False)
                else:
                    command(standalone_mode=False)

            except typer.Exit as e:
                if e.code != 0:
                    console = Console()
                    console.print(f"[bold red]Error:[/bold red] {e}")
                # Do not exit process
            except Exception as e:
                console = Console()
                console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")

        parser.set_defaults(func=lambda args: _run_typer(args))
        # This is a simple way to pass through args. A more robust solution might be needed.
        parser.add_argument("typer_args", nargs="*")

    from cortex_cli.cmd_index import setup_index_parser
    from cortex_cli.cmd_patch import setup_patch_parser
    from cortex_cli.cmd_schema import setup_schema_parser
    from cortex_cli.cmd_test import setup_test_parser
    from cortex_cli._config_helpers import _config

    setup_backfill_parser(subparsers)
    setup_db_parser(subparsers)
    setup_embeddings_parser(subparsers)
    setup_s3_parser(subparsers)
    setup_maintenance_parser(subparsers)
    setup_test_parser(subparsers)
    setup_search_parser(subparsers)
    setup_grounding_parser(subparsers)
    setup_safety_parser(subparsers)
    setup_queue_parser(subparsers)
    setup_login_parser(subparsers)
    setup_typer_command(
        subparsers, "graph", graph_app, help_text="Knowledge Graph commands"
    )
    setup_patch_parser(subparsers)
    setup_index_parser(subparsers)
    setup_schema_parser(subparsers)
    if "sqlite" not in _config.database.url:
        setup_test_parser(subparsers)

    # Parse arguments
    parsed_args = parser.parse_args(args)

    # Handle top-level flags
    if parsed_args.version:
        _print_version()
        return

    if parsed_args.help or parsed_args.command is None:
        _print_usage()
        return

    # Route to appropriate command
    if hasattr(parsed_args, "func"):
        parsed_args.func(parsed_args)
    else:
        _print_usage()
        sys.exit(1)


def _setup_core_commands(subparsers: Any) -> None:
    """Setup core CLI commands: ingest, index, search, validate."""
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Process and ingest email exports",
        description="""
Ingest email exports into the EmailOps system.

Processes conversation folders containing:
  • Conversation.txt - Email transcript
  • manifest.json - Metadata (dates, subject, participants)
  • attachments/ - Extracted attachment files

The ingestion pipeline:
  1. Validates and parses conversation data
  2. Extracts text from attachments (PDF, Word, etc.)
  3. Detects and masks PII
  4. Chunks text for embedding
  5. Generates embeddings
  6. Writes to database
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ingest_parser.add_argument(
        "source",
        metavar="PATH",
        help="Path to email export folder(s)",
    )
    ingest_parser.add_argument(
        "--tenant",
        "-t",
        default="default",
        metavar="ID",
        help="Tenant ID for multi-tenant isolation (default: 'default')",
    )
    ingest_parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Scan and validate without making changes",
    )
    ingest_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress and errors",
    )
    ingest_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    ingest_parser.set_defaults(
        func=lambda args: _run_ingest(
            source_path=args.source,
            tenant_id=args.tenant,
            dry_run=args.dry_run,
            verbose=args.verbose,
            json_output=args.json,
        )
    )

    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the unified ingestion pipeline",
        description="""
Run the end-to-end Cortex pipeline:
  1. Discovery: Scans source (S3/Local) for conversation folders.
  2. Ingestion: Atomic batch processing (Validate -> Ingest).
  3. Embedding: (Optional) Triggers vector generation.

Examples:
  cortex pipeline --source "Outlook/Inbox" --tenant "acme" --auto-embed
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pipeline_parser.add_argument(
        "--source",
        default="Outlook/",
        help="S3 prefix or local path to scan (default: Outlook/)",
    )
    pipeline_parser.add_argument(
        "--tenant",
        default="default",
        help="Tenant ID to associate with data",
    )
    pipeline_parser.add_argument(
        "--limit",
        type=int,
        help="Max folders to process",
    )
    pipeline_parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    pipeline_parser.add_argument(
        "--auto-embed",
        action="store_true",
        help="Trigger embedding generation after ingestion",
    )
    pipeline_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be processed without making changes",
    )
    pipeline_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose/debug output",
    )
    pipeline_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    pipeline_parser.set_defaults(func=lambda args: _run_pipeline(args))

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate export folder structure (B1)",
        description="""
Validate email export folders against the B1 manifest specification.

This command:
  1. Scans the target directory for conversation folders
  2. Verifies/Creates manifest.json files
  3. Calculates SHA256 hashes for integrity
  4. Reports any structural issues
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    validate_parser.add_argument(
        "path",
        metavar="PATH",
        help="Path to export root or conversation folder",
    )
    validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    validate_parser.set_defaults(
        func=lambda args: _run_validate(path=args.path, json_output=args.json)
    )


from cortex_cli.cmd_draft import setup_draft_parser


def _setup_rag_commands(subparsers: Any) -> None:
    """Setup RAG CLI commands: answer, draft, summarize."""
    # Answer command
    answer_parser = subparsers.add_parser(
        "answer",
        help="Ask questions about your emails",
        description="""
Ask questions about your emails using the RAG pipeline.

The system will:
  1. Search for relevant emails
  2. Analyze the context
  3. Generate a grounded answer with citations
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    answer_parser.add_argument(
        "query",
        metavar="QUESTION",
        help="The question to ask",
    )
    answer_parser.add_argument(
        "--tenant",
        "-t",
        default="default",
        metavar="ID",
        help="Tenant ID (default: 'default')",
    )
    answer_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    answer_parser.set_defaults(
        func=lambda args: _run_answer(
            query=args.query, tenant_id=args.tenant, json_output=args.json
        )
    )

    # Draft command
    setup_draft_parser(subparsers)

    # Summarize command
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize email threads",
        description="""
Generate a concise summary of an email thread.

The system will:
  1. Retrieve the full thread history
  2. Extract key facts and decisions
  3. Produce a structured summary
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    summarize_parser.add_argument(
        "thread_id",
        metavar="UUID",
        help="ID of the thread to summarize",
    )
    summarize_parser.add_argument(
        "--tenant",
        "-t",
        default="default",
        metavar="ID",
        help="Tenant ID (default: 'default')",
    )
    summarize_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    summarize_parser.set_defaults(
        func=lambda args: _run_summarize(
            thread_id=args.thread_id, tenant_id=args.tenant, json_output=args.json
        )
    )


# Boolean flags that map directly to --flag-name
_DOCTOR_BOOL_FLAGS = [
    ("auto_install", "--auto-install"),
    ("check_index", "--check-index"),
    ("check_embeddings", "--check-embeddings"),
    ("check_db", "--check-db"),
    ("check_redis", "--check-redis"),
    ("check_exports", "--check-exports"),
    ("check_ingest", "--check-ingest"),
    ("json", "--json"),
    ("verbose", "--verbose"),
]


def _handle_doctor(args: argparse.Namespace) -> None:
    """Handle doctor command by forwarding args to cmd_doctor.main()."""
    from cortex_cli.cmd_doctor import main as doctor_main

    # Handle --all flag
    if getattr(args, "check_all", False):
        args.check_index = True
        args.check_embeddings = True

    # Build args list using lookup table
    doctor_args: list[str] = []
    if args.root != ".":
        doctor_args.extend(["--root", args.root])
    if args.provider != "vertex":
        doctor_args.extend(["--provider", args.provider])

    for attr, flag in _DOCTOR_BOOL_FLAGS:
        if getattr(args, attr, False):
            doctor_args.append(flag)

    if args.pip_timeout != 300:
        doctor_args.extend(["--pip-timeout", str(args.pip_timeout)])

    sys.argv = [sys.argv[0], *doctor_args]
    doctor_main()


def _setup_utility_commands(subparsers: Any) -> None:
    """Setup utility CLI commands: doctor, status, config, version."""
    # Doctor command
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run system diagnostics and health checks",
        description="""
Run comprehensive system diagnostics including:
  • Dependency checks for your chosen provider
  • Index health verification
  • Embedding connectivity tests
  • Configuration validation
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    doctor_parser.add_argument(
        "--root",
        default=".",
        metavar="DIR",
        help="Project root directory (default: current directory)",
    )
    doctor_parser.add_argument(
        "--provider",
        default="digitalocean",
        choices=[
            "vertex",
            "gcp",
            "vertexai",
            "hf",
            "openai",
            "cohere",
            "huggingface",
            "qwen",
            "local",
        ],
        metavar="PROVIDER",
        help=(
            "Embedding provider to check (default: vertex). "
            "Options: vertex, gcp, vertexai, hf, openai, cohere, huggingface, qwen, local"
        ),
    )
    doctor_parser.add_argument(
        "--auto-install",
        action="store_true",
        help="Automatically install missing Python packages",
    )
    doctor_parser.add_argument(
        "--check-index",
        action="store_true",
        help="Verify index health and integrity",
    )
    doctor_parser.add_argument(
        "--check-embeddings",
        action="store_true",
        help="Test embedding API connectivity",
    )
    doctor_parser.add_argument(
        "--check-db",
        action="store_true",
        help="Check database connectivity and migrations",
    )
    doctor_parser.add_argument(
        "--check-redis",
        action="store_true",
        help="Check Redis connectivity",
    )
    doctor_parser.add_argument(
        "--check-exports",
        action="store_true",
        help="Verify export root and list B1 folders",
    )
    doctor_parser.add_argument(
        "--check-ingest",
        action="store_true",
        help="Dry-run ingest of sample data",
    )
    doctor_parser.add_argument(
        "--all",
        action="store_true",
        dest="check_all",
        help="Run all checks (index + embeddings)",
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for CI/CD integration)",
    )
    doctor_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    doctor_parser.add_argument(
        "--pip-timeout",
        type=int,
        default=300,
        metavar="SECONDS",
        help="Timeout for pip install operations (default: 300)",
    )

    doctor_parser.set_defaults(func=_handle_doctor)

    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show current environment and configuration status",
        description="Display an overview of your EmailOps Cortex environment.",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    status_parser.set_defaults(func=lambda args: _show_status(json_output=args.json))

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="View, validate, or export configuration",
        description="Manage EmailOps Cortex configuration settings.",
    )
    config_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration and report issues",
    )
    config_parser.add_argument(
        "--json",
        action="store_true",
        help="Export configuration as JSON",
    )
    config_parser.add_argument(
        "--section",
        help="View specific configuration section (e.g. core, search)",
    )
    config_parser.set_defaults(
        func=lambda args: _show_config(
            validate=args.validate,
            export_format="json" if args.json else None,
            section=args.section,
        )
    )

    # Version command
    version_parser = subparsers.add_parser(
        "version",
        help="Display version information",
    )
    version_parser.set_defaults(func=lambda _: _print_version())

    autofix_parser = subparsers.add_parser(
        "autofix",
        help="Automatically fix common code issues",
        description="Run the auto-fix script to resolve low-hanging fruit issues.",
    )
    autofix_parser.set_defaults(func=lambda _: _run_autofix())


def _run_autofix():
    """Run the autofix script."""
    from cortex_cli.cmd_autofix import main as autofix_main

    autofix_main()


if __name__ == "__main__":
    main()
