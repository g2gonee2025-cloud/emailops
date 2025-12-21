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
    cortex index --provider vertex
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

    def model_dump(self) -> dict[str, Any]:
        ...


# Import config model for typing if available; otherwise fall back to protocol
if TYPE_CHECKING:
    # Use the concrete config class for type checking; fallback to the protocol at runtime.
    from cortex.config.loader import EmailOpsConfig as EmailOpsConfig
else:
    EmailOpsConfig = EmailOpsConfigProto

# Lazy import for heavy dependencies
# from cortex_cli.cmd_doctor import main as doctor_main

# ANSI color codes for terminal output
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


def _colorize(text: str, color: str) -> str:
    """Apply ANSI color to text if terminal supports it."""
    if not sys.stdout.isatty():
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def _print_banner() -> None:
    """Print the CLI banner."""
    banner = f"""
{_colorize("╔═══════════════════════════════════════════════════════════╗", "cyan")}
{_colorize("║", "cyan")}  {_colorize("EmailOps Cortex CLI", "bold")} - Email Intelligence Platform       {_colorize("║", "cyan")}
{_colorize("║", "cyan")}  Powered by Vertex AI & LangGraph                          {_colorize("║", "cyan")}
{_colorize("╚═══════════════════════════════════════════════════════════╝", "cyan")}
"""
    print(banner)


def _print_usage() -> None:
    """Print user-friendly usage information."""
    _print_banner()

    print(f"{_colorize('USAGE:', 'bold')}")
    print(f"    cortex {_colorize('<command>', 'cyan')} [options]\n")

    print(
        f"{_colorize('CORE COMMANDS:', 'bold')}  {_colorize('(Email Processing)', 'dim')}"
    )
    core_commands = [
        ("ingest", "Process and ingest email exports into the system"),
        ("index", "Build/rebuild search index with embeddings"),
        ("search", "Search indexed emails with natural language"),
        ("validate", "Validate export folder structure (B1)"),
    ]
    for cmd, desc in core_commands:
        print(f"    {_colorize(cmd, 'green'):12} {desc}")

    print(
        f"\n{_colorize('RAG COMMANDS:', 'bold')}   {_colorize('(AI Capabilities)', 'dim')}"
    )
    rag_commands = [
        ("answer", "Ask questions about your emails"),
        ("draft", "Draft email replies based on context"),
        ("summarize", "Summarize email threads"),
    ]
    for cmd, desc in rag_commands:
        print(f"    {_colorize(cmd, 'green'):12} {desc}")

    print(f"\n{_colorize('UTILITY COMMANDS:', 'bold')}")
    utility_commands = [
        ("doctor", "Run system diagnostics and health checks"),
        ("status", "Show current environment and configuration"),
        ("config", "View, validate, or export configuration"),
        ("version", "Display version information"),
    ]
    for cmd, desc in utility_commands:
        print(f"    {_colorize(cmd, 'green'):12} {desc}")

    print(f"\n{_colorize('DATA COMMANDS:', 'bold')}")
    data_commands = [
        ("db", "Database management (stats, migrate)"),
        ("embeddings", "Embedding management (stats, backfill)"),
        ("s3", "S3/Spaces storage (list, ingest)"),
    ]
    for cmd, desc in data_commands:
        print(f"    {_colorize(cmd, 'green'):12} {desc}")

    print(f"\n{_colorize('COMMON OPTIONS:', 'bold')}")
    options = [
        ("--help, -h", "Show help for a command"),
        ("--verbose, -v", "Enable verbose output (where supported)"),
        ("--json", "Output in JSON format (where supported)"),
    ]
    for opt, desc in options:
        print(f"    {_colorize(opt, 'yellow'):16} {desc}")

    print(f"\n{_colorize('EXAMPLES:', 'bold')}")
    examples = [
        ("cortex ingest ./exports/emails", "Ingest emails from a folder"),
        ("cortex ingest ./my_export --tenant acme", "Ingest with tenant ID"),
        ("cortex index --workers 4", "Build index with 4 workers"),
        ('cortex search "contract terms"', "Search for contract terms"),
        ("cortex doctor --check-embeddings", "Test embedding connectivity"),
    ]
    for example, desc in examples:
        print(f"    {_colorize(example, 'dim'):44} # {desc}")

    print(f"\n{_colorize('WORKFLOW:', 'bold')}")
    print(f"    1. {_colorize('cortex doctor', 'cyan')} → Check system health")
    print(f"    2. {_colorize('cortex ingest <path>', 'cyan')} → Import email exports")
    print(f"    3. {_colorize('cortex index', 'cyan')} → Build search embeddings")
    print(f"    4. {_colorize('cortex search <query>', 'cyan')} → Query your emails")
    print(f"    5. {_colorize('cortex answer <question>', 'cyan')} → Get AI answers")

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
            "OUTLOOKCORTEX_ENV": os.getenv("OUTLOOKCORTEX_ENV", "not set"),
            "OUTLOOKCORTEX_DB_URL": (
                "***" if os.getenv("OUTLOOKCORTEX_DB_URL") else "not set"
            ),
            "GOOGLE_APPLICATION_CREDENTIALS": os.getenv(
                "GOOGLE_APPLICATION_CREDENTIALS", "not set"
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
        is_set = val != "not set"
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
            import json

            data = config.model_dump()
            if section:
                if section in data:
                    data = {section: data[section]}
                else:
                    print(f"{_colorize('ERROR:', 'red')} Section '{section}' not found")
                    return

            print(json.dumps(data, indent=2, default=str))
        else:
            title = (
                f"Current Configuration ({section})"
                if section
                else "Current Configuration"
            )
            print(f"{_colorize(f'{title}:', 'bold')}\n")

            sections: list[tuple[str, list[tuple[str, object]]]] = [
                (
                    "Core",
                    [
                        ("Environment", config.core.env),
                        ("Provider", config.core.provider),
                        ("Persona", config.core.persona),
                    ],
                ),
                (
                    "Embeddings",
                    [
                        ("Model", config.embedding.model_name),
                        ("Dimensions", config.embedding.output_dimensionality),
                        ("Batch Size", config.embedding.batch_size),
                    ],
                ),
                (
                    "Search",
                    [
                        ("Strategy", config.search.fusion_strategy),
                        ("K", config.search.k),
                        ("Recency", config.search.recency_boost_strength),
                        ("Reranker", config.search.reranker_endpoint),
                    ],
                ),
                (
                    "Processing",
                    [
                        ("Chunk Size", config.processing.chunk_size),
                        ("Chunk Overlap", config.processing.chunk_overlap),
                    ],
                ),
            ]

            # Mapping from display names to config attributes/keys
            section_map = {
                "DigitalOcean LLM": "digitalocean_llm",
                # Standard attributes
                "Core": "core",
                "Embeddings": "embedding",
                "Search": "search",
                "Processing": "processing",
                "Database": "database",
                "Storage": "storage",
                "GCP": "gcp",
                "Retry": "retry",
                "Limits": "limits",
            }

            # Simple fallback for other sections if not explicitly mapped above
            target_section = section_map.get(section, section.lower())

            # Use target_section for lookup, but original `section` for errors/display if needed
            if target_section and target_section.lower() not in [
                s[0].lower() for s in sections
            ]:
                # Try to find attribute
                attr = getattr(config, target_section, None)
                if attr:
                    # Generic display for unmapped sections
                    print(f"  {_colorize(section, 'cyan')}")
                    if hasattr(attr, "model_dump"):
                        for k, v in attr.model_dump().items():
                            print(f"    {k:<20} {v}")
                    else:
                        print(f"    {attr}")
                    return
                else:
                    print(
                        f"{_colorize('ERROR:', 'red')} Section '{section}' (mapped to '{target_section}') not found"
                    )
                    return

            for sec_name, items in sections:
                if section and section.lower() != sec_name.lower():
                    continue

                print(f"  {_colorize(sec_name, 'cyan')}")
                for key, value in items:
                    print(f"    {key:<20} {value}")
                print()

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
    provider: str = "vertex",
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


def _run_search(
    query: str,
    top_k: int = 10,
    tenant_id: str = "default",
    json_output: bool = False,
) -> None:
    """
    Search indexed emails with natural language queries.
    """
    import json

    if not json_output:
        _print_banner()
        print(f"{_colorize('▶ SEARCH', 'bold')}\n")
        print(f"  Query:   {_colorize(query, 'cyan')}")
        print(f"  Top K:   {_colorize(str(top_k), 'dim')}")
        print()

    try:
        from cortex.models.api import (
            SearchRequest as _SearchRequest,  # type: ignore[import]
        )
        from cortex.retrieval.hybrid_search import (
            hybrid_search as _hybrid_search,  # type: ignore[import]; type: ignore[reportUnknownVariableType]
        )

        _SearchRequest = cast(Any, _SearchRequest)
        _hybrid_search = cast(Any, _hybrid_search)
        SearchRequest: type[Any]
        hybrid_search: Callable[[Any], Any]
        SearchRequest = cast(type[Any], _SearchRequest)
        hybrid_search = cast(Callable[[Any], Any], _hybrid_search)

        request: Any = SearchRequest(
            query=query,
            top_k=top_k,
            tenant_id=tenant_id,
        )

        if not json_output:
            print(f"  {_colorize('⏳', 'yellow')} Searching...\n")

        results: Any = hybrid_search(request)

        if json_output:
            # Convert results to JSON-serializable format
            output: dict[str, Any] = {
                "success": True,
                "query": query,
                "results": (
                    [r.model_dump() for r in results.results]
                    if hasattr(results, "results")
                    else []
                ),
                "total": len(results.results) if hasattr(results, "results") else 0,
            }
            print(json.dumps(output, indent=2, default=str))
        else:
            if hasattr(results, "results") and results.results:
                print(
                    f"  {_colorize('✓', 'green')} Found {len(results.results)} result(s):\n"
                )
                for i, r in enumerate(results.results[:top_k], 1):
                    score = getattr(r, "score", 0)
                    text = getattr(r, "text", str(r))[:200]
                    source = getattr(r, "source", "unknown")

                    print(
                        f"  {_colorize(f'[{i}]', 'bold')} Score: {_colorize(f'{score:.3f}', 'cyan')}"
                    )
                    print(f"      Source: {_colorize(str(source), 'dim')}")
                    print(f"      {text}...")
                    print()
            else:
                print(f"  {_colorize('○', 'yellow')} No results found for: {query}")
            print()

    except ImportError as e:
        msg = f"Could not load search module: {e}"
        if json_output:
            print(json.dumps({"error": msg, "success": False}))
        else:
            print(f"\n  {_colorize('ERROR:', 'red')} {msg}")
            print(f"  Make sure you have run {_colorize('cortex index', 'cyan')} first")
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
    import asyncio
    import json

    if not json_output:
        _print_banner()
        print(f"{_colorize('▶ ASK CORTEX', 'bold')}\n")
        print(f"  Query:   {_colorize(query, 'cyan')}")
        print()

    try:
        from cortex.orchestration.graphs import (
            build_answer_graph as _build_answer_graph,  # type: ignore[import]; type: ignore[reportUnknownVariableType]
        )

        _build_answer_graph = cast(Any, _build_answer_graph)
        build_answer_graph: Callable[[], Any]
        build_answer_graph = cast(Callable[[], Any], _build_answer_graph)

        if not json_output:
            print(f"  {_colorize('⏳', 'yellow')} Thinking...")

        async def _execute() -> Any:
            graph = build_answer_graph().compile()
            initial_state: dict[str, Any] = {
                "query": query,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "classification": None,
                "retrieval_results": None,
                "assembled_context": None,
                "answer": None,
                "error": None,
            }
            result = await graph.ainvoke(initial_state)
            return result

        final_state: Any = asyncio.run(_execute())

        if final_state.error:
            raise Exception(final_state.error)

        answer: Any | None = final_state.answer

        if json_output:
            print(
                json.dumps(answer.model_dump() if answer else {}, indent=2, default=str)
            )
        else:
            if answer:
                print(f"\n{_colorize('ANSWER:', 'bold')}")
                print(f"{answer.answer_markdown}\n")

                if answer.evidence:
                    print(f"{_colorize('SOURCES:', 'dim')}")
                    for i, ev in enumerate(answer.evidence, 1):
                        snippet = ev.snippet or ev.text or ""
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


def _run_draft(
    instructions: str,
    thread_id: str | None = None,
    tenant_id: str = "default",
    user_id: str = "cli-user",
    json_output: bool = False,
) -> None:
    """
    Draft email replies based on context.
    """
    import asyncio
    import json

    if not json_output:
        _print_banner()
        print(f"{_colorize('▶ DRAFT EMAIL', 'bold')}\n")
        print(f"  Instructions: {_colorize(instructions, 'cyan')}")
        if thread_id:
            print(f"  Thread ID:    {_colorize(thread_id, 'dim')}")
        print()

    try:
        from cortex.orchestration.graphs import (
            build_draft_graph as _build_draft_graph,  # type: ignore[import]; type: ignore[reportUnknownVariableType]
        )

        _build_draft_graph = cast(Any, _build_draft_graph)
        build_draft_graph: Callable[[], Any]
        build_draft_graph = cast(Callable[[], Any], _build_draft_graph)

        if not json_output:
            print(f"  {_colorize('⏳', 'yellow')} Drafting...")

        async def _execute() -> Any:
            graph = build_draft_graph().compile()
            initial_state: dict[str, Any] = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "thread_id": thread_id,
                "explicit_query": instructions,
                "draft_query": None,
                "retrieval_results": None,
                "assembled_context": None,
                "draft": None,
                "critique": None,
                "iteration_count": 0,
                "error": None,
            }
            result = await graph.ainvoke(initial_state)
            return result

        final_state: Any = asyncio.run(_execute())

        if final_state.error:
            raise Exception(final_state.error)

        draft: Any | None = final_state.draft

        if json_output:
            print(
                json.dumps(draft.model_dump() if draft else {}, indent=2, default=str)
            )
        else:
            if draft:
                print(f"\n{_colorize('DRAFT:', 'bold')}")
                print(f"Subject: {draft.subject}")
                print(f"To: {', '.join(draft.to)}")
                print("-" * 40)
                print(f"{draft.body_markdown}\n")

                if draft.next_actions:
                    print(f"{_colorize('NEXT ACTIONS:', 'dim')}")
                    for i, action in enumerate(draft.next_actions, 1):
                        description = getattr(action, "description", None)
                        owner = getattr(action, "owner", None)
                        due_date = getattr(action, "due_date", None)
                        if description is None and isinstance(action, dict):
                            description = action.get("description")
                            owner = owner or action.get("owner")
                            due_date = due_date or action.get("due_date")
                        if description is None:
                            description = str(action)
                        extras = " · ".join(
                            item
                            for item in (
                                f"Owner: {owner}" if owner else None,
                                f"Due: {due_date}" if due_date else None,
                            )
                            if item
                        )
                        suffix = f" ({extras})" if extras else ""
                        print(f"  {i}. {description}{suffix}")
            else:
                print(f"  {_colorize('⚠', 'yellow')} No draft generated.")
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

        if final_state.error:
            raise Exception(final_state.error)

        summary: Any | None = final_state.summary

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

    # ==========================================================================
    # CORE COMMANDS
    # ==========================================================================

    # Ingest command
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

    # Index command
    index_parser = subparsers.add_parser(
        "index",
        help="Build/rebuild search index with embeddings",
        description="""
Build or rebuild the search index.

This command:
  1. Scans conversation folders for content
  2. Chunks text into embedding-friendly segments
  3. Generates embeddings using the configured provider
  4. Saves index for fast retrieval

Uses parallel workers for faster processing.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    index_parser.add_argument(
        "--root",
        "-r",
        default=".",
        metavar="DIR",
        help="Root directory containing conversations (default: current)",
    )
    index_parser.add_argument(
        "--provider",
        "-p",
        default="vertex",
        choices=["vertex", "openai", "local"],
        help="Embedding provider (default: vertex)",
    )
    index_parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel workers (default: 4)",
    )
    index_parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        metavar="N",
        help="Limit number of conversations to index (for testing)",
    )
    index_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force full reindex (ignore existing)",
    )
    index_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    # Search command
    search_parser = subparsers.add_parser(
        "search",
        help="Search indexed emails with natural language",
        description="""
Search your indexed emails using natural language queries.

Examples:
  cortex search "contract renewal terms"
  cortex search "emails from John about budget" --top-k 20
  cortex search "attachments mentioning quarterly report"

Uses hybrid search (vector + full-text) for best results.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    search_parser.add_argument(
        "query",
        metavar="QUERY",
        help="Natural language search query",
    )
    search_parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
        metavar="N",
        help="Number of results to return (default: 10)",
    )
    search_parser.add_argument(
        "--tenant",
        "-t",
        default="default",
        metavar="ID",
        help="Tenant ID (default: 'default')",
    )
    search_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

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

    # ==========================================================================
    # RAG COMMANDS
    # ==========================================================================

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

    # Draft command
    draft_parser = subparsers.add_parser(
        "draft",
        help="Draft email replies based on context",
        description="""
Draft an email reply based on instructions and optional thread context.

The system will:
  1. Retrieve relevant context (if thread ID provided)
  2. Follow your instructions
  3. Generate a professional email draft
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    draft_parser.add_argument(
        "instructions",
        metavar="INSTRUCTIONS",
        help="Instructions for the draft (e.g. 'Reply politely declining')",
    )
    draft_parser.add_argument(
        "--thread-id",
        metavar="UUID",
        help="ID of the thread to reply to",
    )
    draft_parser.add_argument(
        "--tenant",
        "-t",
        default="default",
        metavar="ID",
        help="Tenant ID (default: 'default')",
    )
    draft_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

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

    # ==========================================================================
    # UTILITY COMMANDS
    # ==========================================================================

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
        default="vertex",
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

    # Version command
    subparsers.add_parser(
        "version",
        help="Display version information",
    )

    # Register new subcommand groups
    from cortex_cli.cmd_db import setup_db_parser
    from cortex_cli.cmd_embeddings import setup_embeddings_parser
    from cortex_cli.cmd_s3 import setup_s3_parser

    setup_db_parser(subparsers)
    setup_embeddings_parser(subparsers)
    setup_s3_parser(subparsers)

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
    if parsed_args.command == "ingest":
        _run_ingest(
            source_path=parsed_args.source,
            tenant_id=parsed_args.tenant,
            dry_run=parsed_args.dry_run,
            verbose=parsed_args.verbose,
            json_output=parsed_args.json,
        )

    elif parsed_args.command == "index":
        _run_index(
            root=parsed_args.root,
            provider=parsed_args.provider,
            workers=parsed_args.workers,
            limit=parsed_args.limit,
            force=parsed_args.force,
            json_output=parsed_args.json,
        )

    elif parsed_args.command == "search":
        _run_search(
            query=parsed_args.query,
            top_k=parsed_args.top_k,
            tenant_id=parsed_args.tenant,
            json_output=parsed_args.json,
        )

    elif parsed_args.command == "doctor":
        # Lazy import to avoid loading heavy dependencies
        from cortex_cli.cmd_doctor import main as doctor_main

        # Handle --all flag
        if parsed_args.check_all:
            parsed_args.check_index = True
            parsed_args.check_embeddings = True

        # Forward to doctor command
        doctor_args = list(args)
        if "doctor" in doctor_args:
            doctor_args.remove("doctor")
        # Convert --all to individual flags for the doctor module
        if "--all" in doctor_args:
            doctor_args.remove("--all")
            if "--check-index" not in doctor_args:
                doctor_args.append("--check-index")
            if "--check-embeddings" not in doctor_args:
                doctor_args.append("--check-embeddings")
        sys.argv = [sys.argv[0], *doctor_args]
        doctor_main()

    elif parsed_args.command == "status":
        _show_status(json_output=parsed_args.json)

    elif parsed_args.command == "config":
        export_format = "json" if parsed_args.json else None
        _show_config(
            validate=parsed_args.validate,
            export_format=export_format,
            section=getattr(parsed_args, "section", None),
        )

    elif parsed_args.command == "validate":
        _run_validate(
            path=parsed_args.path,
            json_output=parsed_args.json,
        )

    elif parsed_args.command == "answer":
        _run_answer(
            query=parsed_args.query,
            tenant_id=parsed_args.tenant,
            json_output=parsed_args.json,
        )

    elif parsed_args.command == "draft":
        _run_draft(
            instructions=parsed_args.instructions,
            thread_id=parsed_args.thread_id,
            tenant_id=parsed_args.tenant,
            json_output=parsed_args.json,
        )

    elif parsed_args.command == "summarize":
        _run_summarize(
            thread_id=parsed_args.thread_id,
            tenant_id=parsed_args.tenant,
            json_output=parsed_args.json,
        )

    elif parsed_args.command == "version":
        _print_version()

    # New subcommand groups
    elif parsed_args.command == "db":
        if hasattr(parsed_args, "func"):
            parsed_args.func(parsed_args)
        else:
            print("Usage: cortex db <stats|migrate>")
            sys.exit(1)

    elif parsed_args.command == "embeddings":
        if hasattr(parsed_args, "func"):
            parsed_args.func(parsed_args)
        else:
            print("Usage: cortex embeddings <stats|backfill>")
            sys.exit(1)

    elif parsed_args.command == "s3":
        if hasattr(parsed_args, "func"):
            parsed_args.func(parsed_args)
        else:
            print("Usage: cortex s3 <list|ingest>")
            sys.exit(1)

    else:
        _print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
