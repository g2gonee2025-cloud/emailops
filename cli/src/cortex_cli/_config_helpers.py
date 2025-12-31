import os
import sys
from pathlib import Path
from typing import Any

from .style import colorize as _colorize

_MISSING = object()


def resolve_index_dir(root_dir: Path | None) -> Path:
    """Resolve the index directory."""
    if root_dir:
        return root_dir / "_index"
    # Fallback to environment variable or default
    from backend.src.cortex.indexing.metadata import INDEX_DIRNAME_DEFAULT

    env_value = os.getenv("INDEX_DIR")
    if env_value:
        return Path(env_value)
    return Path(INDEX_DIRNAME_DEFAULT)


def resolve_sender(sender: str | None) -> str:
    """Resolve the sender."""
    from .cmd_search import SENDER_LOCKED

    if sender:
        return sender
    return str(SENDER_LOCKED)


def _print_json_config(config: Any, section: str | None = None) -> None:
    """Print configuration in JSON format."""
    import json

    if config is None:
        print(
            f"{_colorize('ERROR:', 'red')} Configuration is not available.",
            file=sys.stderr,
        )
        return
    data = config.model_dump()
    if section:
        if section in data:
            data = {section: data[section]}
        else:
            print(
                f"{_colorize('ERROR:', 'red')} Section '{section}' not found",
                file=sys.stderr,
            )
            return
    print(json.dumps(data, indent=2, default=str))


def _print_human_config(config: Any, section: str | None = None) -> None:
    """Print configuration in human-readable format."""
    if config is None:
        print(
            f"{_colorize('ERROR:', 'red')} Configuration is not available.",
            file=sys.stderr,
        )
        return
    title = f"Current Configuration ({section})" if section else "Current Configuration"
    print(f"{_colorize(f'{title}:', 'bold')}\n")

    section_map = {
        "DigitalOcean LLM": "digitalocean_llm",
        "Core": "core",
        "Embeddings": "embedding",
        "Email": "email",
        "Search": "search",
        "Processing": "processing",
        "Database": "database",
        "Storage": "storage",
        "GCP": "gcp",
        "Retry": "retry",
        "Limits": "limits",
    }

    # 1. Print structured summary sections (if no specific section requested)
    if not section:
        _print_summary_sections(config)

    # 2. Print specific section or fallback
    if section:
        target = section_map.get(section, section.lower())
        attr = getattr(config, target, _MISSING)
        if attr is not _MISSING:
            # Generic display for unmapped sections
            if section not in section_map:
                print(f"  {_colorize(section, 'cyan')}")

            if hasattr(attr, "model_dump"):
                for k, v in attr.model_dump().items():
                    print(f"    {k:<20} {v}")
            else:
                print(f"    value               {attr}")
        else:
            print(
                f"{_colorize('ERROR:', 'red')} Section '{section}' not found",
                file=sys.stderr,
            )

    # If we are here and not looking for a section, we printed summary above.


def _print_summary_sections(config: Any) -> None:
    """Print the hardcoded summary sections."""
    sections = [
        (
            "Core",
            [
                ("Environment", _safe_get(config, "core", "env", default="N/A")),
                ("Provider", _safe_get(config, "core", "provider", default="N/A")),
                ("Persona", _safe_get(config, "core", "persona", default="N/A")),
            ],
        ),
        (
            "Embeddings",
            [
                ("Model", _safe_get(config, "embedding", "model_name", default="N/A")),
                (
                    "Dimensions",
                    _safe_get(
                        config, "embedding", "output_dimensionality", default="N/A"
                    ),
                ),
                (
                    "Batch Size",
                    _safe_get(config, "embedding", "batch_size", default="N/A"),
                ),
                ("Mode", _safe_get(config, "embedding", "embed_mode", default="N/A")),
            ],
        ),
        (
            "Email",
            [
                (
                    "Sender Name",
                    _safe_get(config, "email", "sender_locked_name", default="")
                    or "(not set)",
                ),
                (
                    "Sender Email",
                    _safe_get(config, "email", "sender_locked_email", default="")
                    or "(not set)",
                ),
                (
                    "Reply Policy",
                    _safe_get(config, "email", "reply_policy", default="N/A"),
                ),
            ],
        ),
        (
            "Search",
            [
                (
                    "Strategy",
                    _safe_get(config, "search", "fusion_strategy", default="N/A"),
                ),
                ("K", _safe_get(config, "search", "k", default="N/A")),
                (
                    "Recency",
                    _safe_get(
                        config, "search", "recency_boost_strength", default="N/A"
                    ),
                ),
                (
                    "Reranker",
                    _safe_get(config, "search", "reranker_endpoint", default="N/A"),
                ),
            ],
        ),
        (
            "Processing",
            [
                (
                    "Chunk Size",
                    _safe_get(config, "processing", "chunk_size", default="N/A"),
                ),
                (
                    "Chunk Overlap",
                    _safe_get(config, "processing", "chunk_overlap", default="N/A"),
                ),
            ],
        ),
    ]

    for sec_name, items in sections:
        print(f"  {_colorize(sec_name, 'cyan')}")
        for label, val in items:
            print(f"    {label:<15} {val}")
        print()


def _safe_get(config: Any, *path: str, default: Any = None) -> Any:
    current = config
    for attr in path:
        if current is None:
            return default
        current = getattr(current, attr, _MISSING)
        if current is _MISSING:
            return default
    return current
