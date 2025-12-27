from typing import Any

from .style import colorize as _colorize


def _print_json_config(config: Any, section: str | None = None) -> None:
    """Print configuration in JSON format."""
    import json

    data = config.model_dump()
    if section:
        if section in data:
            data = {section: data[section]}
        else:
            print(f"{_colorize('ERROR:', 'red')} Section '{section}' not found")
            return
    print(json.dumps(data, indent=2, default=str))


def _print_human_config(config: Any, section: str | None = None) -> None:
    """Print configuration in human-readable format."""
    title = f"Current Configuration ({section})" if section else "Current Configuration"
    print(f"{_colorize(f'{title}:', 'bold')}\n")

    section_map = {
        "DigitalOcean LLM": "digitalocean_llm",
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

    # 1. Print structured summary sections (if no specific section requested)
    if not section:
        _print_summary_sections(config)

    # 2. Print specific section or fallback
    if section:
        target = section_map.get(section, section.lower())
        attr = getattr(config, target, None)
        if attr:
            # Generic display for unmapped sections
            if section not in section_map:
                print(f"  {_colorize(section, 'cyan')}")

            if hasattr(attr, "model_dump"):
                for k, v in attr.model_dump().items():
                    print(f"    {k:<20} {v}")

    # If we are here and not looking for a section, we printed summary above.


def _print_summary_sections(config: Any) -> None:
    """Print the hardcoded summary sections."""
    sections = [
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
                ("Mode", config.embedding.embed_mode),
            ],
        ),
        (
            "Email",
            [
                ("Sender Name", config.email.sender_locked_name or "(not set)"),
                ("Sender Email", config.email.sender_locked_email or "(not set)"),
                ("Reply Policy", config.email.reply_policy),
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

    for sec_name, items in sections:
        print(f"  {_colorize(sec_name, 'cyan')}")
        for label, val in items:
            print(f"    {label:<15} {val}")
        print()
