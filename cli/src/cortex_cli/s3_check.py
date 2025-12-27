from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from cortex.ingestion.s3_source import S3SourceHandler


def check_s3_structure(
    prefix: str = "raw/outlook/",
    sample_size: int = 20,
) -> dict[str, Any]:
    """
    Analyzes the structure of S3 folders under a given prefix.

    Args:
        prefix: The S3 prefix to check.
        sample_size: The number of folders to sample for checking.

    Returns:
        A dictionary containing the analysis results.
    """
    handler = S3SourceHandler()

    folders: set[str] = set()

    # Using the handler to list conversation folders
    for folder in handler.list_conversation_folders(prefix=prefix):
        folders.add(folder.name)

    if not folders:
        return {
            "sampled_folders": [],
            "issues": {},
            "total_folders_found_in_scan": 0,
            "message": "No folders found.",
        }

    sample: list[str] = random.sample(sorted(folders), min(sample_size, len(folders)))
    expected: set[str] = {"Conversation.txt", "manifest.json", "attachments/"}
    issues: dict[str, dict[str, list[str]]] = defaultdict(dict)

    for folder_name in sample:
        folder_prefix = f"{prefix}{folder_name}/"
        keys, _ = handler.list_files_in_folder(folder_prefix)

        missing = [
            e
            for e in expected
            if not any(k.startswith(f"{folder_prefix}{e}") for k in keys)
        ]

        if missing:
            issues[folder_name]["missing"] = missing

    return {
        "sampled_folders": sample,
        "issues": issues,
        "total_folders_found_in_scan": len(folders),
    }
