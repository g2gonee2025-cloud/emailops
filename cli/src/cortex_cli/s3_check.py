from __future__ import annotations

import random
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
    try:
        sample_size = int(sample_size)
    except (TypeError, ValueError):
        sample_size = 0
    if sample_size < 0:
        sample_size = 0

    prefix = prefix.rstrip("/") + "/"

    try:
        handler = S3SourceHandler()
    except Exception as exc:
        return {
            "sampled_folders": [],
            "issues": {},
            "total_folders_found_in_scan": 0,
            "message": f"Failed to initialize S3 handler: {exc}",
        }

    folders: dict[str, str] = {}

    # Using the handler to list conversation folders
    try:
        for folder in handler.list_conversation_folders(prefix=prefix):
            folder_prefix = (
                getattr(folder, "prefix", None)
                or getattr(folder, "name", None)
                or str(folder)
            )
            if not folder_prefix.endswith("/"):
                folder_prefix = f"{folder_prefix}/"
            if not folder_prefix.startswith(prefix):
                folder_prefix = f"{prefix}{folder_prefix.lstrip('/')}"
            folder_name = (
                getattr(folder, "name", None)
                or folder_prefix.rstrip("/").split("/")[-1]
            )
            folders.setdefault(folder_name, folder_prefix)
    except Exception as exc:
        return {
            "sampled_folders": [],
            "issues": {},
            "total_folders_found_in_scan": 0,
            "message": f"Failed to list folders: {exc}",
        }

    if not folders:
        return {
            "sampled_folders": [],
            "issues": {},
            "total_folders_found_in_scan": 0,
            "message": "No folders found.",
        }

    sample: list[str] = (
        random.sample(list(folders), min(sample_size, len(folders)))
        if sample_size
        else []
    )
    expected_files = {"Conversation.txt", "manifest.json"}
    expected_prefixes = {"attachments/"}
    issues: dict[str, dict[str, list[str]]] = {}

    for folder_name in sample:
        folder_prefix = folders.get(folder_name, f"{prefix}{folder_name}/")
        try:
            keys, _ = handler.list_files_in_folder(folder_prefix)
        except Exception as exc:
            issues.setdefault(folder_name, {})["error"] = [
                f"Failed to list files: {exc}"
            ]
            continue

        keys_set = set(keys)
        missing: list[str] = [
            name for name in expected_files if f"{folder_prefix}{name}" not in keys_set
        ]
        for prefix_name in expected_prefixes:
            prefix_value = f"{folder_prefix}{prefix_name}"
            if not any(k.startswith(prefix_value) for k in keys):
                missing.append(prefix_name)

        if missing:
            issues.setdefault(folder_name, {})["missing"] = missing

    return {
        "sampled_folders": sample,
        "issues": issues,
        "total_folders_found_in_scan": len(folders),
    }
