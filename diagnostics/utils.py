#!/usr/bin/env python3
"""
Shared utility functions for analysis modules.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure logging with consistent format across analysis modules.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Get the logger first
    logger = logging.getLogger(__name__)
    
    # Set the level on the logger itself
    logger.setLevel(log_level)
    
    # Also configure basic config
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    return logger


def get_index_path(root: str | None = None) -> Path:
    """
    Resolve index directory path from root or environment.

    Args:
        root: Optional root directory path. If not provided, uses current directory

    Returns:
        Path to _index directory
    """
    index_dirname = os.getenv("INDEX_DIRNAME", "_index")

    base_path = Path(root) if root else Path.cwd()

    return base_path / index_dirname


def get_export_root() -> Path:
    """
    Determine export root from environment or current directory.

    Returns:
        Path to export root directory
    """
    # Try environment variable first
    export_root = os.getenv("OUTLOOK_EXPORT_ROOT")
    if export_root:
        return Path(export_root)

    # Fall back to current directory
    return Path.cwd()


def format_timestamp(dt: datetime) -> str:
    """
    Consistent timestamp formatting for reports and logs.

    Args:
        dt: datetime object to format

    Returns:
        Formatted timestamp string
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def save_json_report(data: dict[str, Any], filename: str) -> Path:
    """
    Save analysis results as JSON with proper formatting.

    Args:
        data: Dictionary containing report data
        filename: Name of file to save (relative to current directory)

    Returns:
        Path to saved file
    """
    output_path = Path(filename)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path
