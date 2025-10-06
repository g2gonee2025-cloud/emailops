#!/usr/bin/env python3
"""
Shared utility functions for analysis modules.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure logging with consistent format across analysis modules.
    
    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def get_index_path(root: Optional[str] = None) -> Path:
    """
    Resolve index directory path from root or environment.
    
    Args:
        root: Optional root directory path. If not provided, uses current directory
    
    Returns:
        Path to _index directory
    """
    index_dirname = os.getenv("INDEX_DIRNAME", "_index")
    
    if root:
        base_path = Path(root)
    else:
        base_path = Path.cwd()
    
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
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def save_json_report(data: Dict[str, Any], filename: str) -> Path:
    """
    Save analysis results as JSON with proper formatting.
    
    Args:
        data: Dictionary containing report data
        filename: Name of file to save (relative to current directory)
    
    Returns:
        Path to saved file
    """
    output_path = Path(filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return output_path
