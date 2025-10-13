"""
Analysis module for EmailOps Vertex AI.

This module provides consolidated tools for diagnostics, statistics, and monitoring.

Modules:
- diagnostics: Account testing and index verification
- statistics: File analysis, chunk counting, and indexing progress monitoring
- utils: Shared utility functions
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import main functions from consolidated modules
from diagnostics.statistics import (
    analyze_file_processing,
    count_total_chunks,
    get_file_statistics,
    monitor_indexing_progress,
)
from diagnostics.utils import (
    format_timestamp,
    get_export_root,
    get_index_path,
    save_json_report,
    setup_logging,
)

__all__ = [
    "analyze_file_processing",
    "check_index_consistency",
    "count_total_chunks",
    "diagnose_all_accounts",
    "format_timestamp",
    "get_export_root",
    "get_file_statistics",
    "get_index_path",
    "monitor_indexing_progress",
    "save_json_report",
    "setup_logging",
    "test_account",
    "verify_index_alignment",
]
