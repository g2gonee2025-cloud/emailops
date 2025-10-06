"""
Analysis module for EmailOps Vertex AI.

This module provides consolidated tools for diagnostics, statistics, and monitoring.

Modules:
- diagnostics: Account testing and index verification
- statistics: File analysis, chunk counting, and indexing progress monitoring
- utils: Shared utility functions
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import main functions from consolidated modules
from diagnostics.diagnostics import (
    test_account,
    diagnose_all_accounts,
    verify_index_alignment,
    check_index_consistency
)

from diagnostics.statistics import (
    analyze_file_processing,
    get_file_statistics,
    count_total_chunks,
    monitor_indexing_progress
)

from diagnostics.utils import (
    setup_logging,
    get_index_path,
    get_export_root,
    format_timestamp,
    save_json_report
)

__all__ = [
    # Diagnostics
    'test_account',
    'diagnose_all_accounts',
    'verify_index_alignment',
    'check_index_consistency',
    # Statistics
    'analyze_file_processing',
    'get_file_statistics',
    'count_total_chunks',
    'monitor_indexing_progress',
    # Utils
    'setup_logging',
    'get_index_path',
    'get_export_root',
    'format_timestamp',
    'save_json_report',
]
