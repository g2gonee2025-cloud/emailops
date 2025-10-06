"""
Diagnostics module for EmailOps Vertex AI.

This module contains scripts for system diagnostics, debugging, and validation.
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

__all__ = [
    'diagnose_accounts',
    'diagnose_accounts_fixed',
    'debug_parallel_indexer',
    'check_failed_batches',
    'verify_index_alignment',
    'check_all_files'
]