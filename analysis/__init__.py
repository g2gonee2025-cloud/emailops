"""
Analysis module for EmailOps Vertex AI.

This module contains scripts for analyzing file processing, statistics, and monitoring.
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

__all__ = [
    'file_processing_analysis',
    'file_stats',
    'count_chunks',
    'monitor_indexing'
]