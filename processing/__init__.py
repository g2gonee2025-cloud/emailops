"""
Processing Module for EmailOps
Provides text processing, embedding, and monitoring capabilities
"""

from .processor import UnifiedProcessor, ChunkJob, WorkerConfig, WorkerStats, ProcessingStats
from ..diagnostics.monitor import IndexMonitor, IndexStatus, ProcessInfo

__all__ = [
    # Main classes
    'UnifiedProcessor',
    'IndexMonitor',
    
    # Data classes
    'ChunkJob',
    'WorkerConfig', 
    'WorkerStats',
    'ProcessingStats',
    'IndexStatus',
    'ProcessInfo',
]

# Module information
__version__ = '2.0.0'
__author__ = 'EmailOps Team'