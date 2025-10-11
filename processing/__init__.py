"""
Processing Module for EmailOps
Provides text processing, embedding, and monitoring capabilities
"""

from diagnostics.monitor import IndexMonitor, IndexStatus, ProcessInfo
from processing.processor import (
    ChunkJob,
    ProcessingStats,
    UnifiedProcessor,
    WorkerConfig,
    WorkerStats,
)

__all__ = [
    # Data classes
    "ChunkJob",
    "IndexMonitor",
    "IndexStatus",
    "ProcessInfo",
    "ProcessingStats",
    # Main classes
    "UnifiedProcessor",
    "WorkerConfig",
    "WorkerStats",
]

# Module information
__version__ = "2.0.0"
__author__ = "EmailOps Team"
