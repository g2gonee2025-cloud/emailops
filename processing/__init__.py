"""
Processing module for EmailOps Vertex AI.

This module contains scripts for data processing, indexing, and embedding generation.
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

__all__ = [
    'vertex_indexer',
    'parallel_chunker', 
    'parallel_summarizer',
    'fix_failed_embeddings',
    'repair_vertex_parallel_index',
    'run_vertex_finalize'
]