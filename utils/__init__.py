"""
Utilities module for EmailOps Vertex AI.

This module contains utility functions and helper classes.
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

__all__ = [
    'vertex_utils'
]