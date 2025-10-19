"""
Setup module for EmailOps Vertex AI.

This module contains scripts for system setup and configuration.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

__all__ = ["enable_vertex_apis"]
