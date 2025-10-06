"""
Tests module for EmailOps Vertex AI.

This module contains test scripts for validating functionality.
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

__all__ = [
    'test_all_accounts_live'
]