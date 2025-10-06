"""
User Interface module for EmailOps Vertex AI.

This module contains the Streamlit web interface.
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

__all__ = [
    'emailops_ui'
]