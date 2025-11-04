"""
EmailOps Modern GUI Package

A professional, modern GUI for EmailOps built with PyQt6.
Features clean architecture, async operations, and beautiful Material Design.
"""

__version__ = "2.0.0"
__author__ = "EmailOps Team"

from .app import EmailOpsApplication
from .main_window import EmailOpsMainWindow

__all__ = ["EmailOpsApplication", "EmailOpsMainWindow"]
