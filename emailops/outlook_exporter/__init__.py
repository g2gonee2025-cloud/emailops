"""
EmailOps Outlook Exporter Module

Exports Outlook conversations to EmailOps manifest format.
"""

from .cli import main
from .exporter import OutlookExporter

__all__ = ["OutlookExporter", "main"]
