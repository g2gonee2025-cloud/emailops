"""
Base Service Module
"""

from pathlib import Path

from .file_service import FileService


class BaseService:
    """
    A base class for all services that provides access to the file service.
    """

    def __init__(self, export_root: str):
        """
        Initializes the BaseService.
        Args:
            export_root: The root directory for all exports.
        """
        self.export_root = Path(export_root)
        self.file_service = FileService(export_root)
