"""
EmailOps Service Layer

This module provides the service layer that abstracts business logic from the GUI.
All backend operations should be handled through these services, ensuring proper
separation of concerns between presentation and business logic.
"""

from emailops.services.analysis_service import AnalysisService

# Import atomic file service as the default FileService
from emailops.services.atomic_file_service import (
    AtomicFileService as FileService,
)
from emailops.services.atomic_file_service import (
    TransactionalFileService,
)
from emailops.services.batch_service import BatchService
from emailops.services.chat_service import ChatService
from emailops.services.chunking_service import ChunkingService
from emailops.services.config_service import ConfigService
from emailops.services.email_service import EmailService
from emailops.services.indexing_service import IndexingService
from emailops.services.search_service import SearchService

__all__ = [
    "AnalysisService",
    "BatchService",
    "ChatService",
    "ChunkingService",
    "ConfigService",
    "EmailService",
    "FileService",
    "IndexingService",
    "SearchService",
    "TransactionalFileService",
]
