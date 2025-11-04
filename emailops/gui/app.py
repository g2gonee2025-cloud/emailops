"""
EmailOps Application - Main Application Class

Modern PyQt6 application with proper architecture, threading, and error handling.
Features Material Design, dark theme, and smooth animations.
"""

import asyncio
import inspect
import logging
import sys
from pathlib import Path
from typing import Any

from PyQt6.QtCore import (
    QObject,
    QRunnable,
    QSettings,
    Qt,
    QThreadPool,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QFont, QPalette
from PyQt6.QtWidgets import QApplication, QStyleFactory

# Import our services
from emailops.services import (
    AnalysisService,
    BatchService,
    ChatService,
    ChunkingService,
    ConfigService,
    EmailService,
    FileService,
    IndexingService,
    SearchService,
)

from .theme import ThemeManager

logger = logging.getLogger(__name__)


class ServiceWorker(QRunnable):
    """
    Worker for running service operations in thread pool.
    Proper async handling without blocking the GUI.
    """

    class Signals(QObject):
        finished = pyqtSignal(object)
        error = pyqtSignal(Exception)
        progress = pyqtSignal(int, str)

    def __init__(self, func: Any, *args: Any, **kwargs: Any):
        """
        Initialize service worker with function and arguments.

        Args:
            func: Function to execute (sync or async)
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func
        """
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = self.Signals()
        self.setAutoDelete(True)

    def run(self):
        """Execute the service function, handling both sync and async methods."""
        try:
            # Add progress callback adapter if supported
            if "progress_callback" in self.kwargs:
                def progress_adapter(*args):
                    try:
                        if len(args) == 4:
                            current, total, conv_id, status = args
                            pct = int((current / total) * 100) if total > 0 else 0
                            self.signals.progress.emit(pct, f"{conv_id}: {status}")
                        elif len(args) == 3:
                            current, total, message = args
                            pct = int((current / total) * 100) if total > 0 else 0
                            self.signals.progress.emit(pct, str(message))
                        else:
                            logger.warning(f"Unhandled progress signature: {args}")
                    except Exception as e:
                        logger.error(f"Progress adapter error: {e}")
                self.kwargs["progress_callback"] = progress_adapter

            # Check if the function is async and run it in an event loop
            if inspect.iscoroutinefunction(self.func):
                result = asyncio.run(self.func(*self.args, **self.kwargs))
            else:
                result = self.func(*self.args, **self.kwargs)

            self.signals.finished.emit(result)
        except Exception as e:
            logger.error(f"Service worker error: {e}", exc_info=True)
            self.signals.error.emit(e)


class ServiceManager(QObject):
    """
    Central service manager - handles all backend operations.
    Fixed version with proper parameter handling.
    """

    # Signals for GUI updates
    search_completed = pyqtSignal(list)  # Search results
    email_generated = pyqtSignal(dict)  # Email draft
    index_updated = pyqtSignal(dict)  # Index stats
    chunk_completed = pyqtSignal(dict)  # Chunk results
    batch_progress = pyqtSignal(int, str)  # Batch operation progress
    chat_response = pyqtSignal(dict)  # Chat response
    error_occurred = pyqtSignal(str)  # Error messages

    # New signals for missing services
    chunks_updated = pyqtSignal(dict)  # Chunking results
    analysis_completed = pyqtSignal(dict)  # Analysis results
    config_updated = pyqtSignal(dict)  # Config changes
    file_operation_completed = pyqtSignal(dict)  # File operation results

    def __init__(self, config_path: Path):
        super().__init__()

        # Validate thread pool
        self.thread_pool = QThreadPool.globalInstance()
        if self.thread_pool is None:
            raise RuntimeError("Failed to initialize Qt thread pool - cannot proceed")
        self.thread_pool.setMaxThreadCount(4)

        # Get export root from configuration
        from emailops.core_config import get_config
        config = get_config()

        # Use EXPORT_ROOT from config/environment, fall back to config_path.parent if not set
        export_root_path = (
            Path(config.core.export_root)
            if config.core.export_root
            else config_path.parent
        )

        # Validate export root
        if not export_root_path.exists():
            logger.warning(f"Export root does not exist, creating: {export_root_path}")
            export_root_path.mkdir(parents=True, exist_ok=True)
        if not export_root_path.is_dir():
            raise ValueError(f"Export root is not a directory: {export_root_path}")

        export_root = str(export_root_path)
        logger.info(f"Using export root: {export_root}")

        # Get index directory name from configuration
        index_dirname = config.directories.index_dirname
        logger.info(f"Using index directory: {index_dirname}")

        # Initialize all services with validation
        try:
            self.search_service = SearchService(export_root, index_dirname=index_dirname)
            self.email_service = EmailService(export_root)
            self.chunking_service = ChunkingService(export_root)
            self.indexing_service = IndexingService(export_root, index_dirname=index_dirname)
            self.analysis_service = AnalysisService(export_root)
            self.batch_service = BatchService(export_root)
            self.chat_service = ChatService(export_root, index_dirname=index_dirname)
            self.file_service = FileService(export_root)
            self.config_service = ConfigService(config_path)
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}", exc_info=True)
            raise RuntimeError(f"Service initialization failed: {e}") from e

        logger.info("ServiceManager initialized with all services")

    @pyqtSlot(dict)
    def perform_search(self, params: dict):
        """Execute search operation in thread pool - VALIDATION ADDED"""
        # Validate parameters
        if not isinstance(params, dict):
            self._handle_error(TypeError(f"params must be dict, got {type(params)}"))
            return

        # Extract and validate query
        query = params.get("query", "")
        if not isinstance(query, str):
            self._handle_error(TypeError("query must be string"))
            return

        if not query.strip():
            self._handle_error(ValueError("Search query cannot be empty"))
            return

        # Validate numeric parameters
        k = params.get("k", 10)
        if not isinstance(k, int) or k < 1 or k > 100:
            self._handle_error(ValueError("k must be integer between 1 and 100"))
            return

        mmr_lambda = params.get("mmr_lambda", 0.7)
        if not isinstance(mmr_lambda, (int, float)) or not 0 <= mmr_lambda <= 1:
            self._handle_error(ValueError("mmr_lambda must be float between 0 and 1"))
            return

        rerank_alpha = params.get("rerank_alpha", 0.35)
        if not isinstance(rerank_alpha, (int, float)) or not 0 <= rerank_alpha <= 1:
            self._handle_error(ValueError("rerank_alpha must be float between 0 and 1"))
            return

        filters = params.get("filters", {})

        # Build SearchFilters object if filters provided
        search_filters = None
        if filters and any(filters.values()):
            search_filters = self.search_service.build_search_filters(
                from_email=filters.get("from_email", ""),
                to_email=filters.get("to_email", ""),
                cc_email=filters.get("cc_email", ""),
                subject_contains=filters.get("subject_contains", ""),
                file_types=filters.get("file_types", ""),
                has_attachment="yes" if filters.get("has_attachments") else "any",
                date_from=filters.get("date_from", ""),
                date_to=filters.get("date_to", ""),
            )

        worker = ServiceWorker(
            self.search_service.perform_search,
            query=query,
            k=params.get("k", 10),
            provider=params.get("provider", "vertex"),
            mmr_lambda=params.get("mmr_lambda", 0.7),
            rerank_alpha=params.get("rerank_alpha", 0.35),
            filters=search_filters,
        )
        worker.signals.finished.connect(self.search_completed.emit)
        worker.signals.error.connect(self._handle_error)
        if self.thread_pool is not None:
            self.thread_pool.start(worker)

    @pyqtSlot(dict)
    def generate_email(self, params: dict):
        """Generate email draft in thread pool - FIXED"""
        operation = params.get("operation", "reply")

        if operation == "reply":
            worker = ServiceWorker(
                self.email_service.draft_reply,
                conv_id=params.get("conv_id", ""),
                query=params.get("query", ""),
                provider=params.get("provider", "vertex"),
                sim_threshold=params.get("sim_threshold", 0.5),
                target_tokens=params.get("target_tokens", 20000),
                temperature=params.get("temperature", 0.7),
                reply_policy=params.get("reply_policy", "reply_all"),
                include_attachments=params.get("include_attachments", True),
            )
        else:
            worker = ServiceWorker(
                self.email_service.draft_fresh_email,
                to_list=params.get("to_list", []),
                cc_list=params.get("cc_list", []),
                subject=params.get("subject", ""),
                query=params.get("query", ""),
                provider=params.get("provider", "vertex"),
                target_tokens=params.get("target_tokens", 10000),
                temperature=params.get("temperature", 0.7),
                include_attachments=params.get("include_attachments", True),
            )

        worker.signals.finished.connect(self.email_generated.emit)
        worker.signals.error.connect(self._handle_error)
        if self.thread_pool is not None:
            self.thread_pool.start(worker)

    @pyqtSlot(dict)
    def build_index(self, params: dict):
        """Build or update index in thread pool - FIXED"""
        worker = ServiceWorker(
            self.indexing_service.build_index,
            provider=params.get("provider", "vertex"),
            batch_size=params.get("batch_size", 64),
            num_workers=params.get("num_workers", 4),
            force=params.get("force", False),
            limit=params.get("limit"),
            progress_callback=None,
        )
        # progress_callback replaced by ServiceWorker progress signal
        worker.signals.finished.connect(self.index_updated.emit)
        worker.signals.error.connect(self._handle_error)
        worker.signals.progress.connect(self.batch_progress.emit)
        if self.thread_pool is not None:
            self.thread_pool.start(worker)

    @pyqtSlot(dict)
    def process_batch(self, params: dict):
        """Process batch operations in thread pool - FIXED"""
        # Determine output directory
        output_dir = Path(
            params.get("output_dir", Path.home() / ".emailops" / "batch_output")
        )
        operation = params.get("operation", "summarize")

        if operation == "summarize":
            worker = ServiceWorker(
                self.batch_service.batch_summarize,
                conv_ids=params.get("items", []),
                output_dir=output_dir,
                temperature=params.get("temperature", 0.7),
                merge_manifest=params.get("merge_manifest", True),
                progress_callback=None,
            )
            # progress_callback replaced by ServiceWorker progress signal
        else:  # draft_replies
            worker = ServiceWorker(
                self.batch_service.batch_draft_replies,
                conv_ids=params.get("items", []),
                output_dir=output_dir,
                provider=params.get("provider", "vertex"),
                sim_threshold=params.get("sim_threshold", 0.5),
                target_tokens=params.get("target_tokens", 20000),
                temperature=params.get("temperature", 0.7),
                reply_policy=params.get("reply_policy", "reply_all"),
                include_attachments=params.get("include_attachments", True),
                progress_callback=None,
            )
            # progress_callback replaced by ServiceWorker progress signal

        worker.signals.finished.connect(self._batch_completed)
        worker.signals.error.connect(self._handle_error)
        worker.signals.progress.connect(self._batch_progress_handler)
        if self.thread_pool is not None:
            self.thread_pool.start(worker)

    @pyqtSlot(dict)
    def send_chat(self, params: dict):
        """Send chat message in thread pool - Already correct"""
        worker = ServiceWorker(
            self.chat_service.chat_with_query,
            query=params.get("message", ""),
            session_id=params.get("session_id", "default"),
            k=params.get("context_k", 10),
            provider=params.get("provider", "vertex"),
            temperature=params.get("temperature", 0.7),
            max_history=params.get("max_history", 5),
        )
        worker.signals.finished.connect(self.chat_response.emit)
        worker.signals.error.connect(self._handle_error)
        if self.thread_pool is not None:
            self.thread_pool.start(worker)

    # NEW METHODS FOR UNUSED SERVICES

    @pyqtSlot(dict)
    def chunk_conversations(self, params: dict):
        """Process text chunking operations - VALIDATION ADDED"""
        operation = params.get("operation", "incremental")

        # Validate surgical mode parameters
        if operation == "surgical":
            conv_ids = params.get("conv_ids", [])
            if not conv_ids:
                self._handle_error(ValueError("Surgical mode requires conversation IDs"))
                return
            if not isinstance(conv_ids, list):
                self._handle_error(TypeError("conv_ids must be a list"))
                return
            # Validate each conv_id is non-empty string
            invalid_ids = [
                cid for cid in conv_ids
                if not isinstance(cid, str) or not cid.strip()
            ]
            if invalid_ids:
                self._handle_error(ValueError(f"Invalid conversation IDs: {invalid_ids}"))
                return

        if operation == "force_all":
            worker = ServiceWorker(self.chunking_service.force_rechunk_all)
        elif operation == "surgical":
            worker = ServiceWorker(
                self.chunking_service.surgical_rechunk,
                conv_ids=params.get("conv_ids", []),
            )
        else:  # incremental
            worker = ServiceWorker(self.chunking_service.incremental_chunk)

        worker.signals.finished.connect(self.chunks_updated.emit)
        worker.signals.error.connect(self._handle_error)
        if self.thread_pool is not None:
            self.thread_pool.start(worker)

    @pyqtSlot(dict)
    def analyze_conversation(self, params: dict):
        """Analyze email conversations - VALIDATION ADDED"""
        conv_id = params.get("conv_id")

        if conv_id:
            # Single conversation analysis - validate before creating worker
            if not isinstance(conv_id, str) or not conv_id.strip():
                self._handle_error(ValueError("Conversation ID must be a non-empty string"))
                return

            conv_path = self.file_service.get_conversation_path(conv_id)
            if not conv_path:
                self._handle_error(ValueError(f"Invalid conversation ID: {conv_id}"))
                return

            # Verify conversation exists
            conv_txt = conv_path / "Conversation.txt"
            if not conv_txt.exists():
                self._handle_error(ValueError(f"Conversation.txt not found for ID: {conv_id}"))
                return

            worker = ServiceWorker(
                self.analysis_service.analyze_conversation,
                thread_dir=conv_path,
                temperature=params.get("temperature", 0.7),
                merge_manifest=params.get("merge_manifest", True),
                output_format=params.get("output_format", "json"),
            )
        else:
            # Batch analysis - validate conv_ids list
            conv_ids = params.get("conv_ids", [])
            if not conv_ids:
                self._handle_error(ValueError("Batch analysis requires conversation IDs"))
                return
            if not isinstance(conv_ids, list):
                self._handle_error(TypeError("conv_ids must be a list"))
                return

            # Validate each conv_id
            invalid_ids = [
                cid for cid in conv_ids
                if not isinstance(cid, str) or not cid.strip()
            ]
            if invalid_ids:
                self._handle_error(ValueError(f"Invalid conversation IDs in batch: {invalid_ids}"))
                return

            worker = ServiceWorker(
                self.batch_service.batch_summarize,
                conv_ids=conv_ids,
                output_dir=Path(
                    params.get("output_dir", Path.home() / ".emailops" / "analysis")
                ),
                temperature=params.get("temperature", 0.7),
                merge_manifest=params.get("merge_manifest", True),
                progress_callback=None,
            )
            # progress_callback replaced by ServiceWorker progress signal

        worker.signals.finished.connect(self.analysis_completed.emit)
        worker.signals.error.connect(self._handle_error)
        if hasattr(worker.signals, 'progress'):
            worker.signals.progress.connect(self.batch_progress.emit)
        if self.thread_pool is not None:
            self.thread_pool.start(worker)

    @pyqtSlot(dict)
    def update_config(self, params: dict):
        """Update configuration settings"""
        config_dict = params.get("config", {})

        worker = ServiceWorker(
            self.config_service.apply_configuration,
            config_dict=config_dict
        )
        worker.signals.finished.connect(self._config_updated)
        worker.signals.error.connect(self._handle_error)
        if self.thread_pool is not None:
            self.thread_pool.start(worker)

    @pyqtSlot()
    def get_config(self):
        """Get current configuration"""
        worker = ServiceWorker(self.config_service.get_configuration_dict)
        worker.signals.finished.connect(self.config_updated.emit)
        worker.signals.error.connect(self._handle_error)
        if self.thread_pool is not None:
            self.thread_pool.start(worker)

    @pyqtSlot(dict)
    def export_data(self, params: dict):
        """Export data to file"""
        operation = params.get("operation", "export_csv")
        data = params.get("data", [])
        output_path = Path(params.get("output_path", ""))

        if operation == "export_csv":
            worker = ServiceWorker(
                self.file_service.export_csv,
                data=data,
                path=output_path,
                headers=params.get("headers"),
            )
        elif operation == "save_json":
            worker = ServiceWorker(
                self.file_service.save_json,
                data=data,
                path=output_path,
            )
        else:
            worker = ServiceWorker(
                self.file_service.save_text_file,
                content=params.get("content", ""),
                path=output_path,
            )

        worker.signals.finished.connect(self._emit_file_operation_success)
        worker.signals.error.connect(self._handle_error)
        if self.thread_pool is not None:
            self.thread_pool.start(worker)

    @pyqtSlot(dict)
    def import_data(self, params: dict):
        """Import data from file"""
        input_path = Path(params.get("input_path", ""))
        file_type = params.get("file_type", "json")

        if file_type == "csv":
            worker = ServiceWorker(self.file_service.read_csv, path=input_path)
        elif file_type == "json":
            worker = ServiceWorker(self.file_service.load_json, path=input_path)
        else:
            worker = ServiceWorker(self.file_service.read_text_file, path=input_path)

        worker.signals.finished.connect(self._emit_file_operation_data)
        worker.signals.error.connect(self._handle_error)
        if self.thread_pool is not None:
            self.thread_pool.start(worker)

    # HELPER METHODS

    def _handle_error(self, error: Exception):
        """Central error handler"""
        error_msg = str(error)
        logger.error(f"Service error: {error_msg}", exc_info=True)
        self.error_occurred.emit(error_msg)

    def _batch_completed(self, results: dict):
        """Handle batch completion"""
        logger.info(f"Batch completed: {results}")
        self.batch_progress.emit(100, "Batch operation completed")
        # Emit specific signal based on operation
        if results.get("operation") == "summarize":
            self.batch_progress.emit(100, f"Summarized {results.get('completed', 0)} conversations")
        else:
            self.batch_progress.emit(100, f"Drafted {results.get('completed', 0)} replies")

    def _batch_progress_handler(self, current: int, total: int, conv_id: str, status: str):
        """Handle batch progress updates"""
        percentage = int((current / total) * 100) if total > 0 else 0
        message = f"Processing {conv_id}: {status} ({current}/{total})"
        self.batch_progress.emit(percentage, message)

    def _emit_file_operation_success(self, _result: Any):
        """Emit success payload for file operations."""
        self.file_operation_completed.emit({"success": True})

    def _emit_file_operation_data(self, data: Any):
        """Emit data payload produced by file operations."""
        self.file_operation_completed.emit({"data": data})

    def _config_updated(self, result: tuple[bool, str | None]):
        """Handle config update result"""
        success, error = result
        if success:
            self.config_updated.emit({"success": True})
            logger.info("Configuration updated successfully")
        else:
            self._handle_error(Exception(f"Config update failed: {error}"))

    # SERVICE STATUS METHODS

    def get_service_status(self) -> dict[str, Any]:
        """Get status of all services"""
        return {
            "search": self.search_service.validate_index()[0],
            "email": True,  # Email service is always ready
            "chunking": self.chunking_service.validate_export_root()[0],
            "indexing": self.indexing_service.validate_index()[0],
            "analysis": True,  # Analysis service is always ready
            "batch": True,  # Batch service is always ready
            "chat": self.chat_service.index_dir.exists(),
            "file": True,  # File service is always ready
            "config": self.config_service.current_config is not None
        }

    def cleanup(self):
        """Cleanup resources before shutdown"""
        logger.info("Cleaning up ServiceManager resources")
        # Save any pending chat sessions
        for session in self.chat_service.sessions.values():
            session.save()


class EmailOpsApplication(QApplication):
    """
    Main application class with proper initialization and theming.
    """

    def __init__(self, argv):
        super().__init__(argv)

        # Set application metadata
        self.setApplicationName("EmailOps Professional")
        self.setOrganizationName("EmailOps")
        self.setOrganizationDomain("emailops.ai")

        # Initialize settings
        self.settings = QSettings()

        # Apply theme
        self.theme_manager = ThemeManager()
        self.theme = self.theme_manager.theme
        self._apply_theme()

        # Initialize service manager
        config_path = Path.home() / ".emailops" / "config.json"
        self.service_manager = ServiceManager(config_path)

        # Setup global exception handler
        sys.excepthook = self._handle_exception

        logger.info("EmailOps Application initialized")

    def _apply_theme(self):
        """Apply modern dark theme with Material Design"""

        # Set Fusion style for modern look
        self.setStyle(QStyleFactory.create("Fusion"))

        # Create dark palette
        palette = QPalette()

        # Window colors
        palette.setColor(
            QPalette.ColorRole.Window,
            self.theme_manager.get_color("background"),
        )
        palette.setColor(
            QPalette.ColorRole.WindowText,
            self.theme_manager.get_color("text_primary"),
        )

        # Base colors (for input fields)
        palette.setColor(
            QPalette.ColorRole.Base,
            self.theme_manager.get_color("surface"),
        )
        palette.setColor(
            QPalette.ColorRole.AlternateBase, self.theme_manager.get_color("surface_elevated")
        )

        # Text colors
        palette.setColor(
            QPalette.ColorRole.Text,
            self.theme_manager.get_color("text_primary"),
        )
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.white)

        # Button colors
        palette.setColor(
            QPalette.ColorRole.Button,
            self.theme_manager.get_color("surface_elevated"),
        )
        palette.setColor(
            QPalette.ColorRole.ButtonText,
            self.theme_manager.get_color("text_primary"),
        )

        # Highlight colors
        palette.setColor(
            QPalette.ColorRole.Highlight,
            self.theme_manager.get_color("primary"),
        )
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)

        # Disabled colors
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.WindowText,
            self.theme_manager.get_color("text_disabled"),
        )
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.Text,
            self.theme_manager.get_color("text_disabled"),
        )
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.ButtonText,
            self.theme_manager.get_color("text_disabled"),
        )

        self.setPalette(palette)

        # Set application-wide stylesheet for fine control
        self.setStyleSheet(self._get_stylesheet())

        # Load custom fonts
        self._load_fonts()

    def _get_stylesheet(self) -> str:
        """Generate application stylesheet with Material Design principles"""
        return f"""
        /* Global Styles */
        QWidget {{
            font-family: 'Roboto', 'Segoe UI', sans-serif;
            font-size: 14px;
        }}

        /* Custom Scrollbars */
        QScrollBar:vertical {{
            background: {self.theme.surface};
            width: 12px;
            border-radius: 6px;
        }}

        QScrollBar::handle:vertical {{
            background: {self.theme.border};
            border-radius: 6px;
            min-height: 20px;
        }}

        QScrollBar::handle:vertical:hover {{
            background: {self.theme.primary_light};
        }}

        /* Buttons with hover effects */
        QPushButton {{
            background-color: {self.theme.primary};
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: 500;
            text-transform: uppercase;
        }}

        QPushButton:hover {{
            background-color: {self.theme.primary_light};
        }}

        QPushButton:pressed {{
            background-color: {self.theme.primary_dark};
        }}

        QPushButton:disabled {{
            background-color: {self.theme.surface_elevated};
            color: {self.theme.text_disabled};
        }}

        /* Input fields */
        QLineEdit, QTextEdit, QPlainTextEdit {{
            background-color: {self.theme.surface};
            border: 1px solid {self.theme.border};
            border-radius: 4px;
            padding: 8px;
            color: {self.theme.text_primary};
        }}

        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
            border: 2px solid {self.theme.primary};
        }}

        /* ComboBox styling */
        QComboBox {{
            background-color: {self.theme.surface};
            border: 1px solid {self.theme.border};
            border-radius: 4px;
            padding: 6px;
            min-width: 100px;
        }}

        QComboBox:hover {{
            border: 1px solid {self.theme.primary_light};
        }}

        QComboBox::drop-down {{
            border: none;
        }}

        QComboBox::down-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid {self.theme.text_secondary};
            margin-right: 8px;
        }}

        /* Tab Widget */
        QTabWidget::pane {{
            background-color: {self.theme.surface};
            border: 1px solid {self.theme.border};
            border-radius: 4px;
        }}

        QTabBar::tab {{
            background-color: {self.theme.surface};
            color: {self.theme.text_secondary};
            padding: 10px 20px;
            margin-right: 2px;
        }}

        QTabBar::tab:selected {{
            background-color: {self.theme.surface_elevated};
            color: {self.theme.primary};
            border-bottom: 3px solid {self.theme.primary};
        }}

        QTabBar::tab:hover {{
            background-color: {self.theme.surface_elevated};
            color: {self.theme.text_primary};
        }}

        /* Progress Bar */
        QProgressBar {{
            background-color: {self.theme.surface};
            border: 1px solid {self.theme.border};
            border-radius: 4px;
            text-align: center;
            color: {self.theme.text_primary};
        }}

        QProgressBar::chunk {{
            background-color: {self.theme.primary};
            border-radius: 3px;
        }}

        /* ToolTips */
        QToolTip {{
            background-color: {self.theme.surface_elevated};
            color: {self.theme.text_primary};
            border: 1px solid {self.theme.border};
            padding: 4px;
            border-radius: 4px;
        }}
        """

    def _load_fonts(self):
        """Load custom fonts for better typography"""
        # In production, you'd load actual font files
        # For now, we'll just set a nice font stack
        font = QFont("Segoe UI", 10)
        self.setFont(font)

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        """Global exception handler"""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )


def run_application():
    """Entry point for the EmailOps GUI application"""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and run application
    app = EmailOpsApplication(sys.argv)

    # Import and show main window (will create next)
    from .main_window import EmailOpsMainWindow

    window = EmailOpsMainWindow(app.service_manager)
    window.show()

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    run_application()
