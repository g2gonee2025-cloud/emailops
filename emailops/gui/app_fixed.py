"""
EmailOps Application - Fixed Version with Critical Issues Resolved

Fixes:
- Thread pool null reference crashes
- Blocking asyncio.run() calls replaced with proper async handling
- Added thread safety and proper error handling
- Resource cleanup and leak prevention
"""

import asyncio
import inspect
import logging
import sys
import threading
import weakref
from pathlib import Path
from typing import Any

from PyQt6.QtCore import (
    QObject,
    QRunnable,
    QSettings,
    Qt,
    QThreadPool,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QFont, QPalette
from PyQt6.QtWidgets import QApplication, QMessageBox, QStyleFactory

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


class AsyncEventLoopThread(threading.Thread):
    """Dedicated thread for running async operations without blocking GUI"""

    def __init__(self):
        super().__init__(daemon=True)
        self.loop: asyncio.AbstractEventLoop | None = None
        self._started = threading.Event()

    def run(self):
        """Run the event loop in this thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._started.set()
        self.loop.run_forever()

    def wait_started(self):
        """Wait for the event loop to start"""
        self._started.wait(timeout=5.0)

    def run_coroutine(self, coro):
        """Schedule a coroutine to run in this thread's event loop"""
        if not self.loop:
            raise RuntimeError("Event loop not started")
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future

    def stop(self):
        """Stop the event loop"""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)


class ServiceWorker(QRunnable):
    """
    Worker for running service operations in thread pool.
    Fixed to handle async properly without blocking.
    """

    class Signals(QObject):
        finished = pyqtSignal(object)
        error = pyqtSignal(Exception)
        progress = pyqtSignal(int, str)

    def __init__(
        self,
        func: Any,
        async_loop: AsyncEventLoopThread,
        *args: Any,
        **kwargs: Any
    ):
        """
        Initialize service worker with function, async loop, and arguments.

        Args:
            func: Function to execute (sync or async)
            async_loop: Dedicated async event loop thread
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func
        """
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = self.Signals()
        self.async_loop = async_loop
        self.setAutoDelete(True)
        self._cancelled = False

    def run(self):
        """Execute the service function, handling both sync and async methods."""
        if self._cancelled:
            return

        try:
            # Add progress callback adapter if supported
            if "progress_callback" in self.kwargs:
                def progress_adapter(*args):
                    if self._cancelled:
                        return
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

            # Check if the function is async
            if inspect.iscoroutinefunction(self.func):
                # Run async function in dedicated event loop thread
                future = self.async_loop.run_coroutine(
                    self.func(*self.args, **self.kwargs)
                )
                result = future.result(timeout=300)  # 5 minute timeout
            else:
                # Run sync function normally
                result = self.func(*self.args, **self.kwargs)

            if not self._cancelled:
                self.signals.finished.emit(result)
        except Exception as e:
            if not self._cancelled:
                logger.error(f"Service worker error: {e}", exc_info=True)
                self.signals.error.emit(e)

    def cancel(self):
        """Cancel the worker"""
        self._cancelled = True


class ThreadSafeServiceManager(QObject):
    """
    Central service manager with thread safety and proper error handling.
    """

    # Signals for GUI updates
    search_completed = pyqtSignal(list)
    email_generated = pyqtSignal(dict)
    index_updated = pyqtSignal(dict)
    chunk_completed = pyqtSignal(dict)
    batch_progress = pyqtSignal(int, str)
    chat_response = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    user_notification = pyqtSignal(str, str)  # title, message

    # Service signals
    chunks_updated = pyqtSignal(dict)
    analysis_completed = pyqtSignal(dict)
    config_updated = pyqtSignal(dict)
    file_operation_completed = pyqtSignal(dict)

    def __init__(self, config_path: Path):
        super().__init__()

        # Thread safety
        self._lock = threading.RLock()
        self._workers = weakref.WeakSet()
        self._shutting_down = False

        # Async event loop thread
        self.async_loop = AsyncEventLoopThread()
        self.async_loop.start()
        self.async_loop.wait_started()

        # Initialize thread pool with proper error handling
        self.thread_pool = QThreadPool.globalInstance()
        if self.thread_pool is None:
            self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)

        # Health check timer
        self.health_timer = QTimer()
        self.health_timer.timeout.connect(self._health_check)
        self.health_timer.start(30000)  # Every 30 seconds

        # Initialize services with validation
        self._initialize_services(config_path)

    def _initialize_services(self, config_path: Path):
        """Initialize all services with proper error handling"""
        try:
            # Validate and create export root
            export_root_path = config_path.parent
            export_root_path.mkdir(parents=True, exist_ok=True)

            if not export_root_path.is_dir():
                raise ValueError(f"Export root is not a directory: {export_root_path}")

            export_root = str(export_root_path)

            # Initialize services
            self.search_service = SearchService(export_root)
            self.email_service = EmailService(export_root)
            self.chunking_service = ChunkingService(export_root)
            self.indexing_service = IndexingService(export_root)
            self.analysis_service = AnalysisService(export_root)
            self.batch_service = BatchService(export_root)
            self.chat_service = ChatService(export_root)
            self.file_service = FileService(export_root)
            self.config_service = ConfigService(config_path)

            logger.info("ServiceManager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize services: {e}", exc_info=True)
            self.user_notification.emit(
                "Initialization Error",
                f"Failed to initialize services: {e!s}\nThe application may not function properly."
            )
            # Initialize with None to prevent crashes
            self.search_service = None
            self.email_service = None
            self.chunking_service = None
            self.indexing_service = None
            self.analysis_service = None
            self.batch_service = None
            self.chat_service = None
            self.file_service = None
            self.config_service = None

    def _health_check(self):
        """Periodic health check of services"""
        if self._shutting_down:
            return

        with self._lock:
            try:
                # Check thread pool health
                if self.thread_pool is None or self.thread_pool.maxThreadCount() == 0:
                    logger.error("Thread pool unhealthy, attempting recovery")
                    self.thread_pool = QThreadPool()
                    self.thread_pool.setMaxThreadCount(4)

                # Check async loop health
                if not self.async_loop.is_alive():
                    logger.error("Async loop died, restarting")
                    self.async_loop = AsyncEventLoopThread()
                    self.async_loop.start()
                    self.async_loop.wait_started()

            except Exception as e:
                logger.error(f"Health check failed: {e}")

    def _submit_worker(self, worker: ServiceWorker):
        """Safely submit a worker to the thread pool"""
        if self._shutting_down:
            logger.warning("Cannot submit worker, service manager shutting down")
            return False

        with self._lock:
            if self.thread_pool is None:
                logger.error("Thread pool is None, cannot submit worker")
                self._handle_error(RuntimeError("Thread pool not available"))
                return False

            try:
                self._workers.add(worker)
                self.thread_pool.start(worker)
                return True
            except Exception as e:
                logger.error(f"Failed to submit worker: {e}")
                self._handle_error(e)
                return False

    @pyqtSlot(dict)
    def perform_search(self, params: dict):
        """Execute search operation with full validation"""
        if not self.search_service:
            self._handle_error(RuntimeError("Search service not available"))
            return

        # Validate parameters
        if not isinstance(params, dict):
            self._handle_error(
                TypeError(f"params must be dict, got {type(params)}"),
            )
            return

        query = params.get("query", "")
        if not isinstance(query, str) or not query.strip():
            self._handle_error(ValueError("Search query must be a non-empty string"))
            return

        # Validate numeric parameters
        k = params.get("k", 10)
        if not isinstance(k, int) or k < 1 or k > 100:
            self._handle_error(ValueError("k must be integer between 1 and 100"))
            return

        mmr_lambda = params.get("mmr_lambda", 0.7)
        if not isinstance(mmr_lambda, (int, float)) or not 0 <= mmr_lambda <= 1:
            self._handle_error(ValueError("mmr_lambda must be between 0 and 1"))
            return

        # Build search filters
        filters = params.get("filters", {})
        search_filters = None
        if filters and any(filters.values()):
            try:
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
            except Exception as e:
                self._handle_error(ValueError(f"Invalid search filters: {e}"))
                return

        # Create and submit worker
        worker = ServiceWorker(
            self.search_service.perform_search,
            self.async_loop,
            query=query,
            k=k,
            provider=params.get("provider", "vertex"),
            mmr_lambda=mmr_lambda,
            rerank_alpha=params.get("rerank_alpha", 0.35),
            filters=search_filters,
        )

        worker.signals.finished.connect(self.search_completed.emit)
        worker.signals.error.connect(self._handle_error)
        self._submit_worker(worker)

    @pyqtSlot(dict)
    def generate_email(self, params: dict):
        """Generate email draft with proper validation"""
        if not self.email_service:
            self._handle_error(RuntimeError("Email service not available"))
            return

        operation = params.get("operation", "reply")

        if operation == "reply":
            # Validate reply parameters
            conv_id = params.get("conv_id", "")
            if not conv_id:
                self._handle_error(ValueError("Conversation ID required for reply"))
                return

            worker = ServiceWorker(
                self.email_service.draft_reply,
                self.async_loop,
                conv_id=conv_id,
                query=params.get("query", ""),
                provider=params.get("provider", "vertex"),
                sim_threshold=params.get("sim_threshold", 0.5),
                target_tokens=params.get("target_tokens", 20000),
                temperature=params.get("temperature", 0.7),
                reply_policy=params.get("reply_policy", "reply_all"),
                include_attachments=params.get("include_attachments", True),
            )
        else:
            # Validate fresh email parameters
            to_list = params.get("to_list", [])
            if not to_list:
                self._handle_error(ValueError("Recipients required for new email"))
                return

            worker = ServiceWorker(
                self.email_service.draft_fresh_email,
                self.async_loop,
                to_list=to_list,
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
        self._submit_worker(worker)

    @pyqtSlot(dict)
    def build_index(self, params: dict):
        """Build index with atomic operations"""
        if not self.indexing_service:
            self._handle_error(RuntimeError("Indexing service not available"))
            return

        worker = ServiceWorker(
            self.indexing_service.build_index,
            self.async_loop,
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
        self._submit_worker(worker)

    @pyqtSlot(dict)
    def analyze_conversation(self, params: dict):
        """Analyze conversation with proper validation"""
        if not self.analysis_service:
            self._handle_error(RuntimeError("Analysis service not available"))
            return

        conv_id = params.get("conv_id")

        if conv_id:
            # Single conversation analysis
            if not isinstance(conv_id, str) or not conv_id.strip():
                self._handle_error(ValueError("Invalid conversation ID"))
                return

            # Validate conversation exists
            if self.file_service:
                conv_path = self.file_service.get_conversation_path(conv_id)
                if not conv_path:
                    self._handle_error(ValueError(f"Conversation not found: {conv_id}"))
                    return

                worker = ServiceWorker(
                    self.analysis_service.analyze_conversation,
                    self.async_loop,
                    thread_dir=conv_path,
                    temperature=params.get("temperature", 0.7),
                    merge_manifest=params.get("merge_manifest", True),
                    output_format=params.get("output_format", "json"),
                )
            else:
                self._handle_error(RuntimeError("File service not available"))
                return
        else:
            # Batch analysis
            if not self.batch_service:
                self._handle_error(RuntimeError("Batch service not available"))
                return

            conv_ids = params.get("conv_ids", [])
            if not conv_ids:
                self._handle_error(ValueError("No conversations specified"))
                return

            worker = ServiceWorker(
                self.batch_service.batch_summarize,
                self.async_loop,
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
        self._submit_worker(worker)

    def _handle_error(self, error: Exception):
        """Central error handler with user notification"""
        error_msg = str(error)
        logger.error(f"Service error: {error_msg}", exc_info=True)

        # Emit error signal for status updates
        self.error_occurred.emit(error_msg)

        # Emit user notification for important errors
        if isinstance(error, (RuntimeError, ValueError, TypeError)):
            self.user_notification.emit("Operation Failed", error_msg)

    def get_service_status(self) -> dict[str, Any]:
        """Get status of all services with null checks"""
        with self._lock:
            return {
                "search": self.search_service.validate_index()[0] if self.search_service else False,
                "email": self.email_service is not None,
                "chunking": (
                    self.chunking_service.validate_export_root()[0]
                    if self.chunking_service
                    else False
                ),
                "indexing": (
                    self.indexing_service.validate_index()[0]
                    if self.indexing_service
                    else False
                ),
                "analysis": self.analysis_service is not None,
                "batch": self.batch_service is not None,
                "chat": (
                    self.chat_service.index_dir.exists()
                    if self.chat_service
                    else False
                ),
                "file": self.file_service is not None,
                "config": (
                    self.config_service.current_config is not None
                    if self.config_service
                    else False
                ),
                "thread_pool": self.thread_pool is not None,
                "async_loop": self.async_loop.is_alive(),
            }

    def cleanup(self):
        """Cleanup resources with proper error handling"""
        logger.info("Starting ServiceManager cleanup")
        self._shutting_down = True

        with self._lock:
            # Cancel pending workers
            for worker in self._workers:
                try:
                    worker.cancel()
                except Exception as e:
                    logger.error(f"Error cancelling worker: {e}")

            # Stop health timer
            if hasattr(self, 'health_timer'):
                self.health_timer.stop()

            # Save chat sessions
            if self.chat_service:
                try:
                    for session in self.chat_service.sessions.values():
                        session.save()
                except Exception as e:
                    logger.error(f"Error saving chat sessions: {e}")

            # Stop async loop
            if hasattr(self, 'async_loop'):
                try:
                    self.async_loop.stop()
                    self.async_loop.join(timeout=2.0)
                except Exception as e:
                    logger.error(f"Error stopping async loop: {e}")

            # Wait for thread pool
            if self.thread_pool:
                try:
                    self.thread_pool.waitForDone(5000)
                except Exception as e:
                    logger.error(f"Error waiting for thread pool: {e}")

        logger.info("ServiceManager cleanup complete")


class EmailOpsApplication(QApplication):
    """
    Main application class with improved error handling.
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

        # Initialize service manager with error handling
        try:
            config_path = Path.home() / ".emailops" / "config.json"
            self.service_manager = ThreadSafeServiceManager(config_path)

            # Connect notification signal
            self.service_manager.user_notification.connect(self._show_notification)

        except Exception as e:
            logger.critical(f"Failed to initialize service manager: {e}", exc_info=True)
            QMessageBox.critical(
                None,
                "Initialization Failed",
                f"Failed to initialize EmailOps:\n{e!s}\n\nThe application will exit."
            )
            sys.exit(1)

        # Setup global exception handler
        sys.excepthook = self._handle_exception

        logger.info("EmailOps Application initialized")

    def _show_notification(self, title: str, message: str):
        """Show user notification"""
        QMessageBox.warning(None, title, message)

    def _apply_theme(self):
        """Apply modern dark theme with Material Design"""
        self.setStyle(QStyleFactory.create("Fusion"))

        palette = QPalette()

        # Apply theme colors
        palette.setColor(
            QPalette.ColorRole.Window,
            self.theme_manager.get_color("background"),
        )
        palette.setColor(
            QPalette.ColorRole.WindowText,
            self.theme_manager.get_color("text_primary"),
        )
        palette.setColor(
            QPalette.ColorRole.Base,
            self.theme_manager.get_color("surface"),
        )
        palette.setColor(
            QPalette.ColorRole.AlternateBase,
            self.theme_manager.get_color("surface_elevated"),
        )
        palette.setColor(
            QPalette.ColorRole.Text,
            self.theme_manager.get_color("text_primary"),
        )
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.white)
        palette.setColor(
            QPalette.ColorRole.Button,
            self.theme_manager.get_color("surface_elevated"),
        )
        palette.setColor(
            QPalette.ColorRole.ButtonText,
            self.theme_manager.get_color("text_primary"),
        )
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

        # Load custom fonts
        font = QFont("Segoe UI", 10)
        self.setFont(font)

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        """Global exception handler"""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

        # Show error dialog
        QMessageBox.critical(
            None,
            "Application Error",
            f"An unexpected error occurred:\n{exc_type.__name__}: {exc_value}\n\n"
            "Please check the logs for more details."
        )


def run_application():
    """Entry point for the EmailOps GUI application"""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Path.home() / ".emailops" / "emailops.log"),
            logging.StreamHandler()
        ]
    )

    # Create and run application
    app = EmailOpsApplication(sys.argv)

    # Import and show main window
    from .main_window import EmailOpsMainWindow

    window = EmailOpsMainWindow(app.service_manager)
    window.show()

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    run_application()
