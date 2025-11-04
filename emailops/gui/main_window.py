"""
EmailOps Main Window - Professional Dashboard Interface

A modern, beautiful email operations center with Material Design.
Features smooth animations, responsive layout, and clean architecture.
"""

import logging
from datetime import datetime

from PyQt6.QtCore import (
    Qt,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import (
    QAction,
    QColor,
    QIcon,
)
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QSystemTrayIcon,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .analysis_panel import AnalysisPanel

# Import the new panels
from .chunking_panel import ChunkingPanel
from .components import ChatInterface, EmailComposer
from .config_panel import ConfigPanel
from .file_panel import FilePanel
from .search_panel import SearchPanel

logger = logging.getLogger(__name__)


class AnimatedCard(QFrame):
    """
    Beautiful card widget with hover animations and shadows.
    Used throughout the UI for a consistent, modern look.
    """

    clicked = pyqtSignal()

    def __init__(self, title: str, subtitle: str = "", icon: QIcon | None = None):
        super().__init__()
        self.setObjectName("AnimatedCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Setup card appearance
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            #AnimatedCard {
                background-color: #1E1E1E;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 16px;
            }
            #AnimatedCard:hover {
                background-color: #2A2A2A;
                border: 1px solid #1E88E5;
            }
        """)

        # Add drop shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

        # Layout
        layout = QVBoxLayout(self)

        # Header with icon
        header_layout = QHBoxLayout()

        if icon:
            icon_label = QLabel()
            icon_label.setPixmap(icon.pixmap(32, 32))
            header_layout.addWidget(icon_label)

        # Title and subtitle
        text_layout = QVBoxLayout()

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("""
            font-size: 16px;
            font-weight: 600;
            color: #E0E0E0;
        """)
        text_layout.addWidget(self.title_label)

        if subtitle:
            self.subtitle_label = QLabel(subtitle)
            self.subtitle_label.setStyleSheet("""
                font-size: 12px;
                color: #B0B0B0;
            """)
            text_layout.addWidget(self.subtitle_label)

        header_layout.addLayout(text_layout)
        header_layout.addStretch()

        layout.addLayout(header_layout)

    # Qt override: method name must remain camelCase for Qt event system
    def mousePressEvent(self, event):
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)




class DashboardWidget(QWidget):
    """
    Main dashboard with statistics and quick actions.
    """

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Welcome header
        header = QLabel("📊 EmailOps Dashboard")
        header.setStyleSheet("""
            font-size: 24px;
            font-weight: 700;
            color: #1E88E5;
            padding: 16px 0;
        """)
        layout.addWidget(header)

        # Statistics cards
        stats_layout = QHBoxLayout()

        # Total emails card
        self.emails_card = AnimatedCard(
            "Total Emails",
            "0 emails processed",
            QIcon(),  # Would load actual icon
        )
        stats_layout.addWidget(self.emails_card)

        # Active conversations card
        self.convs_card = AnimatedCard("Conversations", "0 active threads", QIcon())
        stats_layout.addWidget(self.convs_card)

        # Index status card
        self.index_card = AnimatedCard("Index Status", "Ready", QIcon())
        stats_layout.addWidget(self.index_card)

        layout.addLayout(stats_layout)

        # Quick actions
        actions_label = QLabel("Quick Actions")
        actions_label.setStyleSheet("""
            font-size: 18px;
            font-weight: 600;
            margin-top: 24px;
            margin-bottom: 16px;
        """)
        layout.addWidget(actions_label)

        actions_layout = QHBoxLayout()

        # Action buttons with icons
        self.build_index_btn = self._create_action_button(
            "🔨 Build Index", "Update the search index"
        )
        actions_layout.addWidget(self.build_index_btn)

        self.draft_email_btn = self._create_action_button(
            "✉️ Draft Email", "Create a new email"
        )
        actions_layout.addWidget(self.draft_email_btn)

        self.batch_process_btn = self._create_action_button(
            "⚡ Batch Process", "Process multiple items"
        )
        actions_layout.addWidget(self.batch_process_btn)

        layout.addLayout(actions_layout)

        # Recent activity
        activity_label = QLabel("Recent Activity")
        activity_label.setStyleSheet("""
            font-size: 18px;
            font-weight: 600;
            margin-top: 24px;
            margin-bottom: 16px;
        """)
        layout.addWidget(activity_label)

        self.activity_list = QTreeWidget()
        self.activity_list.setHeaderLabels(["Time", "Action", "Status"])
        self.activity_list.setAlternatingRowColors(True)
        self.activity_list.setStyleSheet("""
            QTreeWidget {
                background-color: #1E1E1E;
                border: 1px solid #333;
                border-radius: 8px;
            }
        """)
        layout.addWidget(self.activity_list)

        layout.addStretch()

    def _create_action_button(self, text: str, tooltip: str) -> QPushButton:
        """Create a styled action button"""
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #2A2A2A;
                color: #E0E0E0;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 16px;
                font-size: 14px;
                font-weight: 600;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #333;
                border-color: #1E88E5;
            }
            QPushButton:pressed {
                background-color: #1E1E1E;
            }
        """)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        return btn

    def add_activity(self, action: str, status: str):
        """Add item to activity list"""
        time_str = datetime.now().strftime("%H:%M:%S")
        item = QTreeWidgetItem([time_str, action, status])

        # Color code based on status
        if status == "Success":
            item.setForeground(2, QColor("#4CAF50"))
        elif status == "Error":
            item.setForeground(2, QColor("#F44336"))
        else:
            item.setForeground(2, QColor("#FF9800"))

        self.activity_list.insertTopLevelItem(0, item)

        # Keep only last 50 items
        while self.activity_list.topLevelItemCount() > 50:
            self.activity_list.takeTopLevelItem(50)


class EmailOpsMainWindow(QMainWindow):
    """
    Main application window with modern navigation and beautiful UI.
    """

    def __init__(self, service_manager):
        super().__init__()
        self.service_manager = service_manager
        self.service_health_status = {}
        self._init_ui()
        self._connect_signals()
        self._check_service_health()
        self._start_animations()

    def _init_ui(self):
        """Initialize the user interface"""

        # Window setup
        self.setWindowTitle("EmailOps Professional")
        self.setGeometry(100, 100, 1400, 900)

        # Set window icon (would load actual icon in production)
        # self.setWindowIcon(QIcon("assets/icon.png"))

        # Central widget with main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar navigation
        self.sidebar = self._create_sidebar()
        main_layout.addWidget(self.sidebar)

        # Main content area
        self.content_stack = QStackedWidget()
        self.content_stack.setStyleSheet("""
            QStackedWidget {
                background-color: #121212;
                border: none;
            }
        """)

        # Add main panels
        self.dashboard = DashboardWidget()
        self.content_stack.addWidget(self.dashboard)

        self.search_panel = SearchPanel()
        self.content_stack.addWidget(self.search_panel)

        # Email panel with composer
        self.email_panel = EmailComposer()
        self.content_stack.addWidget(self.email_panel)

        # Chat panel with interface
        self.chat_panel = ChatInterface()
        self.content_stack.addWidget(self.chat_panel)

        # Batch panel (keep as placeholder for now)
        self.batch_panel = QWidget()  # BatchProcessor
        self.content_stack.addWidget(self.batch_panel)

        # Settings panel replaced with ConfigPanel
        self.settings_panel = ConfigPanel()
        self.content_stack.addWidget(self.settings_panel)

        # Add new service panels
        self.chunking_panel = ChunkingPanel()
        self.content_stack.addWidget(self.chunking_panel)

        self.analysis_panel = AnalysisPanel()
        self.content_stack.addWidget(self.analysis_panel)

        self.file_panel = FilePanel()
        self.content_stack.addWidget(self.file_panel)

        main_layout.addWidget(self.content_stack)

        # Status bar
        self._create_status_bar()

        # System tray icon - Disabled to prevent "No Icon set" warning
        # Re-enable when icon file is available
        # self._create_tray_icon()

    def _create_sidebar(self) -> QWidget:
        """Create beautiful sidebar navigation"""
        sidebar = QFrame()
        sidebar.setFixedWidth(240)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #1E1E1E;
                border-right: 1px solid #333;
            }
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Logo/Title
        title = QLabel("EmailOps")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: 700;
                color: #1E88E5;
                padding: 24px;
                background-color: #161616;
            }
        """)
        layout.addWidget(title)

        # Navigation buttons
        nav_buttons = [
            ("📊 Dashboard", 0),
            ("🔍 Search", 1),
            ("✉️ Compose", 2),
            ("💬 Chat", 3),
            ("⚡ Batch", 4),
            ("⚙️ Settings", 5),
            ("🔧 Chunking", 6),
            ("📊 Analysis", 7),
            ("📁 Files", 8),
        ]

        self.nav_buttons = []
        for text, index in nav_buttons:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #B0B0B0;
                    border: none;
                    padding: 16px 24px;
                    text-align: left;
                    font-size: 14px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: #2A2A2A;
                    color: #E0E0E0;
                }
                QPushButton:checked {
                    background-color: #1E88E5;
                    color: white;
                }
            """)
            btn.clicked.connect(lambda _checked, idx=index: self._navigate_to(idx))
            layout.addWidget(btn)
            self.nav_buttons.append(btn)

        # Set dashboard as default
        self.nav_buttons[0].setChecked(True)

        layout.addStretch()

        # User info at bottom
        user_info = QFrame()
        user_info.setStyleSheet("""
            QFrame {
                background-color: #161616;
                border-top: 1px solid #333;
                padding: 16px;
            }
        """)

        user_layout = QVBoxLayout(user_info)
        user_label = QLabel("User")
        user_label.setStyleSheet("color: #B0B0B0; font-size: 12px;")
        email_label = QLabel("user@emailops.ai")
        email_label.setStyleSheet("color: #E0E0E0; font-weight: 600;")

        user_layout.addWidget(user_label)
        user_layout.addWidget(email_label)

        layout.addWidget(user_info)

        return sidebar

    def _navigate_to(self, index: int):
        """Navigate to different panels"""
        # Update button states
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

        # Animate transition
        self.content_stack.setCurrentIndex(index)

        # Load conversations for batch panels
        if index == 6:  # Chunking panel
            self._load_conversations_for_panel(self.chunking_panel)
        elif index == 7:  # Analysis panel
            self._load_conversations_for_panel(self.analysis_panel)

        # Update status
        panel_names = [
            "Dashboard",
            "Search",
            "Compose",
            "Chat",
            "Batch",
            "Settings",
            "Chunking",
            "Analysis",
            "Files",
        ]
        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.showMessage(f"Switched to {panel_names[index]}", 2000)

    def _create_status_bar(self):
        """Create status bar with progress indicator"""
        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.setStyleSheet("""
                QStatusBar {
                    background-color: #1E1E1E;
                    color: #B0B0B0;
                    border-top: 1px solid #333;
                }
            """)

            # Progress bar
            self.progress_bar = QProgressBar()
            self.progress_bar.setMaximumWidth(200)
            self.progress_bar.setVisible(False)
            status_bar.addPermanentWidget(self.progress_bar)

            # Connection status
            self.connection_label = QLabel("● Connected")
            self.connection_label.setStyleSheet("color: #4CAF50;")
            status_bar.addPermanentWidget(self.connection_label)

            status_bar.showMessage("Ready", 5000)

    def _create_tray_icon(self):
        """Create system tray icon"""
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_icon = QSystemTrayIcon(self)
            # self.tray_icon.setIcon(QIcon("assets/icon.png"))

            # Tray menu
            tray_menu = QMenu()

            show_action = QAction("Show", self)
            show_action.triggered.connect(self.show)
            tray_menu.addAction(show_action)

            quit_action = QAction("Quit", self)
            quit_action.triggered.connect(self.close)
            tray_menu.addAction(quit_action)

            self.tray_icon.setContextMenu(tray_menu)
            self.tray_icon.show()

    def _connect_signals(self):
        """Connect service signals to UI updates"""

        # Search signals
        self.search_panel.search_requested.connect(self.service_manager.perform_search)
        self.service_manager.search_completed.connect(self.search_panel.display_results)

        # Dashboard actions
        self.dashboard.build_index_btn.clicked.connect(
            lambda: self.service_manager.build_index({
                "provider": "vertex",
                "batch_size": 64,
                "num_workers": 4,
                "force": False
            })
        )

        # Email panel connections
        self.email_panel.email_sent.connect(
            lambda data: self.service_manager.generate_email({"operation": "fresh", **data})
        )

        # Chat panel connections
        self.chat_panel.message_sent.connect(
            lambda msg: self.service_manager.send_chat({"message": msg})
        )
        self.service_manager.chat_response.connect(
            lambda resp: self.chat_panel.add_message(resp.get("answer", ""), is_user=False)
        )

        # Settings/Config panel connections
        self.settings_panel.config_update_requested.connect(self.service_manager.update_config)
        self.settings_panel.config_get_requested.connect(self.service_manager.get_config)
        self.service_manager.config_updated.connect(self.settings_panel.on_config_updated)

        # Chunking panel connections
        self.chunking_panel.chunk_requested.connect(self.service_manager.chunk_conversations)
        self.service_manager.chunks_updated.connect(self.chunking_panel.on_chunks_updated)

        # Analysis panel connections
        self.analysis_panel.analysis_requested.connect(self.service_manager.analyze_conversation)
        self.service_manager.analysis_completed.connect(self.analysis_panel.on_analysis_completed)

        # File panel connections - FIXED: Route based on operation type
        def handle_file_operation(params: dict):
            """Route file operations to correct handler based on operation type"""
            operation = params.get("operation", "")
            is_export = operation.startswith("export") or operation in {"save_json", "save_text"}
            is_import = operation.startswith("import") or operation in {"load_json", "read_csv"}

            if is_export:
                self.service_manager.export_data(params)
            elif is_import:
                self.service_manager.import_data(params)
            else:
                logger.error(f"Unknown file operation: {operation}")
                self._show_error(f"Unknown file operation: {operation}")

        self.file_panel.file_operation_requested.connect(handle_file_operation)
        self.service_manager.file_operation_completed.connect(self.file_panel.on_file_operation_completed)

        # Error handling
        self.service_manager.error_occurred.connect(self._show_error)

        # Progress updates
        self.service_manager.batch_progress.connect(self._update_progress)

    def _load_conversations_for_panel(self, panel):
        """Load available conversations for batch operations panel"""
        try:
            # Get conversation list from analysis service
            export_root = self.service_manager.analysis_service.export_root

            if not export_root.exists():
                logger.warning(f"Export root does not exist: {export_root}")
                return

            # Scan for conversation directories
            conv_ids = []
            for item in export_root.iterdir():
                if item.is_dir() and (item / "Conversation.txt").exists():
                    conv_ids.append(item.name)

            # Sort alphabetically
            conv_ids.sort()

            # Load into panel
            if conv_ids:
                panel.load_conversations(conv_ids)
                logger.info(f"Loaded {len(conv_ids)} conversations for {panel.__class__.__name__}")
            else:
                logger.warning("No conversations found in export root")

        except Exception as e:
            logger.error(f"Failed to load conversations: {e}", exc_info=True)
            self._show_error(f"Failed to load conversations: {e}")

    def _check_service_health(self):
        """Check health of all backend services on startup"""
        try:
            logger.info("Performing service health checks...")
            self.service_health_status = self.service_manager.get_service_status()

            # Count healthy services
            healthy = sum(1 for status in self.service_health_status.values() if status)
            total = len(self.service_health_status)

            # Update connection status
            if healthy == total:
                self.connection_label.setText("● All Services Ready")
                self.connection_label.setStyleSheet("color: #4CAF50;")
                logger.info(f"All {total} services operational")
            elif healthy > 0:
                self.connection_label.setText(f"⚠ {healthy}/{total} Services Ready")
                self.connection_label.setStyleSheet("color: #FF9800;")
                logger.warning(f"Only {healthy}/{total} services operational")

                # Show warning dialog
                failed_services = [
                    name
                    for name, status in self.service_health_status.items()
                    if not status
                ]
                QMessageBox.warning(
                    self,
                    "Service Health Warning",
                    (
                        "Some services are not ready:\n"
                        + "\n".join(f"  • {svc}" for svc in failed_services)
                        + "\n\nFunctionality may be limited."
                    ),
                )
            else:
                self.connection_label.setText("● No Services Available")
                self.connection_label.setStyleSheet("color: #F44336;")
                logger.error("No services operational - application non-functional")

                QMessageBox.critical(
                    self,
                    "Service Health Critical",
                    "No backend services are available.\n\n"
                    "The application cannot function without backend services.\n"
                    "Please check configuration and try again."
                )

        except Exception as e:
            logger.error(f"Service health check failed: {e}", exc_info=True)
            self.connection_label.setText("● Health Check Failed")
            self.connection_label.setStyleSheet("color: #F44336;")

    def _start_animations(self):
        """Start UI animations"""
        # Service health check timer (every 30 seconds)
        self.health_check_timer = QTimer()
        self.health_check_timer.timeout.connect(self._update_connection_status)
        self.health_check_timer.start(30000)  # 30 seconds

    def _update_connection_status(self):
        """Update connection indicator with actual service health"""
        try:
            status = self.service_manager.get_service_status()
            healthy = sum(1 for s in status.values() if s)
            total = len(status)

            if healthy == total:
                self.connection_label.setText("● All Services Ready")
                self.connection_label.setStyleSheet("color: #4CAF50;")
            elif healthy > 0:
                self.connection_label.setText(f"⚠ {healthy}/{total} Services Ready")
                self.connection_label.setStyleSheet("color: #FF9800;")
            else:
                self.connection_label.setText("● No Services Available")
                self.connection_label.setStyleSheet("color: #F44336;")

        except Exception as e:
            logger.error(f"Connection status update failed: {e}")
            self.connection_label.setText("● Status Unknown")
            self.connection_label.setStyleSheet("color: #666;")

    @pyqtSlot(str)
    def _show_error(self, message: str):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)
        self.dashboard.add_activity("Error", "Error")
        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.showMessage(f"Error: {message}", 5000)

    @pyqtSlot(int, str)
    def _update_progress(self, value: int, message: str):
        """Update progress bar"""
        if value == 0:
            self.progress_bar.setVisible(True)
        elif value >= 100:
            self.progress_bar.setVisible(False)

        self.progress_bar.setValue(value)
        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.showMessage(message, 2000)

    # Qt override: method name must remain camelCase for Qt event system
    def closeEvent(self, event):
        """Handle window close event - PROPER CLEANUP ADDED"""
        # Tray icon disabled, so just do cleanup
        if event is not None:
            # Cleanup resources before closing
            logger.info("Cleaning up application resources...")

            # Stop all timers
            if hasattr(self, 'health_check_timer'):
                self.health_check_timer.stop()

            # Cleanup service manager
            try:
                self.service_manager.cleanup()
            except Exception as e:
                logger.error(f"Service cleanup error: {e}", exc_info=True)

            # Wait for thread pool to finish
            thread_pool = self.service_manager.thread_pool
            if thread_pool is not None:
                logger.info("Waiting for worker threads to complete...")
                thread_pool.waitForDone(5000)  # Wait max 5 seconds

            event.accept()
            logger.info("Application shutdown complete")
