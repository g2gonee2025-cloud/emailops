"""
EmailOps Modern GUI Launcher

Main entry point for the new PyQt6-based EmailOps application.
"""

import logging
import os
import sys

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QSplashScreen,
    QVBoxLayout,
    QWidget,
)

from emailops.gui.app import EmailOpsApplication
from emailops.gui.main_window import EmailOpsMainWindow as MainWindow

logger = logging.getLogger(__name__)


class SplashScreen(QSplashScreen):
    """
    Beautiful animated splash screen.
    """

    def __init__(self):
        super().__init__()

        # Create splash widget
        self.splash_widget = QWidget()
        self.splash_widget.setFixedSize(600, 400)
        self.splash_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #0F0F0F,
                    stop: 0.5 #1E1E1E,
                    stop: 1 #2A2A2A
                );
                border: 2px solid #1E88E5;
                border-radius: 16px;
            }
        """)

        # Layout
        layout = QVBoxLayout(self.splash_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        # Logo/Title
        title = QLabel("EmailOps")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            font-size: 48px;
            font-weight: bold;
            color: #1E88E5;
            letter-spacing: 4px;
            text-transform: uppercase;
        """)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Professional Email Operations Suite")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("""
            font-size: 16px;
            color: #999;
            font-style: italic;
        """)
        layout.addWidget(subtitle)

        # Version
        version = QLabel("Version 3.0 - Modern Edition")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setStyleSheet("""
            font-size: 12px;
            color: #666;
        """)
        layout.addWidget(version)

        # Spacer
        layout.addStretch()

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                background-color: #1E1E1E;
                border: 1px solid #333;
                border-radius: 8px;
                height: 8px;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #1E88E5,
                    stop: 1 #42A5F5
                );
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress)

        # Status label
        self.status = QLabel("Initializing...")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet("""
            font-size: 12px;
            color: #999;
        """)
        layout.addWidget(self.status)

        # Credits
        credits_label = QLabel("Built with ❤️ by EmailOps Team")
        credits_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        credits_label.setStyleSheet("""
            font-size: 10px;
            color: #555;
            margin-top: 20px;
        """)
        layout.addWidget(credits_label)

        # Set the widget
        self.setPixmap(QPixmap(600, 400))
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )

        # Center on screen
        self.center_on_screen()

        # Start loading animation
        self.loading_progress = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(50)

    def center_on_screen(self):
        """Center splash screen on display"""
        screen = QApplication.primaryScreen()
        if screen is not None:
            screen_rect = screen.availableGeometry()
            self.move(screen_rect.center().x() - 300, screen_rect.center().y() - 200)

    def update_progress(self):
        """Update loading progress"""
        self.loading_progress += 2
        self.progress.setValue(self.loading_progress)

        # Update status messages
        if self.loading_progress == 20:
            self.status.setText("Loading services...")
        elif self.loading_progress == 40:
            self.status.setText("Initializing UI components...")
        elif self.loading_progress == 60:
            self.status.setText("Connecting to backend...")
        elif self.loading_progress == 80:
            self.status.setText("Preparing workspace...")
        elif self.loading_progress >= 100:
            self.status.setText("Ready!")
            self.timer.stop()
            QTimer.singleShot(500, self.close)

    # Qt override: method name must remain camelCase for Qt event system
    def paintEvent(self, event):  # noqa: ARG002
        """Custom paint to show the widget"""
        from PyQt6.QtGui import QPainter

        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap())

        # Draw the custom widget
        self.splash_widget.render(painter)


class LoadingThread(QThread):
    """
    Background thread for loading heavy resources.
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal()

    def run(self):
        """Load resources in background"""
        try:
            # Simulate loading various components
            steps = [
                (10, "Initializing configuration..."),
                (20, "Loading service modules..."),
                (30, "Setting up database connections..."),
                (40, "Loading email templates..."),
                (50, "Initializing search index..."),
                (60, "Loading user preferences..."),
                (70, "Setting up authentication..."),
                (80, "Preparing UI components..."),
                (90, "Finalizing setup..."),
                (100, "Complete!"),
            ]

            for progress, message in steps:
                self.progress.emit(progress, message)
                self.msleep(200)  # Simulate work

            self.finished.emit()

        except Exception:
            logger.exception("Loading error")
            self.finished.emit()


class EmailOpsLauncher:
    """
    Main launcher class for EmailOps application.
    """

    def __init__(self):
        self.app: QApplication | None = None
        self.main_window: MainWindow | None = None
        self.splash: SplashScreen | None = None

    def launch(self) -> int:
        """
        Launch the EmailOps application.

        Returns:
            Application exit code
        """
        # Create main EmailOps application (inherits QApplication)
        self.email_ops_app = EmailOpsApplication(sys.argv)
        self.app = self.email_ops_app  # Keep reference for compatibility

        # Show splash screen
        self.splash = SplashScreen()
        self.splash.show()

        # Process events to show splash immediately
        self.email_ops_app.processEvents()

        # Create main window after splash
        QTimer.singleShot(3000, self._show_main_window)

        # Start event loop
        return self.email_ops_app.exec()


    def _show_main_window(self):
        """Show the main window after splash"""
        # Create and show main window - pass ServiceManager, not EmailOpsApplication
        self.main_window = MainWindow(self.email_ops_app.service_manager)
        if self.main_window is not None:
            self.main_window.show()

        # Close splash
        if self.splash:
            self.splash.close()
            self.splash = None

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        """Global exception handler"""
        import traceback

        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Log the exception
        error_msg = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        logger.error("Uncaught exception:\n%s", error_msg)

        # Show error dialog if app is running
        if self.app:
            from PyQt6.QtWidgets import QMessageBox

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Application Error")
            msg.setText("An unexpected error occurred")
            msg.setDetailedText(error_msg)
            msg.exec()


def main():
    """
    Main entry point for EmailOps GUI.
    """
    # Set high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Enable high DPI support
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    # Create and launch application
    launcher = EmailOpsLauncher()

    # Install exception handler
    sys.excepthook = launcher._handle_exception

    # Launch and exit with return code
    sys.exit(launcher.launch())


if __name__ == "__main__":
    main()
