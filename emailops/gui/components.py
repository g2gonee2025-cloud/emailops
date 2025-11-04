"""
EmailOps UI Components - Reusable Modern Widgets

Beautiful, animated components following Material Design principles.
"""

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPen,
)
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .theme import ThemeManager


class AnimatedButton(QPushButton):
    """
    Modern button with hover animations and ripple effects.
    """

    def __init__(self, text: str, primary: bool = False):
        super().__init__(text)
        self.primary = primary
        self.theme_manager = ThemeManager()
        self._setup_style()
        self._setup_animations()

    def _setup_style(self):
        """Apply Material Design styling"""
        self.setStyleSheet(self.theme_manager.get_stylesheet("AnimatedButton"))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _setup_animations(self):
        """Setup hover and click animations"""
        self.opacity_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)

        self.hover_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.hover_animation.setDuration(200)
        self.hover_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

    def enterEvent(self, event):
        """Animate on hover"""
        self.hover_animation.setStartValue(1.0)
        self.hover_animation.setEndValue(0.9)
        self.hover_animation.start()
        super().enterEvent(event)

    # Qt override: method name must remain camelCase for Qt event system
    def leaveEvent(self, event):
        """Animate on leave"""
        self.hover_animation.setStartValue(0.9)
        self.hover_animation.setEndValue(1.0)
        self.hover_animation.start()
        super().leaveEvent(event)


class MaterialCard(QFrame):
    """
    Material Design card with elevation and hover effects.
    """

    def __init__(
        self,
        title: str = "",
        content: QWidget | None = None,
        content_layout: QLayout | None = None,
    ):
        super().__init__()
        self.setObjectName("MaterialCard")
        self.theme_manager = ThemeManager()
        self._setup_ui(title, content, content_layout)

    def _setup_ui(self, title: str, content: QWidget | None, content_layout: QLayout | None):
        """Setup card UI - Fixed to prevent double layout addition"""
        self.setStyleSheet(self.theme_manager.get_stylesheet("MaterialCard"))

        layout = QVBoxLayout(self)

        if title:
            title_label = QLabel(title)
            title_label.setStyleSheet("""
                font-size: 18px;
                font-weight: 600;
                color: #E0E0E0;
                margin-bottom: 12px;
            """)
            layout.addWidget(title_label)

        # Only add ONE of content OR content_layout, not both
        if content:
            layout.addWidget(content)
        elif content_layout:  # Use elif to prevent adding both
            layout.addLayout(content_layout)


class SearchableList(QWidget):
    """
    List widget with built-in search functionality.
    """

    item_selected = pyqtSignal(str)

    def __init__(self, items: list[str] | None = None):
        super().__init__()
        self.all_items = items if items is not None else []
        self._setup_ui()

    def _setup_ui(self):
        """Setup searchable list UI"""
        layout = QVBoxLayout(self)

        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search...")
        self.search_input.textChanged.connect(self._filter_items)
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: #2A2A2A;
                border: 1px solid #333;
                border-radius: 20px;
                padding: 8px 16px;
                color: #E0E0E0;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #1E88E5;
            }
        """)
        layout.addWidget(self.search_input)

        # List widget
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #1E1E1E;
                border: none;
                border-radius: 4px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
                margin: 2px;
            }
            QListWidget::item:hover {
                background-color: #2A2A2A;
            }
            QListWidget::item:selected {
                background-color: #1E88E5;
                color: white;
            }
        """)
        self.list_widget.itemClicked.connect(
            lambda item: self.item_selected.emit(item.text())
        )

        layout.addWidget(self.list_widget)

        # Populate list
        self.set_items(self.all_items)

    def set_items(self, items: list[str]):
        """Set list items"""
        self.all_items = items
        self._filter_items("")

    def _filter_items(self, text: str):
        """Filter items based on search text"""
        self.list_widget.clear()

        filtered = [item for item in self.all_items if text.lower() in item.lower()]

        for item in filtered:
            self.list_widget.addItem(item)


class ProgressIndicator(QWidget):
    """
    Modern circular progress indicator with proper Qt property.
    """

    def __init__(self, size: int = 100):
        super().__init__()
        self._size = size
        self.value = 0
        self.max_value = 100
        self._rotation = 0

        self.setFixedSize(size, size)

        # Animation - only create if we actually need it
        # Commented out to eliminate warning since rotation animation isn't critical
        # self.animation = QPropertyAnimation(self, b"rotation")
        # self.animation.setDuration(1000)
        # self.animation.setStartValue(0)
        # self.animation.setEndValue(360)
        # self.animation.setLoopCount(-1)  # Infinite
        # self.animation.start()

    # Removed rotation property and animation to eliminate PyQt6 warnings
    # The progress indicator still works without the spinning animation

    def set_value(self, value: int):
        """Set progress value (0-100)"""
        self.value = min(max(value, 0), 100)
        self.update()

    # Qt override: method name must remain camelCase for Qt event system
    def paintEvent(self, event):  # noqa: ARG002
        """Custom paint for circular progress"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background circle
        painter.setPen(QPen(QColor("#333"), 8, Qt.PenStyle.SolidLine))
        painter.drawEllipse(10, 10, self._size - 20, self._size - 20)

        # Progress arc
        painter.setPen(QPen(QColor("#1E88E5"), 8, Qt.PenStyle.SolidLine))

        # Calculate sweep angle
        sweep_angle = int(360 * (self.value / self.max_value))

        painter.drawArc(
            10,
            10,
            self._size - 20,
            self._size - 20,
            90 * 16 - self._rotation * 16,  # Start angle
            -sweep_angle * 16,  # Sweep angle (negative for clockwise)
        )

        # Center text
        painter.setPen(QPen(QColor("#E0E0E0")))
        painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{self.value}%")


class NotificationWidget(QFrame):
    """
    Toast-style notification widget.
    """

    closed = pyqtSignal()

    def __init__(
        self, message: str, notification_type: str = "info", duration: int = 3000
    ):
        super().__init__()
        self.message = message
        self.notification_type = notification_type
        self.duration = duration

        self._setup_ui()
        self._setup_animation()

        # Auto-close timer
        if duration > 0:
            QTimer.singleShot(duration, self.close_notification)

    def _setup_ui(self):
        """Setup notification UI"""

        # Style based on type
        colors = {
            "info": "#2196F3",
            "success": "#4CAF50",
            "warning": "#FF9800",
            "error": "#F44336",
        }

        color = colors.get(self.notification_type, "#2196F3")

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {color};
                border-radius: 4px;
                padding: 12px;
            }}
        """)

        layout = QHBoxLayout(self)

        # Icon
        icon_map = {"info": "i", "success": "✅", "warning": "⚠️", "error": "❌"}

        icon_label = QLabel(icon_map.get(self.notification_type, "i"))
        icon_label.setStyleSheet("font-size: 18px;")
        layout.addWidget(icon_label)

        # Message
        message_label = QLabel(self.message)
        message_label.setStyleSheet("""
            color: white;
            font-size: 14px;
            font-weight: 500;
        """)
        message_label.setWordWrap(True)
        layout.addWidget(message_label, 1)

        # Close button
        close_btn = QPushButton("✕")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
                font-size: 16px;
                font-weight: bold;
                padding: 0;
                min-width: 20px;
                max-width: 20px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.2);
                border-radius: 10px;
            }
        """)
        close_btn.clicked.connect(self.close_notification)
        layout.addWidget(close_btn)

    def _setup_animation(self):
        """Setup slide-in animation"""
        self.opacity_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)

        # Fade in
        self.fade_in = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_in.setDuration(300)
        self.fade_in.setStartValue(0.0)
        self.fade_in.setEndValue(1.0)
        self.fade_in.start()

    def close_notification(self):
        """Animate out and close"""
        # Fade out
        fade_out = QPropertyAnimation(self.opacity_effect, b"opacity")
        fade_out.setDuration(300)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.finished.connect(self.close)
        fade_out.finished.connect(self.closed.emit)
        fade_out.start()


class ChipWidget(QFrame):
    """
    Material Design chip/tag widget.
    """

    removed = pyqtSignal(str)

    def __init__(self, text: str, removable: bool = True, color: str = "#1E88E5"):
        super().__init__()
        self.text = text
        self.removable = removable
        self.color = color

        self._setup_ui()

    def _setup_ui(self):
        """Setup chip UI"""
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {self.color}20;
                border: 1px solid {self.color};
                border-radius: 12px;
                padding: 4px 8px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # Label
        label = QLabel(self.text)
        label.setStyleSheet(f"""
            color: {self.color};
            font-size: 12px;
            font-weight: 500;
        """)
        layout.addWidget(label)

        # Remove button
        if self.removable:
            remove_btn = QPushButton("✕")
            remove_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {self.color};
                    border: none;
                    font-size: 12px;
                    padding: 0;
                    min-width: 16px;
                    max-width: 16px;
                }}
                QPushButton:hover {{
                    background-color: {self.color}40;
                    border-radius: 8px;
                }}
            """)
            remove_btn.clicked.connect(lambda: self.removed.emit(self.text))
            layout.addWidget(remove_btn)


class EmailComposer(QWidget):
    """
    Modern email composition widget.
    """

    email_sent = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        """Setup email composer UI"""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("✉️ Compose Email")
        header.setStyleSheet("""
            font-size: 20px;
            font-weight: 600;
            color: #1E88E5;
            margin-bottom: 16px;
        """)
        layout.addWidget(header)

        # Recipients card
        recipients_layout = QVBoxLayout()

        # To field
        to_layout = QHBoxLayout()
        to_layout.addWidget(QLabel("To:"))
        self.to_input = QLineEdit()
        self.to_input.setPlaceholderText("recipient@example.com")
        to_layout.addWidget(self.to_input)
        recipients_layout.addLayout(to_layout)

        # CC field
        cc_layout = QHBoxLayout()
        cc_layout.addWidget(QLabel("CC:"))
        self.cc_input = QLineEdit()
        self.cc_input.setPlaceholderText("cc@example.com (optional)")
        cc_layout.addWidget(self.cc_input)
        recipients_layout.addLayout(cc_layout)

        # BCC field
        bcc_layout = QHBoxLayout()
        bcc_layout.addWidget(QLabel("BCC:"))
        self.bcc_input = QLineEdit()
        self.bcc_input.setPlaceholderText("bcc@example.com (optional)")
        bcc_layout.addWidget(self.bcc_input)
        recipients_layout.addLayout(bcc_layout)

        recipients_card = MaterialCard("Recipients", content_layout=recipients_layout)
        layout.addWidget(recipients_card)

        # Subject
        subject_layout = QHBoxLayout()
        subject_layout.addWidget(QLabel("Subject:"))
        self.subject_input = QLineEdit()
        self.subject_input.setPlaceholderText("Email subject...")
        subject_layout.addWidget(self.subject_input)
        layout.addLayout(subject_layout)

        # Email body
        body_label = QLabel("Message:")
        body_label.setStyleSheet("font-weight: 600; margin-top: 16px;")
        layout.addWidget(body_label)

        self.body_input = QTextEdit()
        self.body_input.setPlaceholderText("Type your message here...")
        self.body_input.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 12px;
                color: #E0E0E0;
                font-size: 14px;
                min-height: 200px;
            }
            QTextEdit:focus {
                border-color: #1E88E5;
            }
        """)
        layout.addWidget(self.body_input)

        # Attachments area
        attachments_label = QLabel("📎 Attachments:")
        attachments_label.setStyleSheet("margin-top: 16px;")
        layout.addWidget(attachments_label)

        self.attachments_area = QFrame()
        self.attachments_area.setStyleSheet("""
            QFrame {
                background-color: #1E1E1E;
                border: 2px dashed #333;
                border-radius: 8px;
                min-height: 60px;
            }
        """)

        attachments_layout = QHBoxLayout(self.attachments_area)
        attach_btn = AnimatedButton("Add Files", primary=False)
        attach_btn.clicked.connect(self._add_attachments)
        self.attachments_layout = attachments_layout
        self.attachments_layout.addWidget(attach_btn)
        self.attachments_layout.addStretch()

        layout.addWidget(self.attachments_area)

        # Action buttons
        actions_layout = QHBoxLayout()
        actions_layout.addStretch()

        save_draft_btn = AnimatedButton("Save Draft", primary=False)
        actions_layout.addWidget(save_draft_btn)

        send_btn = AnimatedButton("Send", primary=True)
        send_btn.clicked.connect(self._send_email)
        actions_layout.addWidget(send_btn)

        layout.addLayout(actions_layout)

        layout.addStretch()

    def _add_attachments(self):
        """Add file attachments"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Files", "", "All Files (*.*)"
        )

        if files:
            # Add chips for each file
            for file in files:
                chip = ChipWidget(file.split("/")[-1], removable=True, color="#4CAF50")
                self.attachments_layout.insertWidget(
                    self.attachments_layout.count() - 1, chip
                )

    def _send_email(self):
        """Emit email data"""
        email_data = {
            "to": self.to_input.text(),
            "cc": self.cc_input.text(),
            "bcc": self.bcc_input.text(),
            "subject": self.subject_input.text(),
            "body": self.body_input.toPlainText(),
            "attachments": [],  # Would collect from chips
        }

        self.email_sent.emit(email_data)


class ChatInterface(QWidget):
    """
    Modern chat interface for AI assistance.
    """

    message_sent = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        """Setup chat interface"""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("💬 AI Assistant")
        header.setStyleSheet("""
            font-size: 20px;
            font-weight: 600;
            color: #1E88E5;
            margin-bottom: 16px;
        """)
        layout.addWidget(header)

        # Chat area
        self.chat_area = QScrollArea()
        self.chat_area.setWidgetResizable(True)
        self.chat_area.setStyleSheet("""
            QScrollArea {
                background-color: #121212;
                border: 1px solid #333;
                border-radius: 8px;
            }
        """)

        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.addStretch()

        self.chat_area.setWidget(self.chat_widget)
        layout.addWidget(self.chat_area)

        # Input area
        input_container = QFrame()
        input_container.setStyleSheet("""
            QFrame {
                background-color: #1E1E1E;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 8px;
            }
        """)

        input_layout = QHBoxLayout(input_container)

        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your message...")
        self.message_input.setStyleSheet("""
            QLineEdit {
                background-color: transparent;
                border: none;
                color: #E0E0E0;
                font-size: 14px;
            }
        """)
        self.message_input.returnPressed.connect(self._send_message)

        send_btn = AnimatedButton("Send", primary=True)
        send_btn.clicked.connect(self._send_message)

        input_layout.addWidget(self.message_input)
        input_layout.addWidget(send_btn)

        layout.addWidget(input_container)

    def _send_message(self):
        """Send chat message"""
        message = self.message_input.text().strip()
        if message:
            # Add user message to chat
            self.add_message(message, is_user=True)

            # Clear input
            self.message_input.clear()

            # Emit signal
            self.message_sent.emit(message)

    def add_message(self, text: str, is_user: bool = False):
        """Add message to chat"""
        message_frame = QFrame()
        message_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {"#1E88E5" if is_user else "#2A2A2A"};
                border-radius: 12px;
                padding: 12px;
                margin: 4px;
            }}
        """)

        message_layout = QVBoxLayout(message_frame)

        # Sender label
        sender = QLabel("You" if is_user else "Assistant")
        sender.setStyleSheet("""
            font-size: 12px;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.7);
        """)
        message_layout.addWidget(sender)

        # Message text
        message_label = QLabel(text)
        message_label.setWordWrap(True)
        message_label.setStyleSheet("""
            color: white;
            font-size: 14px;
        """)
        message_layout.addWidget(message_label)

        # Add to chat
        self.chat_layout.insertWidget(
            self.chat_layout.count() - 1,
            message_frame,
            0,
            Qt.AlignmentFlag.AlignRight if is_user else Qt.AlignmentFlag.AlignLeft,
        )

        # Scroll to bottom
        def scroll_to_bottom():
            vbar = self.chat_area.verticalScrollBar()
            if vbar is not None:
                vbar.setValue(vbar.maximum())

        QTimer.singleShot(100, scroll_to_bottom)
