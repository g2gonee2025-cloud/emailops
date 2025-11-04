"""
Base Panel Module
"""

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class BasePanel(QWidget):
    """
    A base class for all panels that provides a consistent layout and header.
    """

    def __init__(self, title: str):
        """
        Initializes the BasePanel.
        Args:
            title: The title to be displayed in the header.
        """
        super().__init__()
        self.main_layout = QVBoxLayout(self)
        self.header = QLabel(title)
        self.header.setStyleSheet("""
            font-size: 20px;
            font-weight: 600;
            color: #1E88E5;
            padding: 16px 0;
        """)
        self.main_layout.addWidget(self.header)
