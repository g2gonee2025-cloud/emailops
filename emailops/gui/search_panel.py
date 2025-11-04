"""
Search Panel - GUI for advanced email search
"""

import logging

from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import constants
from .base_panel import BasePanel
from .components import AnimatedButton

logger = logging.getLogger(__name__)


class SearchPanel(BasePanel):
    """
    Advanced search panel with filters and real-time results.
    """

    search_requested = pyqtSignal(dict)

    def __init__(self):
        super().__init__(constants.SEARCH_PANEL_TITLE)
        self._init_ui()

    def _init_ui(self):
        layout = self.main_layout

        # Search input with animation
        search_container = QWidget()
        search_container.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
                border-radius: 24px;
                padding: 4px;
            }
        """)

        search_layout = QHBoxLayout(search_container)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Search emails, conversations, attachments..."
        )
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: transparent;
                border: none;
                padding: 12px 16px;
                font-size: 14px;
                color: #E0E0E0;
            }
        """)

        search_btn = AnimatedButton("Search", primary=True)
        search_btn.clicked.connect(self._on_search)

        search_layout.addWidget(self.search_input)
        search_layout.addWidget(search_btn)

        layout.addWidget(search_container)

        # Advanced filters (collapsible)
        self.filters_container = QGroupBox("Advanced Filters")
        self.filters_container.setCheckable(True)
        self.filters_container.setChecked(False)
        self.filters_container.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                color: #B0B0B0;
                border: 1px solid #333;
                border-radius: 8px;
                margin-top: 16px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)

        filters_layout = QVBoxLayout()

        # Date range
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("Date:"))
        self.date_from = QLineEdit()
        self.date_from.setPlaceholderText("From (YYYY-MM-DD)")
        self.date_to = QLineEdit()
        self.date_to.setPlaceholderText("To (YYYY-MM-DD)")
        date_layout.addWidget(self.date_from)
        date_layout.addWidget(self.date_to)
        filters_layout.addLayout(date_layout)

        # Email filters
        email_layout = QHBoxLayout()
        email_layout.addWidget(QLabel("From:"))
        self.from_filter = QLineEdit()
        self.from_filter.setPlaceholderText("sender@example.com")
        email_layout.addWidget(self.from_filter)
        filters_layout.addLayout(email_layout)

        # Attachment filter
        self.has_attachments = QCheckBox("Has attachments")
        filters_layout.addWidget(self.has_attachments)

        self.filters_container.setLayout(filters_layout)
        layout.addWidget(self.filters_container)

        # Results area
        results_label = QLabel("Search Results")
        results_label.setStyleSheet("font-weight: 600; margin-top: 16px;")
        layout.addWidget(results_label)

        self.results_tree = QTreeWidget()
        self.results_tree.setHeaderLabels(["Score", "Subject", "Date", "From"])
        self.results_tree.setAlternatingRowColors(True)
        self.results_tree.setStyleSheet("""
            QTreeWidget {
                background-color: #1E1E1E;
                border: 1px solid #333;
                border-radius: 8px;
            }
            QTreeWidget::item:selected {
                background-color: #1E88E5;
            }
            QTreeWidget::item:hover {
                background-color: #2A2A2A;
            }
        """)
        layout.addWidget(self.results_tree)

        # Search status
        self.status_label = QLabel("Ready to search")
        self.status_label.setStyleSheet("color: #B0B0B0; font-style: italic;")
        layout.addWidget(self.status_label)

    def _on_search(self):
        """Emit search request with parameters"""
        params = {
            "query": self.search_input.text(),
            "filters": {
                "date_from": self.date_from.text()
                if self.filters_container.isChecked()
                else None,
                "date_to": self.date_to.text()
                if self.filters_container.isChecked()
                else None,
                "from_email": self.from_filter.text()
                if self.filters_container.isChecked()
                else None,
                "has_attachments": self.has_attachments.isChecked()
                if self.filters_container.isChecked()
                else None,
            },
        }
        self.search_requested.emit(params)
        self.status_label.setText("Searching...")

    @pyqtSlot(list)
    def display_results(self, results: list[dict]):
        """Display search results"""
        self.results_tree.clear()

        for result in results:
            item = QTreeWidgetItem(
                [
                    f"{result.get('score', 0):.2f}",
                    result.get("subject", "No subject"),
                    result.get("date", ""),
                    result.get("from", ""),
                ]
            )
            self.results_tree.addTopLevelItem(item)

        self.status_label.setText(f"Found {len(results)} results")
