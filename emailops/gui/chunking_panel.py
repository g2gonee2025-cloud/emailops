"""
Chunking Panel - GUI for text chunking operations
"""

import logging
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QTextEdit,
    QVBoxLayout,
)

from . import constants
from .base_panel import BasePanel
from .components import AnimatedButton, MaterialCard, ProgressIndicator

logger = logging.getLogger(__name__)


class ChunkingPanel(BasePanel):
    """
    Panel for managing text chunking operations.
    """

    # Signals
    chunk_requested = pyqtSignal(dict)

    def __init__(self):
        super().__init__(constants.CHUNKING_PANEL_TITLE)
        self._init_ui()

    def _init_ui(self):
        """Initialize the chunking panel UI"""
        layout = self.main_layout

        # Operation selector card
        operation_layout = QVBoxLayout()

        # Operation type selector
        self.operation_combo = QComboBox()
        self.operation_combo.addItems([
            "Incremental Chunk (New/Changed Only)",
            "Force Re-chunk All",
            "Surgical Re-chunk (Selected)"
        ])
        self.operation_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border-radius: 4px;
                background-color: #2A2A2A;
                min-height: 40px;
            }
        """)
        self.operation_combo.currentIndexChanged.connect(self._on_operation_changed)
        operation_layout.addWidget(self.operation_combo)

        # Description label
        self.description_label = QLabel(constants.CHUNKING_DEFAULT_DESC)
        self.description_label.setStyleSheet("""
            color: #999;
            font-style: italic;
            padding: 8px 0;
        """)
        operation_layout.addWidget(self.description_label)

        operation_card = MaterialCard("Chunking Operation", content_layout=operation_layout)
        layout.addWidget(operation_card)

        # Conversation selector (for surgical mode)
        conv_layout = QVBoxLayout()

        # Search input
        self.conv_search = QLineEdit()
        self.conv_search.setPlaceholderText("Search conversations...")
        self.conv_search.textChanged.connect(self._filter_conversations)
        conv_layout.addWidget(self.conv_search)

        # Conversation list
        self.conv_list = QListWidget()
        self.conv_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.conv_list.setStyleSheet("""
            QListWidget {
                background-color: #1E1E1E;
                border: 1px solid #333;
                border-radius: 4px;
                min-height: 200px;
            }
            QListWidget::item:selected {
                background-color: #1E88E5;
            }
        """)
        conv_layout.addWidget(self.conv_list)

        self.conv_selector_card = MaterialCard("Select Conversations", content_layout=conv_layout)
        self.conv_selector_card.setVisible(False)
        layout.addWidget(self.conv_selector_card)

        # Statistics card
        stats_layout = QHBoxLayout()

        self.total_convs_label = QLabel("Total Conversations: 0")
        self.total_chunks_label = QLabel("Total Chunks: 0")
        self.avg_chunks_label = QLabel("Avg Chunks/Conv: 0")

        for label in [self.total_convs_label, self.total_chunks_label, self.avg_chunks_label]:
            label.setStyleSheet("font-size: 14px; padding: 8px;")
            stats_layout.addWidget(label)

        stats_card = MaterialCard("Chunking Statistics", content_layout=stats_layout)
        layout.addWidget(stats_card)

        # Progress indicator
        self.progress = ProgressIndicator()
        self.progress.setVisible(False)
        layout.addWidget(self.progress, alignment=Qt.AlignmentFlag.AlignCenter)

        # Results area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Chunking results will appear here...")
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 12px;
                min-height: 150px;
            }
        """)
        layout.addWidget(self.results_text)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.clear_chunks_btn = AnimatedButton("Clear All Chunks", primary=False)
        self.clear_chunks_btn.clicked.connect(self._on_clear_chunks)
        button_layout.addWidget(self.clear_chunks_btn)

        self.execute_btn = AnimatedButton("Execute Chunking", primary=True)
        self.execute_btn.clicked.connect(self._on_execute)
        button_layout.addWidget(self.execute_btn)

        layout.addLayout(button_layout)
        layout.addStretch()

    def _on_operation_changed(self, index: int):
        """Handle operation type change"""
        descriptions = [
            constants.CHUNKING_DEFAULT_DESC,
            "Re-process all conversations (may take time)",
            "Re-process specific selected conversations"
        ]
        self.description_label.setText(descriptions[index])

        # Show/hide conversation selector
        self.conv_selector_card.setVisible(index == 2)

    def _filter_conversations(self, text: str):
        """Filter conversation list based on search text"""
        for i in range(self.conv_list.count()):
            item = self.conv_list.item(i)
            if item:
                item.setHidden(text.lower() not in item.text().lower())

    def _on_execute(self):
        """Execute chunking operation - VALIDATION ENHANCED"""
        operation_map = {
            0: "incremental",
            1: "force_all",
            2: "surgical"
        }

        operation = operation_map[self.operation_combo.currentIndex()]

        params: dict[str, Any] = {"operation": operation}

        # For surgical mode, validate selected conversations
        if operation == "surgical":
            selected_items = self.conv_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(
                    self,
                    "No Selection",
                    "Please select conversations to re-chunk."
                )
                return

            # Validate conv_ids format
            conv_ids = [item.text() for item in selected_items]
            import re
            invalid_ids = [cid for cid in conv_ids if not re.match(r'^[a-zA-Z0-9_-]+$', cid)]
            if invalid_ids:
                invalid_sample = ", ".join(invalid_ids[:3])
                QMessageBox.critical(
                    self,
                    "Invalid Input",
                    (
                        f"Invalid conversation ID format: {invalid_sample}\n"
                        "IDs can only contain letters, numbers, hyphens, and underscores."
                    ),
                )
                return

            params["conv_ids"] = conv_ids

        # Show progress
        self.progress.setVisible(True)
        self.progress.set_value(0)
        self.execute_btn.setEnabled(False)
        self.results_text.clear()

        # Emit signal
        self.chunk_requested.emit(params)

    def _on_clear_chunks(self):
        """Clear all chunks after confirmation"""
        reply = QMessageBox.question(
            self,
            "Clear All Chunks",
            (
                "Are you sure you want to clear all chunked data?\n"
                "This will require re-chunking before search works."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # This would call the chunking service to clear chunks
            self.results_text.append("Clearing all chunks...")
            # Emit signal to clear chunks

    @pyqtSlot(dict)
    def on_chunks_updated(self, results: dict):
        """Handle chunking completion - ERROR HANDLING ENHANCED"""
        self.progress.setVisible(False)
        self.execute_btn.setEnabled(True)

        # Validate results structure
        if not isinstance(results, dict):
            self.results_text.append("❌ Invalid results format received!")
            logger.error(f"Invalid results type: {type(results)}")
            return

        # Update results
        if results.get("success"):
            self.results_text.append("✅ Chunking completed successfully!")

            # Safely extract statistics with defaults
            total_convs = int(results.get('total_conversations', 0))
            total_chunks = int(results.get('total_chunks', 0))

            self.results_text.append(f"Total conversations: {total_convs}")
            self.results_text.append(f"Total chunks created: {total_chunks}")

            # Update statistics safely
            try:
                self.update_statistics(results)
            except Exception as e:
                logger.error(f"Failed to update statistics: {e}")
                self.results_text.append("Warning: Could not update statistics display")

            # Show details
            conv_details = results.get("conversation_details", [])
            if conv_details and isinstance(conv_details, list):
                self.results_text.append("\nConversation Details:")
                for detail in conv_details[:10]:  # Show first 10
                    if isinstance(detail, dict):
                        conv_id = detail.get('conv_id', 'unknown')
                        chunks = detail.get('chunks', 0)
                        self.results_text.append(f"  - {conv_id}: {chunks} chunks")
                if len(conv_details) > 10:
                    self.results_text.append(f"  ... and {len(conv_details) - 10} more")
        else:
            self.results_text.append("❌ Chunking failed!")
            error_msg = results.get("error_details") or results.get("error") or "Unknown error"
            self.results_text.append(f"Error: {error_msg}")

            # Log for debugging
            logger.error(f"Chunking operation failed: {error_msg}")

    def update_statistics(self, stats: dict):
        """Update statistics display - NULL SAFETY ADDED"""
        # Validate input
        if not isinstance(stats, dict):
            logger.warning(f"Invalid stats type: {type(stats)}")
            return

        # Safe extraction with type validation
        try:
            total_convs = int(stats.get("total_conversations", 0))
            total_chunks = int(stats.get("total_chunks", 0))
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid statistic values: {e}")
            total_convs = 0
            total_chunks = 0

        avg_chunks = total_chunks / total_convs if total_convs > 0 else 0.0

        self.total_convs_label.setText(f"Total Conversations: {total_convs}")
        self.total_chunks_label.setText(f"Total Chunks: {total_chunks}")
        self.avg_chunks_label.setText(f"Avg Chunks/Conv: {avg_chunks:.1f}")

    def load_conversations(self, conv_ids: list[str]):
        """Load conversation list for surgical mode"""
        self.conv_list.clear()
        self.conv_list.addItems(conv_ids)
