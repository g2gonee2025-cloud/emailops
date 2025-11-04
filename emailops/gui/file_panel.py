"""
File Panel - GUI for file import/export operations
"""

import logging
from typing import Any

from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTextEdit,
    QVBoxLayout,
)

from . import constants
from .base_panel import BasePanel
from .components import AnimatedButton, MaterialCard

logger = logging.getLogger(__name__)


class FilePanel(BasePanel):
    """
    Panel for managing file import and export operations.
    """

    # Signals
    file_operation_requested = pyqtSignal(dict)

    def __init__(self):
        super().__init__(constants.FILE_PANEL_TITLE)
        self._init_ui()

    def _init_ui(self):
        """Initialize the file panel UI"""
        layout = self.main_layout

        # Operation selection
        op_layout = QHBoxLayout()
        op_layout.addWidget(QLabel("Operation:"))
        self.op_combo = QComboBox()
        self.op_combo.addItems(["Export", "Import"])
        self.op_combo.currentTextChanged.connect(self._on_operation_changed)
        op_layout.addWidget(self.op_combo)
        op_layout.addStretch()
        layout.addLayout(op_layout)

        # File path card
        path_layout = QGridLayout()
        path_layout.addWidget(QLabel("File Path:"), 0, 0)
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Select a file or directory...")
        path_layout.addWidget(self.path_input, 0, 1)
        self.browse_btn = AnimatedButton("Browse", primary=False)
        self.browse_btn.clicked.connect(self._browse_path)
        path_layout.addWidget(self.browse_btn, 0, 2)
        path_card = MaterialCard("File Path", content_layout=path_layout)
        layout.addWidget(path_card)

        # Data type card (for export)
        export_layout = QGridLayout()
        export_layout.addWidget(QLabel("Data Type:"), 0, 0)
        self.export_type_combo = QComboBox()
        self.export_type_combo.addItems(
            ["Search Results (CSV)", "Analysis (JSON)", "Chat History (Text)"]
        )
        export_layout.addWidget(self.export_type_combo, 0, 1)
        self.export_options_card = MaterialCard("Export Options", content_layout=export_layout)
        layout.addWidget(self.export_options_card)

        # Data type card (for import)
        import_layout = QGridLayout()
        import_layout.addWidget(QLabel("Data Type:"), 0, 0)
        self.import_type_combo = QComboBox()
        self.import_type_combo.addItems(
            ["Configuration (JSON)", "Conversation IDs (Text)"]
        )
        import_layout.addWidget(self.import_type_combo, 0, 1)
        self.import_options_card = MaterialCard("Import Options", content_layout=import_layout)
        self.import_options_card.setVisible(False)
        layout.addWidget(self.import_options_card)

        # Data preview area
        preview_layout = QVBoxLayout()
        preview_layout.addWidget(QLabel("Data Preview:"))
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText(
            "Data to be exported or imported will be shown here..."
        )
        preview_layout.addWidget(self.preview_text)
        preview_card = MaterialCard("Preview", content_layout=preview_layout)
        layout.addWidget(preview_card)

        # Action button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.execute_btn = AnimatedButton("Execute", primary=True)
        self.execute_btn.clicked.connect(self._on_execute)
        button_layout.addWidget(self.execute_btn)
        layout.addLayout(button_layout)

        layout.addStretch()

    def _on_operation_changed(self, operation: str):
        """Handle operation change (Import/Export)"""
        is_export = operation == "Export"
        self.export_options_card.setVisible(is_export)
        self.import_options_card.setVisible(not is_export)
        self.execute_btn.setText("Export" if is_export else "Import")

    def _browse_path(self):
        """Browse for file or directory based on operation"""
        operation = self.op_combo.currentText()
        if operation == "Export":
            filename, _ = QFileDialog.getSaveFileName(self, "Save File")
            if filename:
                self.path_input.setText(filename)
        else:  # Import
            filename, _ = QFileDialog.getOpenFileName(self, "Open File")
            if filename:
                self.path_input.setText(filename)

    def _on_execute(self):
        """Execute the selected file operation"""
        operation = self.op_combo.currentText()
        path = self.path_input.text().strip()

        if not path:
            QMessageBox.warning(self, "Missing Path", "Please specify a file path.")
            return

        params = {"path": path}
        if operation == "Export":
            params["operation"] = "export"
            params["data_type"] = self.export_type_combo.currentText()
            params["data"] = self.preview_text.toPlainText()  # Simplified for example
        else:  # Import
            params["operation"] = "import"
            params["data_type"] = self.import_type_combo.currentText()

        self.file_operation_requested.emit(params)

    @pyqtSlot(dict)
    def on_file_operation_completed(self, result: dict):
        """Handle file operation completion"""
        if result.get("success"):
            QMessageBox.information(self, "Success", "File operation completed successfully.")
        elif result.get("data"):
            self.preview_text.setText(str(result["data"]))
            QMessageBox.information(self, "Imported Data", "Data loaded into preview.")
        else:
            QMessageBox.critical(
                self,
                "Error",
                f"File operation failed: {result.get('error', 'Unknown error')}",
            )

    def set_export_data(self, data: Any, data_type: str):
        """Set data to be exported"""
        self.op_combo.setCurrentText("Export")
        self.export_type_combo.setCurrentText(data_type)

        if isinstance(data, (dict, list)):
            import json
            self.preview_text.setText(json.dumps(data, indent=2))
        else:
            self.preview_text.setText(str(data))
