"""
Analysis Panel - GUI for email conversation analysis
"""

import logging
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
)

from . import constants
from .base_panel import BasePanel
from .components import AnimatedButton, MaterialCard, ProgressIndicator

logger = logging.getLogger(__name__)


class AnalysisPanel(BasePanel):
    """
    Panel for email conversation analysis and summarization.
    """

    # Signals
    analysis_requested = pyqtSignal(dict)

    def __init__(self):
        super().__init__(constants.ANALYSIS_PANEL_TITLE)
        self._init_ui()

    def _init_ui(self):
        """Initialize the analysis panel UI"""
        layout = self.main_layout

        # Analysis mode card
        mode_layout = QVBoxLayout()
        self.single_radio = QRadioButton("Single Conversation Analysis")
        self.single_radio.setChecked(True)
        self.single_radio.toggled.connect(self._on_mode_changed)
        self.batch_radio = QRadioButton("Batch Analysis")
        self.batch_radio.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.single_radio)
        mode_layout.addWidget(self.batch_radio)
        mode_card = MaterialCard("Analysis Mode", content_layout=mode_layout)
        layout.addWidget(mode_card)

        # Single conversation selector
        single_layout = QVBoxLayout()
        conv_select_layout = QHBoxLayout()
        self.conv_input = QLineEdit()
        self.conv_input.setPlaceholderText("Enter conversation ID...")
        conv_select_layout.addWidget(self.conv_input)
        self.browse_btn = AnimatedButton("Browse", primary=False)
        self.browse_btn.clicked.connect(self._browse_conversation)
        conv_select_layout.addWidget(self.browse_btn)
        single_layout.addLayout(conv_select_layout)
        self.single_conv_card = MaterialCard("Select Conversation", content_layout=single_layout)
        layout.addWidget(self.single_conv_card)

        # Batch conversation selector
        batch_layout = QVBoxLayout()
        self.batch_search = QLineEdit()
        self.batch_search.setPlaceholderText("Search conversations...")
        self.batch_search.textChanged.connect(self._filter_batch_list)
        batch_layout.addWidget(self.batch_search)
        self.batch_list = QListWidget()
        self.batch_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.batch_list.setStyleSheet("""
            QListWidget {
                background-color: #1E1E1E;
                border: 1px solid #333;
                border-radius: 4px;
                min-height: 150px;
            }
            QListWidget::item:selected {
                background-color: #1E88E5;
            }
        """)
        batch_layout.addWidget(self.batch_list)
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText(str(constants.ANALYSIS_OUTPUT_DIR))
        output_layout.addWidget(self.output_dir_input)
        self.output_browse_btn = AnimatedButton("Browse", primary=False)
        self.output_browse_btn.clicked.connect(self._browse_output_dir)
        output_layout.addWidget(self.output_browse_btn)
        batch_layout.addLayout(output_layout)
        self.batch_conv_card = MaterialCard(
            "Select Conversations for Batch",
            content_layout=batch_layout,
        )
        self.batch_conv_card.setVisible(False)
        layout.addWidget(self.batch_conv_card)

        # Analysis options card
        options_layout = QVBoxLayout()
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self.temp_spin = QLineEdit(constants.ANALYSIS_DEFAULT_TEMP)
        self.temp_spin.setMaximumWidth(100)
        temp_layout.addWidget(self.temp_spin)
        temp_layout.addWidget(QLabel("(0.0 - 2.0)"))
        temp_layout.addStretch()
        options_layout.addLayout(temp_layout)
        self.merge_manifest_check = QCheckBox("Merge manifest data")
        self.merge_manifest_check.setChecked(True)
        options_layout.addWidget(self.merge_manifest_check)
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Output Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["JSON", "Markdown", "Both"])
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        options_layout.addLayout(format_layout)
        options_card = MaterialCard("Analysis Options", content_layout=options_layout)
        layout.addWidget(options_card)

        # Progress indicator
        self.progress = ProgressIndicator()
        self.progress.setVisible(False)
        layout.addWidget(self.progress, alignment=Qt.AlignmentFlag.AlignCenter)

        # Results area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Analysis results will appear here...")
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 12px;
                min-height: 200px;
            }
        """)
        layout.addWidget(self.results_text)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.export_btn = AnimatedButton("Export Results", primary=False)
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)

        self.analyze_btn = AnimatedButton("Analyze", primary=True)
        self.analyze_btn.clicked.connect(self._on_analyze)
        button_layout.addWidget(self.analyze_btn)

        layout.addLayout(button_layout)
        layout.addStretch()

    def _on_mode_changed(self):
        """Handle analysis mode change"""
        is_single = self.single_radio.isChecked()
        self.single_conv_card.setVisible(is_single)
        self.batch_conv_card.setVisible(not is_single)

    def _browse_conversation(self):
        """Browse for conversation directory"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Conversation Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if folder:
            conv_id = Path(folder).name
            self.conv_input.setText(conv_id)

    def _browse_output_dir(self):
        """Browse for output directory"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            str(constants.ANALYSIS_OUTPUT_DIR),
            QFileDialog.Option.ShowDirsOnly
        )
        if folder:
            self.output_dir_input.setText(folder)

    def _filter_batch_list(self, text: str):
        """Filter batch conversation list"""
        for i in range(self.batch_list.count()):
            item = self.batch_list.item(i)
            if item:
                item.setHidden(text.lower() not in item.text().lower())

    def _on_analyze(self):
        """Start analysis - VALIDATION ENHANCED"""
        try:
            temperature = float(self.temp_spin.text())
            if temperature < 0 or temperature > 2:
                raise ValueError("Temperature must be between 0 and 2")
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Temperature", str(e))
            return

        format_map = {"JSON": "json", "Markdown": "markdown", "Both": "both"}
        output_format = format_map[self.format_combo.currentText()]

        params: dict[str, Any] = {
            "temperature": temperature,
            "merge_manifest": self.merge_manifest_check.isChecked(),
            "output_format": output_format
        }

        if self.single_radio.isChecked():
            # Single conversation - ENHANCED VALIDATION
            conv_id = self.conv_input.text().strip()
            if not conv_id:
                QMessageBox.warning(
                    self,
                    "No Conversation",
                    "Please enter a conversation ID."
                )
                return

            # Security: Check for path traversal
            if ".." in conv_id or "/" in conv_id or "\\" in conv_id:
                QMessageBox.critical(
                    self,
                    "Invalid Input",
                    (
                        "Conversation ID contains invalid characters.\n"
                        "Path traversal attempts are blocked."
                    ),
                )
                return

            # Length validation
            if len(conv_id) > 255:
                QMessageBox.warning(
                    self,
                    "Invalid Input",
                    "Conversation ID is too long (max 255 characters)."
                )
                return

            # Format validation (alphanumeric, hyphens, underscores only)
            import re
            if not re.match(r'^[a-zA-Z0-9_-]+$', conv_id):
                QMessageBox.warning(
                    self,
                    "Invalid Format",
                    "Conversation ID can only contain letters, numbers, hyphens, and underscores."
                )
                return

            params["conv_id"] = conv_id
        else:
            # Batch analysis
            selected_items = self.batch_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(
                    self,
                    "No Selection",
                    "Please select conversations to analyze."
                )
                return

            params["conv_ids"] = [item.text() for item in selected_items]

            output_dir = self.output_dir_input.text() or str(constants.ANALYSIS_OUTPUT_DIR)
            params["output_dir"] = output_dir

        # Show progress
        self.progress.setVisible(True)
        self.progress.set_value(0)
        self.analyze_btn.setEnabled(False)
        self.results_text.clear()

        # Emit signal
        self.analysis_requested.emit(params)

    def _export_results(self):
        """Export analysis results"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Analysis Results",
            "analysis_results.txt",
            "Text Files (*.txt);;All Files (*.*)"
        )
        if filename:
            try:
                with Path(filename).open('w', encoding='utf-8') as f:
                    f.write(self.results_text.toPlainText())
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Results exported to {filename}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to export results: {e}"
                )

    @pyqtSlot(dict)
    def on_analysis_completed(self, results: dict):
        """Handle analysis completion - TYPE VALIDATION FIXED"""
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

        # Validate results type BEFORE accessing keys
        if not isinstance(results, dict):
            self.results_text.append("❌ Invalid results format received!")
            logger.error(f"Invalid results type: {type(results)}")
            return

        if results.get("analysis"):
            analysis = results["analysis"]

            # Show brief summary
            if analysis.get("brief_summary"):
                self.results_text.append("📋 BRIEF SUMMARY:")
                self.results_text.append(analysis["brief_summary"])
                self.results_text.append("")

            # Show detailed summary
            if analysis.get("detailed_summary"):
                self.results_text.append("📖 DETAILED SUMMARY:")
                detailed = analysis["detailed_summary"]
                if isinstance(detailed, dict):
                    for key, value in detailed.items():
                        self.results_text.append(f"  • {key}: {value}")
                else:
                    self.results_text.append(str(detailed))
                self.results_text.append("")

            # Show next actions
            if analysis.get("next_actions"):
                self.results_text.append("✅ NEXT ACTIONS:")
                for action in analysis["next_actions"]:
                    self.results_text.append(f"  □ {action}")
                self.results_text.append("")

            # Show files created
            if results.get("files_created"):
                self.results_text.append("💾 FILES CREATED:")
                for file in results["files_created"]:
                    self.results_text.append(f"  • {file}")

        elif results.get("completed"):
            # Batch results
            self.results_text.append("✅ Batch analysis completed!")
            self.results_text.append(f"Total analyzed: {results.get('completed', 0)}")
            self.results_text.append(f"Failed: {results.get('failed', 0)}")

            if results.get("analyses"):
                self.results_text.append("\nSummaries:")
                for item in results["analyses"][:5]:  # Show first 5
                    self.results_text.append(f"\n📧 {item['conv_id']}:")
                    self.results_text.append(f"  {item.get('summary', 'No summary')[:200]}...")

        else:
            self.results_text.append("❌ Analysis failed!")
            if results.get("error"):
                self.results_text.append(f"Error: {results['error']}")

    def load_conversations(self, conv_ids: list[str]):
        """Load conversation list for batch mode"""
        self.batch_list.clear()
        self.batch_list.addItems(conv_ids)
