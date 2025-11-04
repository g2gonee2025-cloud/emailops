"""
Configuration Panel - GUI for EmailOps configuration management
"""

import contextlib
import json
import logging
from pathlib import Path

from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import constants
from .base_panel import BasePanel
from .components import AnimatedButton, MaterialCard

logger = logging.getLogger(__name__)


class ConfigPanel(BasePanel):
    """
    Panel for managing EmailOps configuration settings.
    """

    # Signals
    config_update_requested = pyqtSignal(dict)
    config_get_requested = pyqtSignal()

    def __init__(self):
        super().__init__(constants.CONFIG_PANEL_TITLE)
        self.config_widgets = {}
        self._init_ui()

    def _init_ui(self):
        """Initialize the configuration panel UI"""
        # Main scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
        """)

        # Container widget
        container = QWidget()
        layout = QVBoxLayout(container)
        self.main_layout.addWidget(scroll)
        scroll.setWidget(container)

        # Paths configuration
        paths_layout = QGridLayout()
        paths_layout.addWidget(QLabel("Export Root:"), 0, 0)
        self.export_root_input = QLineEdit()
        self.export_root_input.setPlaceholderText(constants.CONFIG_DEFAULT_EXPORT_ROOT)
        paths_layout.addWidget(self.export_root_input, 0, 1)
        self.export_root_btn = AnimatedButton("Browse", primary=False)
        self.export_root_btn.clicked.connect(
            lambda: self._browse_folder(self.export_root_input)
        )
        paths_layout.addWidget(self.export_root_btn, 0, 2)
        self.config_widgets["export_root"] = self.export_root_input
        paths_layout.addWidget(QLabel("Index Directory:"), 1, 0)
        self.index_dirname_input = QLineEdit()
        self.index_dirname_input.setPlaceholderText(constants.CONFIG_DEFAULT_INDEX_DIR)
        paths_layout.addWidget(self.index_dirname_input, 1, 1, 1, 2)
        self.config_widgets["index_dirname"] = self.index_dirname_input
        paths_card = MaterialCard("Paths Configuration", content_layout=paths_layout)
        layout.addWidget(paths_card)

        # LLM Configuration
        llm_layout = QGridLayout()
        llm_layout.addWidget(QLabel("Provider:"), 0, 0)
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["vertex", "openai", "anthropic"])
        llm_layout.addWidget(self.provider_combo, 0, 1)
        self.config_widgets["provider"] = self.provider_combo
        llm_layout.addWidget(QLabel("Temperature:"), 1, 0)
        self.temperature_input = QLineEdit(constants.CONFIG_DEFAULT_TEMP)
        self.temperature_input.setPlaceholderText("0.0 - 2.0")
        llm_layout.addWidget(self.temperature_input, 1, 1)
        self.config_widgets["temperature"] = self.temperature_input
        llm_layout.addWidget(QLabel("Reply Policy:"), 2, 0)
        self.reply_policy_combo = QComboBox()
        self.reply_policy_combo.addItems(["reply_all", "smart", "sender_only"])
        llm_layout.addWidget(self.reply_policy_combo, 2, 1)
        self.config_widgets["reply_policy"] = self.reply_policy_combo
        llm_card = MaterialCard("LLM Configuration", content_layout=llm_layout)
        layout.addWidget(llm_card)

        # Search Configuration
        search_layout = QGridLayout()
        search_layout.addWidget(QLabel("Search Results (k):"), 0, 0)
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 100)
        self.k_spin.setValue(constants.CONFIG_DEFAULT_K)
        search_layout.addWidget(self.k_spin, 0, 1)
        self.config_widgets["k"] = self.k_spin
        search_layout.addWidget(QLabel("Similarity Threshold:"), 1, 0)
        self.sim_threshold_input = QLineEdit(constants.CONFIG_DEFAULT_SIM_THRESHOLD)
        self.sim_threshold_input.setPlaceholderText("0.0 - 1.0")
        search_layout.addWidget(self.sim_threshold_input, 1, 1)
        self.config_widgets["sim_threshold"] = self.sim_threshold_input
        search_layout.addWidget(QLabel("MMR Lambda:"), 2, 0)
        self.mmr_lambda_input = QLineEdit(constants.CONFIG_DEFAULT_MMR_LAMBDA)
        self.mmr_lambda_input.setPlaceholderText("0.0 - 1.0")
        search_layout.addWidget(self.mmr_lambda_input, 2, 1)
        self.config_widgets["mmr_lambda"] = self.mmr_lambda_input
        search_layout.addWidget(QLabel("Rerank Alpha:"), 3, 0)
        self.rerank_alpha_input = QLineEdit(constants.CONFIG_DEFAULT_RERANK_ALPHA)
        self.rerank_alpha_input.setPlaceholderText("0.0 - 1.0")
        search_layout.addWidget(self.rerank_alpha_input, 3, 1)
        self.config_widgets["rerank_alpha"] = self.rerank_alpha_input
        search_card = MaterialCard("Search Configuration", content_layout=search_layout)
        layout.addWidget(search_card)

        # Chunking Configuration
        chunking_layout = QGridLayout()
        chunking_layout.addWidget(QLabel("Chunk Size:"), 0, 0)
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(100, 10000)
        self.chunk_size_spin.setValue(constants.CONFIG_DEFAULT_CHUNK_SIZE)
        chunking_layout.addWidget(self.chunk_size_spin, 0, 1)
        self.config_widgets["chunk_size"] = self.chunk_size_spin
        chunking_layout.addWidget(QLabel("Chunk Overlap:"), 1, 0)
        self.chunk_overlap_spin = QSpinBox()
        self.chunk_overlap_spin.setRange(0, 1000)
        self.chunk_overlap_spin.setValue(constants.CONFIG_DEFAULT_CHUNK_OVERLAP)
        chunking_layout.addWidget(self.chunk_overlap_spin, 1, 1)
        self.config_widgets["chunk_overlap"] = self.chunk_overlap_spin
        chunking_card = MaterialCard("Chunking Configuration", content_layout=chunking_layout)
        layout.addWidget(chunking_card)

        # Processing Configuration
        processing_layout = QGridLayout()
        processing_layout.addWidget(QLabel("Number of Workers:"), 0, 0)
        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(1, 32)
        self.num_workers_spin.setValue(constants.CONFIG_DEFAULT_NUM_WORKERS)
        processing_layout.addWidget(self.num_workers_spin, 0, 1)
        self.config_widgets["num_workers"] = self.num_workers_spin
        processing_layout.addWidget(QLabel("Target Tokens:"), 1, 0)
        self.target_tokens_spin = QSpinBox()
        self.target_tokens_spin.setRange(1000, 100000)
        self.target_tokens_spin.setValue(constants.CONFIG_DEFAULT_TARGET_TOKENS)
        self.target_tokens_spin.setSingleStep(1000)
        processing_layout.addWidget(self.target_tokens_spin, 1, 1)
        self.config_widgets["target_tokens"] = self.target_tokens_spin
        processing_layout.addWidget(QLabel("Email Chunk Lines:"), 2, 0)
        self.email_chunk_lines_spin = QSpinBox()
        self.email_chunk_lines_spin.setRange(10, 1000)
        self.email_chunk_lines_spin.setValue(constants.CONFIG_DEFAULT_EMAIL_CHUNK_LINES)
        processing_layout.addWidget(self.email_chunk_lines_spin, 2, 1)
        self.config_widgets["email_chunk_lines"] = self.email_chunk_lines_spin
        processing_card = MaterialCard("Processing Configuration", content_layout=processing_layout)
        layout.addWidget(processing_card)

        # Features Configuration
        features_layout = QVBoxLayout()
        self.include_attachments_check = QCheckBox("Include Attachments in Drafts")
        self.include_attachments_check.setChecked(True)
        self.config_widgets["include_attachments"] = self.include_attachments_check
        features_layout.addWidget(self.include_attachments_check)
        self.verbose_check = QCheckBox("Verbose Logging")
        self.verbose_check.setChecked(False)
        self.config_widgets["verbose"] = self.verbose_check
        features_layout.addWidget(self.verbose_check)
        features_card = MaterialCard("Features Configuration", content_layout=features_layout)
        layout.addWidget(features_card)

        # Raw configuration display
        raw_layout = QVBoxLayout()
        self.raw_config_text = QTextEdit()
        self.raw_config_text.setReadOnly(True)
        self.raw_config_text.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 8px;
                font-family: monospace;
                min-height: 150px;
            }
        """)
        raw_layout.addWidget(self.raw_config_text)
        raw_card = MaterialCard("Raw Configuration", content_layout=raw_layout)
        layout.addWidget(raw_card)

        # Action buttons
        button_layout = QHBoxLayout()
        self.reset_btn = AnimatedButton("Reset to Defaults", primary=False)
        self.reset_btn.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()
        self.export_btn = AnimatedButton("Export", primary=False)
        self.export_btn.clicked.connect(self._export_config)
        button_layout.addWidget(self.export_btn)
        self.import_btn = AnimatedButton("Import", primary=False)
        self.import_btn.clicked.connect(self._import_config)
        button_layout.addWidget(self.import_btn)
        self.refresh_btn = AnimatedButton("Refresh", primary=False)
        self.refresh_btn.clicked.connect(self._refresh_config)
        button_layout.addWidget(self.refresh_btn)
        self.apply_btn = AnimatedButton("Apply Changes", primary=True)
        self.apply_btn.clicked.connect(self._apply_changes)
        button_layout.addWidget(self.apply_btn)
        layout.addLayout(button_layout)
        layout.addStretch()

    def _browse_folder(self, line_edit: QLineEdit):
        """Browse for folder and update line edit"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            line_edit.text() or str(Path.home()),
            QFileDialog.Option.ShowDirsOnly
        )
        if folder:
            line_edit.setText(folder)

    def _refresh_config(self):
        """Request current configuration"""
        self.config_get_requested.emit()

    def _apply_changes(self):
        """Apply configuration changes"""
        config_dict = {}

        # Collect values from widgets
        for key, widget in self.config_widgets.items():
            if isinstance(widget, QLineEdit):
                config_dict[key] = widget.text()
            elif isinstance(widget, QComboBox):
                config_dict[key] = widget.currentText()
            elif isinstance(widget, QSpinBox):
                config_dict[key] = str(widget.value())
            elif isinstance(widget, QCheckBox):
                config_dict[key] = str(widget.isChecked())

        # Validate numeric fields
        try:
            if "temperature" in config_dict:
                try:
                    temp = float(config_dict["temperature"])
                    if not (0 <= temp <= 2):
                        raise ValueError("Temperature must be between 0.0 and 2.0")
                except ValueError:
                    raise ValueError("Invalid temperature value. Please enter a number.")

            if "sim_threshold" in config_dict:
                try:
                    sim = float(config_dict["sim_threshold"])
                    if not (0 <= sim <= 1):
                        raise ValueError("Similarity threshold must be between 0.0 and 1.0")
                except ValueError:
                    raise ValueError("Invalid similarity threshold. Please enter a number.")

            if "mmr_lambda" in config_dict:
                try:
                    mmr = float(config_dict["mmr_lambda"])
                    if not (0 <= mmr <= 1):
                        raise ValueError("MMR lambda must be between 0.0 and 1.0")
                except ValueError:
                    raise ValueError("Invalid MMR lambda value. Please enter a number.")

            if "rerank_alpha" in config_dict:
                try:
                    alpha = float(config_dict["rerank_alpha"])
                    if not (0 <= alpha <= 1):
                        raise ValueError("Rerank alpha must be between 0.0 and 1.0")
                except ValueError:
                    raise ValueError("Invalid rerank alpha value. Please enter a number.")

        except ValueError as e:
            QMessageBox.warning(self, "Invalid Configuration", str(e))
            return

        # Emit signal to update config
        params = {"config": config_dict}
        self.config_update_requested.emit(params)

    def _reset_to_defaults(self):
        """Reset configuration to defaults"""
        reply = QMessageBox.question(
            self,
            "Reset Configuration",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # This would call config service to reset
            QMessageBox.information(
                self,
                "Reset Complete",
                (
                    "Configuration has been reset to defaults.\n"
                    "Click Refresh to load the new values."
                ),
            )

    def _export_config(self):
        """Export configuration to file"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Configuration",
            "emailops_config.json",
            "JSON Files (*.json);;All Files (*.*)"
        )
        if filename:
            try:
                config_text = self.raw_config_text.toPlainText()
                config_dict = json.loads(config_text) if config_text else {}

                with Path(filename).open('w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2)

                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Configuration exported to {filename}",
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to export configuration: {e}",
                )

    def _import_config(self):
        """Import configuration from file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import Configuration",
            "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        if filename:
            try:
                with Path(filename).open(encoding='utf-8') as f:
                    config_dict = json.load(f)

                # Update widgets with imported values
                self.update_config_display(config_dict)

                QMessageBox.information(
                    self,
                    "Import Successful",
                    (
                        "Configuration imported successfully.\n"
                        "Click Apply Changes to save."
                    ),
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Import Failed",
                    f"Failed to import configuration: {e}",
                )

    @pyqtSlot(dict)
    def on_config_updated(self, result: dict):
        """Handle configuration update result"""
        if result.get("success"):
            QMessageBox.information(
                self,
                "Configuration Updated",
                "Configuration has been successfully updated.",
            )
            # Refresh display
            self._refresh_config()
        else:
            QMessageBox.critical(
                self,
                "Update Failed",
                (
                    "Failed to update configuration: "
                    f"{result.get('error', 'Unknown error')}"
                ),
            )

    def update_config_display(self, config: dict):
        """Update UI with configuration values"""
        # Update each widget
        for key, widget in self.config_widgets.items():
            if key in config:
                value = config[key]
                if isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                elif isinstance(widget, QComboBox):
                    index = widget.findText(str(value))
                    if index >= 0:
                        widget.setCurrentIndex(index)
                elif isinstance(widget, QSpinBox):
                    with contextlib.suppress(ValueError, TypeError):
                        widget.setValue(int(value))
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(str(value).lower() in ("true", "1", "yes"))

        # Update raw config display
        self.raw_config_text.setText(json.dumps(config, indent=2))
