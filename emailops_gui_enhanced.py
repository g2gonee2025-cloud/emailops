#!/usr/bin/env python3
from __future__ import annotations

"""
EmailOps GUI - Best-in-Class Production Interface

A professional Tkinter/ttk GUI exposing all EmailOps features with production-grade UX.
"""

import argparse
import contextlib
import json
import logging
import os
import queue
import sys
import threading
import tkinter as tk
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import ttk
from typing import Any

# ------------------------------- Robust imports -------------------------------

def _try_imports():
    """Import modules with package/local fallbacks."""
    ns: dict[str, Any] = {}

    # Processor
    try:
        from emailops import processor as _processor
    except Exception:
        import processor as _processor  # type: ignore
    ns["processor"] = _processor

    # Indexer
    try:
        from emailops import email_indexer as _indexer
    except Exception:
        import email_indexer as _indexer  # type: ignore
    ns["indexer"] = _indexer

    # Summarizer
    try:
        from emailops import summarize_email_thread as _summarizer
    except Exception:
        import summarize_email_thread as _summarizer  # type: ignore
    ns["summarizer"] = _summarizer

    # Text Chunker
    try:
        from emailops import text_chunker as _chunker
    except Exception:
        with contextlib.suppress(Exception):
            import text_chunker as _chunker  # type: ignore
            ns["chunker"] = _chunker

    # Doctor
    try:
        from emailops import doctor as _doctor
    except Exception:
        import doctor as _doctor  # type: ignore
    ns["doctor"] = _doctor

    # Config
    try:
        from emailops.config import EmailOpsConfig, get_config
        ns["EmailOpsConfig"] = EmailOpsConfig
        ns["get_config"] = get_config
    except Exception:
        from config import EmailOpsConfig, get_config  # type: ignore
        ns["EmailOpsConfig"] = EmailOpsConfig
        ns["get_config"] = get_config

    # Validators
    try:
        from emailops.validators import validate_directory_path
    except Exception:
        from validators import validate_directory_path  # type: ignore
    ns["validate_directory_path"] = validate_directory_path

    # Utils logger
    try:
        from emailops.utils import logger as _module_logger
    except Exception:
        try:
            from utils import logger as _module_logger  # type: ignore
        except Exception:
            _module_logger = logging.getLogger("emailops")
    ns["module_logger"] = _module_logger

    return ns

NS = _try_imports()
processor = NS.get("processor")
email_indexer = NS.get("indexer")
summarizer = NS.get("summarizer")
text_chunker = NS.get("chunker")
doctor = NS.get("doctor")
EmailOpsConfig = NS.get("EmailOpsConfig")
get_config = NS.get("get_config")
validate_directory_path = NS.get("validate_directory_path")
module_logger = NS.get("module_logger")

# ------------------------------- Logging infrastructure -----------------------------

class QueueHandler(logging.Handler):
    """Log handler writing to a queue for GUI consumption."""
    def __init__(self, log_queue: queue.Queue[str]):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.log_queue.put_nowait(msg)
        except Exception:
            pass

def configure_logging(log_queue: queue.Queue[str]) -> None:
    """Configure logging to feed the GUI."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s: %(message)s")
    qh = QueueHandler(log_queue)
    qh.setFormatter(fmt)

    for h in root.handlers:
        root.removeHandler(h)
    root.addHandler(qh)

    if module_logger:
        module_logger.propagate = True
        module_logger.setLevel(logging.INFO)

# --------------------------------- Settings persistence -------------------------------

SETTINGS_FILE = Path.home() / ".emailops_gui.json"

@dataclass
class AppSettings:
    """Persistent application settings."""
    export_root: str = ""
    provider: str = "vertex"
    persona: str = os.getenv("PERSONA", "expert insurance CSR")
    sim_threshold: float = 0.30
    reply_tokens: int = 20000
    fresh_tokens: int = 10000
    reply_policy: str = "reply_all"
    temperature: float = 0.2
    k: int = 25
    last_to: str = ""
    last_cc: str = ""
    last_subject: str = ""
    mmr_lambda: float = 0.70
    rerank_alpha: float = 0.35
    chat_session_id: str = "default"
    max_chat_history: int = 5

    def save(self) -> None:
        with contextlib.suppress(Exception):
            SETTINGS_FILE.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load() -> AppSettings:
        try:
            if SETTINGS_FILE.exists():
                raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                return AppSettings(**raw)
        except Exception:
            pass
        return AppSettings()

# ------------------------------- Task management -----------------------------

class TaskController:
    """Cancellation and busy state management with enhanced tracking."""
    def __init__(self) -> None:
        self._busy = False
        self._cancel = False
        self._lock = threading.Lock()
        self._current_task: str = ""
        self._progress: float = 0.0

    def start(self, task_name: str = "") -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            self._cancel = False
            self._current_task = task_name
            self._progress = 0.0
            return True

    def done(self) -> None:
        with self._lock:
            self._busy = False
            self._cancel = False
            self._current_task = ""
            self._progress = 0.0

    def cancel(self) -> None:
        with self._lock:
            self._cancel = True

    def cancelled(self) -> bool:
        with self._lock:
            return self._cancel

    def busy(self) -> bool:
        with self._lock:
            return self._busy

# ----------------------------------- Main GUI Application ----------------------------------

class EmailOpsApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("EmailOps — Professional Assistant v3.0")
        self.geometry("1600x1000")
        self.minsize(1200, 700)

        # State
        self.settings = AppSettings.load()
        self.task = TaskController()
        self.log_queue: queue.Queue[str] = queue.Queue()
        configure_logging(self.log_queue)

        # Enhanced status colors with modern palette
        self.colors = {
            "success": "#28a745",
            "warning": "#ff9800",
            "error": "#dc3545",
            "info": "#0288d1",
            "primary": "#2962ff",
            "secondary": "#5e35b1",
            "accent": "#00bcd4",
            "bg_light": "#f5f5f5",
            "bg_dark": "#263238",
            "text_primary": "#212121",
            "text_secondary": "#757575",
        }

        # Progress tracking state
        self.batch_progress: dict[str, Any] = {}
        self.progress_queue: queue.Queue[dict[str, Any]] = queue.Queue()

        # Define UI elements that are referenced across methods
        self.txt_logs = tk.Text(self) # Define here to solve attribute error

        # Build UI with enhanced styling
        self._apply_theme()
        self._build_menu()
        self._build_header()
        self._build_tabs()
        self._build_log_tab()

        # Start background tasks
        self.after(100, self._drain_logs)
        self.after(200, self._update_progress_displays)

    def _apply_theme(self) -> None:
        """Apply modern theme and styling to the application."""
        try:
            style = ttk.Style()
            style.theme_use('clam')

            style.configure('TFrame', background=self.colors['bg_light'])
            style.configure('TLabel', background=self.colors['bg_light'], foreground=self.colors['text_primary'], font=('Arial', 10))
            style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground=self.colors['primary'])
            style.configure('TButton', font=('Arial', 10), padding=6)
            style.configure('Action.TButton', font=('Arial', 10, 'bold'), foreground='white', background=self.colors['primary'])
            style.map('Action.TButton', background=[('active', self.colors['accent'])])
            style.configure('TNotebook', background=self.colors['bg_light'], borderwidth=0)
            style.configure('TNotebook.Tab', font=('Arial', 10, 'bold'), foreground=self.colors['text_secondary'])
            style.map('TNotebook.Tab', foreground=[('selected', self.colors['primary'])])
            style.configure('Treeview', rowheight=25, font=('Arial', 10))
            style.configure('Treeview.Heading', font=('Arial', 10, 'bold'))

        except Exception as e:
            if module_logger:
                module_logger.warning(f"Failed to apply theme: {e}")

    def _update_progress_displays(self) -> None:
        """Update all progress displays from queue."""
        # This will be implemented in a future step
        self.after(200, self._update_progress_displays)

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)

        # File menu
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save Settings", command=self._save_settings, accelerator="Ctrl+S")
        filemenu.add_command(label="Load Settings", command=self._load_settings, accelerator="Ctrl+O")
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.destroy, accelerator="Ctrl+Q")
        menubar.add_cascade(label="File", menu=filemenu)

        # View menu
        viewmenu = tk.Menu(menubar, tearoff=0)
        viewmenu.add_command(label="Clear All Logs", command=lambda: self.txt_logs.delete("1.0", tk.END))
        viewmenu.add_command(label="Jump to Logs Tab", command=lambda: self.nb.select(self.tab_logs))
        menubar.add_cascade(label="View", menu=viewmenu)

        # Tools menu
        toolsmenu = tk.Menu(menubar, tearoff=0)
        toolsmenu.add_command(label="Run System Diagnostics", command=lambda: self.nb.select(self.tab_diagnostics))
        toolsmenu.add_command(label="Edit Configuration", command=lambda: self.nb.select(self.tab_config))
        menubar.add_cascade(label="Tools", menu=toolsmenu)

        # Help menu
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self._show_about)
        helpmenu.add_command(label="Documentation", command=self._show_docs)
        menubar.add_cascade(label="Help", menu=helpmenu)

        self.config(menu=menubar)

        # Keyboard shortcuts
        self.bind("<Control-s>", lambda e: self._save_settings())
        self.bind("<Control-o>", lambda e: self._load_settings())
        self.bind("<Control-q>", lambda e: self.destroy())

    def _build_header(self) -> None:
        frm = ttk.Frame(self, padding=8)
        frm.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(frm, text="Export Root:").pack(side=tk.LEFT)
        self.var_root = tk.StringVar(value=self.settings.export_root)
        self.ent_root = ttk.Entry(frm, width=70, textvariable=self.var_root)
        self.ent_root.pack(side=tk.LEFT, padx=6)
        ttk.Button(frm, text="Browse…", command=self._choose_root).pack(side=tk.LEFT, padx=4)

        ttk.Label(frm, text="Provider:").pack(side=tk.LEFT, padx=(20, 0))
        self.var_provider = tk.StringVar(value=self.settings.provider)
        ttk.Combobox(frm, width=10, state="readonly", textvariable=self.var_provider,
                     values=["vertex"]).pack(side=tk.LEFT, padx=4)

        ttk.Label(frm, text="Temp:").pack(side=tk.LEFT, padx=(20,0))
        self.var_temp = tk.DoubleVar(value=self.settings.temperature)
        ttk.Spinbox(frm, from_=0.0, to=1.0, increment=0.05, width=6, textvariable=self.var_temp).pack(side=tk.LEFT)

        ttk.Label(frm, text="Persona:").pack(side=tk.LEFT, padx=(20, 0))
        self.var_persona = tk.StringVar(value=self.settings.persona)
        ttk.Entry(frm, width=25, textvariable=self.var_persona).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(frm, textvariable=self.status_var, foreground="#555")
        self.status_label.pack(side=tk.RIGHT, padx=10)

    def _build_tabs(self) -> None:
        self.nb = ttk.Notebook(self)
        self.nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Create all tabs
        self.tab_search = ttk.Frame(self.nb)
        self.tab_reply = ttk.Frame(self.nb)
        self.tab_fresh = ttk.Frame(self.nb)
        self.tab_chat = ttk.Frame(self.nb)
        self.tab_convs = ttk.Frame(self.nb)
        self.tab_index = ttk.Frame(self.nb)
        self.tab_config = ttk.Frame(self.nb)
        self.tab_diagnostics = ttk.Frame(self.nb)
        self.tab_chunking = ttk.Frame(self.nb)
        self.tab_analyze = ttk.Frame(self.nb)
        self.tab_logs = ttk.Frame(self.nb)

        # Add tabs in logical order
        self.nb.add(self.tab_search, text="🔍 Search")
        self.nb.add(self.tab_reply, text="↩️ Draft Reply")
        self.nb.add(self.tab_fresh, text="✉️ Draft Fresh")
        self.nb.add(self.tab_chat, text="💬 Chat")
        self.nb.add(self.tab_convs, text="📁 Conversations")
        self.tab_batch = ttk.Frame(self.nb)
        self.nb.add(self.tab_batch, text="⚡ Batch Operations")
        self.nb.add(self.tab_index, text="🔨 Index")
        self.nb.add(self.tab_config, text="⚙️ Configuration")
        self.nb.add(self.tab_diagnostics, text="🏥 Diagnostics")
        self.nb.add(self.tab_chunking, text="✂️ Chunking")
        self.nb.add(self.tab_analyze, text="📊 Analyze")
        self.nb.add(self.tab_logs, text="📝 Logs")

    def _build_log_tab(self) -> None:
        # Placeholder for future log tab implementation
        pass

    def _drain_logs(self) -> None:
        # Placeholder for future log draining implementation
        self.after(100, self._drain_logs)

    def _save_settings(self):
        # Placeholder for save settings functionality
        pass

    def _load_settings(self):
        # Placeholder for load settings functionality
        pass

    def _show_about(self):
        # Placeholder for about dialog
        pass

    def _show_docs(self):
        # Placeholder for documentation dialog
        pass

    def _choose_root(self):
        # Placeholder for export root selection
        pass

def main() -> None:
    """Main entry point for the GUI application."""
    parser = argparse.ArgumentParser(description="EmailOps Professional GUI")
    parser.add_argument("--root", help="Initial export root directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug and module_logger:
        logging.getLogger().setLevel(logging.DEBUG)
        module_logger.setLevel(logging.DEBUG)

    app = EmailOpsApp()

    if args.root:
        app.settings.export_root = args.root
        app.settings.save()

    try:
        app.mainloop()
    except KeyboardInterrupt:
        if module_logger:
            module_logger.info("Application interrupted by user")
    except Exception as e:
        if module_logger:
            module_logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
