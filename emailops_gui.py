#!/usr/bin/env python3
from __future__ import annotations

"""
EmailOps GUI - Best-in-Class Production Interface

A professional Tkinter/ttk GUI exposing all EmailOps features with production-grade UX:

CORE FEATURES:
- Advanced Search with filters, MMR, and reranking
- Draft Reply (.eml) with full parameter control + BATCH operations
- Draft Fresh Email (.eml) with advanced options
- Chat with session management and history
- Conversation browsing with FULL VIEWING capabilities
- Index building with parallel workers + REAL-TIME progress
- Configuration management for all settings
- System diagnostics and health checks
- Text chunking with live preview
- Thread analysis with BATCH SUMMARIZATION
- Comprehensive logging with level filtering

PRODUCTION ENHANCEMENTS:
- Batch email summarization across multiple conversations
- Batch draft and reply operations
- Real-time progress visualization for multiprocessing
- File pointers with Conversation.txt viewing
- Attachment opening from conversation browser
- Best-in-class visual design with modern UI elements
- Comprehensive progress tracking and status updates
- Export capabilities for results
"""

import argparse
import asyncio
import contextlib
import csv
import json
import logging
import os
import platform
import queue
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

# ------------------------------- Robust imports -------------------------------

# It's generally better to have clear, direct imports.
# If this is intended to run in a different environment,
# it's better to handle that with PYTHONPATH or packaging.
try:
    from emailops import doctor, email_indexer, processor, text_chunker
    from emailops import summarize_email_thread as summarizer
    from emailops.config import EmailOpsConfig, get_config
    from emailops.utils import logger as module_logger
    from emailops.validators import validate_directory_path
except ImportError as e:
    module_logger = logging.getLogger("emailops")
    module_logger.error(f"Failed to import from emailops package: {e}. Ensure it's installed and in the PYTHONPATH.")
    # Provide a graceful exit or fallback if the GUI can't function.
    sys.exit(f"Critical import error: {e}")

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

    try:
        module_logger.propagate = True
        module_logger.setLevel(logging.INFO)
    except Exception:
        pass


# --------------------------------- Settings persistence -------------------------------

SETTINGS_FILE = Path.home() / ".emailops_gui.json"

@dataclass
class AppSettings:
    """Persistent application settings."""
    # Export and provider settings
    export_root: str = ""
    provider: str = "vertex"
    persona: str = os.getenv("PERSONA", "expert insurance CSR")

    # Search settings
    sim_threshold: float = 0.30
    k: int = 25
    mmr_lambda: float = 0.70
    rerank_alpha: float = 0.35

    # Email generation settings
    reply_tokens: int = 20000
    fresh_tokens: int = 10000
    reply_policy: str = "reply_all"
    temperature: float = 0.2

    # Chat settings
    chat_session_id: str = "default"
    max_chat_history: int = 5

    # Last used values
    last_to: str = ""
    last_cc: str = ""
    last_subject: str = ""

    # Vertex/GCP settings
    vertex_embed_model: str = "gemini-embedding-001"
    gcp_project: str = ""
    gcp_region: str = "global"
    vertex_location: str = "us-central1"

    # Email settings
    sender_locked_name: str = ""
    sender_locked_email: str = ""
    message_id_domain: str = ""

    # MEDIUM #35: Fixed duplicate fields - consolidated num_workers and default_workers into single field
    # Processing settings
    num_workers: int = 4
    batch_size: int = 64
    chunk_size: int = 1600
    chunk_overlap: int = 200

    def save(self) -> None:
        try:
            # Ensure parent directory exists
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict and save
            settings_dict = asdict(self)
            json_content = json.dumps(settings_dict, ensure_ascii=False, indent=2)

            # Write atomically using a temporary file and os.replace
            import tempfile
            fd, temp_path_str = tempfile.mkstemp(dir=SETTINGS_FILE.parent, prefix=SETTINGS_FILE.name)
            temp_path = Path(temp_path_str)

            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(json_content)
                os.replace(temp_path, SETTINGS_FILE)
            finally:
                if temp_path.exists():
                    with contextlib.suppress(OSError):
                        temp_path.unlink()

            module_logger.info(f"✓ Settings saved to {SETTINGS_FILE}")
            module_logger.debug(f"Settings content: {json_content[:200]}...")
        except Exception as e:
            module_logger.error(f"✗ Failed to save settings to {SETTINGS_FILE}: {e}", exc_info=True)
            raise

    @staticmethod
    def load() -> AppSettings:
        try:
            if SETTINGS_FILE.exists():
                content = SETTINGS_FILE.read_text(encoding="utf-8")
                module_logger.debug(f"Loading settings from {SETTINGS_FILE}, size: {len(content)} bytes")
                raw = json.loads(content)
                module_logger.info(f"✓ Loaded settings from {SETTINGS_FILE}")
                return AppSettings(**raw)
            else:
                module_logger.info(f"No settings file found at {SETTINGS_FILE}, using defaults")
        except Exception as e:
            module_logger.warning(f"Failed to load settings from {SETTINGS_FILE}: {e}, using defaults")
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
        self._status_message: str = ""

    def start(self, task_name: str = "") -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            self._cancel = False
            self._current_task = task_name
            self._progress = 0.0
            self._status_message = ""
            return True

    def done(self) -> None:
        with self._lock:
            self._busy = False
            self._cancel = False
            self._current_task = ""
            self._progress = 0.0
            self._status_message = ""

    def cancel(self) -> None:
        with self._lock:
            self._cancel = True

    def cancelled(self) -> bool:
        with self._lock:
            return self._cancel

    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def set_progress(self, progress: float, message: str = "") -> None:
        """Update progress and status message."""
        with self._lock:
            self._progress = max(0.0, min(1.0, progress))
            self._status_message = message

    def get_status(self) -> tuple[float, str]:
        """Get current progress and status message."""
        with self._lock:
            return self._progress, self._status_message


# ----------------------------------- Main GUI Application ----------------------------------

def run_with_progress(task_name: str, progress_bar: str, status_label: str, *buttons_to_disable):
    """Decorator to run a class method in a thread and manage UI state."""
    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        def wrapper(self: 'EmailOpsApp', *args: Any, **kwargs: Any) -> None:
            if not self.task.start(task_name):
                return

            pb = getattr(self, progress_bar, None)
            lbl = getattr(self, status_label, None)
            buttons = [getattr(self, btn_name, None) for btn_name in buttons_to_disable if hasattr(self, btn_name)]

            def set_buttons_state(state: str) -> None:
                for btn in buttons:
                    if btn:
                        self.after(0, lambda b=btn: b.config(state=state))

            def start_progress() -> None:
                if pb:
                    self.after(0, pb.start)

            def stop_progress() -> None:
                if pb:
                    self.after(0, pb.stop)

            def update_progress(current: int, total: int, message: str = "") -> None:
                if pb:
                    pb['maximum'] = total
                    pb['value'] = current
                if lbl:
                    lbl.config(text=message)

            def show_error(e: Exception) -> None:
                module_logger.error(f"Task '{task_name}' failed: {e}", exc_info=True)
                self.after(0, lambda: messagebox.showerror("Error", f"Task failed:\n{e!s}"))

            def task_wrapper() -> None:
                try:
                    set_buttons_state("disabled")
                    start_progress()
                    # The decorated function expects `update_progress` as a keyword argument.
                    kwargs['update_progress'] = update_progress
                    func(self, *args, **kwargs)
                except Exception as e:
                    show_error(e)
                finally:
                    set_buttons_state("normal")
                    stop_progress()
                    self.task.done()

            threading.Thread(target=task_wrapper, daemon=True).start()
        return wrapper
    return decorator

class LabeledSpinbox(ttk.Frame):
    """A custom widget combining a Label and a Spinbox."""
    def __init__(self, parent, text, from_, to, increment, width, textvariable):
        super().__init__(parent)
        ttk.Label(self, text=text).pack(side=tk.LEFT)
        ttk.Spinbox(self, from_=from_, to=to, increment=increment, width=width, textvariable=textvariable).pack(side=tk.LEFT, padx=4)

class EmailOpsApp(tk.Tk):
    def _reset_config(self) -> None:
        try:
            # Set UI fields to provided default values
            self.var_gcp_project.set("semiotic-nexus-470620-f3")
            self.var_gcp_region.set("global")
            self.var_vertex_location.set("us-central1")
            self.var_cfg_chunk_size.set(1600)
            self.var_cfg_chunk_overlap.set(200)
            self.var_cfg_batch.set(128)
            self.var_cfg_workers.set(8)
            self.var_sender_name.set("")
            self.var_sender_email.set("")
            self.var_msg_id_domain.set("")

            # Update settings object as well
            self.settings.gcp_project = "semiotic-nexus-470620-f3"
            self.settings.gcp_region = "global"
            self.settings.vertex_location = "us-central1"
            self.settings.chunk_size = 1600
            self.settings.chunk_overlap = 200
            self.settings.batch_size = 128
            self.settings.num_workers = 8
            self.settings.sender_locked_name = ""
            self.settings.sender_locked_email = ""
            self.settings.message_id_domain = ""
            self.settings.save()
            self.config_status.config(text="✓ Configuration reset to defaults", foreground=self.colors["info"])
        except Exception as e:
            self.config_status.config(text=f"✗ Failed to reset config: {e}", foreground=self.colors["error"])
    def __init__(self) -> None:
        super().__init__()
        self.title("EmailOps — Professional Assistant v3.0")
        self.geometry("1200x800")
        self.minsize(1000, 600)

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
            "progress_bg": "#e0e0e0",
            "progress_fg": "#4caf50",
        }

        # Progress tracking state
        self.batch_progress: dict[str, Any] = {}
        self.progress_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.current_operation: str = ""

        # Search results state
        self.search_results: list[dict[str, Any]] = []
        self._chunk_results: list[dict[str, Any]] = []

        # Build UI with enhanced styling
        self._apply_theme()
        self._build_menu()
        self._build_header()
        self._build_tabs()

        # Start background tasks
        self.after(100, self._drain_logs)
        self.after(200, self._update_progress_displays)

        module_logger.info("EmailOps GUI v3.0 initialized")

    def print_total_chunks(self) -> None:
        """Show total number of chunks made across all conversations."""
        if not hasattr(self, '_chunk_results') or not self._chunk_results:
            messagebox.showinfo("Chunk Count", "No chunk results available. Run chunking operations first.")
            return
        total = sum(conv.get("chunks", 0) for conv in self._chunk_results)
        messagebox.showinfo("Chunk Count", f"Total chunks created: {total}")
        module_logger.info(f"Total chunks across all conversations: {total}")

    # ------------- Theme and Progress Methods -------------

    def _apply_theme(self) -> None:
        """Apply modern theme and styling to the application."""
        try:
            style = ttk.Style()
            style.theme_use('clam')

            # Configure button styles
            style.configure('Action.TButton', font=('Arial', 10, 'bold'), padding=6)
            style.configure('Primary.TButton', foreground=self.colors['primary'])
            style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground=self.colors['primary'])

        except Exception as e:
            module_logger.warning(f"Failed to apply theme: {e}")

    def _update_progress_displays(self) -> None:
        """Update all progress displays from queue with enhanced visualization."""
        try:
            while True:
                try:
                    progress_info = self.progress_queue.get_nowait()
                    operation = progress_info.get("operation", "")
                    progress = progress_info.get("progress", 0)
                    total = progress_info.get("total", 100)
                    message = progress_info.get("message", "Processing...")

                    # Update appropriate progress display based on operation
                    if operation == "index" and hasattr(self, 'lbl_index_progress'):
                        self.lbl_index_progress.config(text=f"{message} ({progress}/{total})")
                        if hasattr(self, 'pb_index') and total > 0:
                            self.pb_index['mode'] = 'determinate'
                            self.pb_index['maximum'] = total
                            self.pb_index['value'] = progress
                    elif (operation == "batch_summarize" and hasattr(self, 'lbl_batch_progress')) or (operation == "batch_reply" and hasattr(self, 'lbl_batch_progress')):
                        self.lbl_batch_progress.config(text=f"{message} ({progress}/{total})")
                        if hasattr(self, 'pb_batch') and total > 0:
                            self.pb_batch['maximum'] = total
                            self.pb_batch['value'] = progress

                except queue.Empty:
                    break
        except Exception as e:
            module_logger.debug(f"Progress update error: {e}")

        self.after(200, self._update_progress_displays)

    # ------------- Menu Bar -------------

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
        viewmenu.add_separator()
        viewmenu.add_command(label="Export Search Results", command=self._export_search_results)
        viewmenu.add_command(label="Export Chat History", command=self._export_chat_history)
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
        self.bind("<Control-s>", self._save_settings)
        self.bind("<Control-o>", self._load_settings)
        self.bind("<Control-q>", lambda e: self.destroy())

    # ------------- Header -------------

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
        self.status_label.config(foreground=self.colors['success'])
    # ------------- Tab Structure -------------

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

        # Build each tab
        self._build_search_tab()
        self._build_reply_tab()
        self._build_fresh_tab()
        self._build_chat_tab()
        self._build_conversations_tab()
        self._build_batch_tab()
        self._build_index_tab()
        self._build_config_tab()
        self._build_diagnostics_tab()
        self._build_chunking_tab()
        self._build_analyze_tab()
        self._build_log_tab()

    # ------------- Search Tab (Enhanced) -------------

    def _build_search_tab(self) -> None:
        frm = self.tab_search

        # Basic search controls
        basic_frame = ttk.LabelFrame(frm, text="Search Query", padding=8)
        basic_frame.pack(fill=tk.X, padx=8, pady=8)

        row1 = ttk.Frame(basic_frame)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="Query:").pack(side=tk.LEFT)
        self.var_search_q = tk.StringVar()
        ttk.Entry(row1, width=70, textvariable=self.var_search_q).pack(side=tk.LEFT, padx=6)
        self.var_k = tk.IntVar(value=self.settings.k)
        LabeledSpinbox(row1, "k:", 1, 250, 1, 6, self.var_k).pack(side=tk.LEFT, padx=4)
        self.var_sim = tk.DoubleVar(value=self.settings.sim_threshold)
        LabeledSpinbox(row1, "Sim ≥", 0.0, 1.0, 0.01, 6, self.var_sim).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(basic_frame)
        row2.pack(fill=tk.X, pady=(8,0))
        self.btn_search = ttk.Button(row2, text="Search", command=partial(self._on_search))
        self.btn_search.pack(side=tk.LEFT)
        self.pb_search = ttk.Progressbar(row2, mode="indeterminate", length=150)
        self.pb_search.pack(side=tk.LEFT, padx=8)
        ttk.Button(row2, text="Clear Results", command=lambda: self.tree.delete(*self.tree.get_children())).pack(side=tk.LEFT, padx=4)

        # Advanced filters (collapsible)
        self.show_advanced = tk.BooleanVar(value=False)
        adv_toggle = ttk.Checkbutton(frm, text="Show Advanced Filters",
                                     variable=self.show_advanced, command=self._toggle_advanced_search)
        adv_toggle.pack(anchor="w", padx=8)

        self.advanced_frame = ttk.LabelFrame(frm, text="Advanced Filters & Ranking", padding=8)

        # Filters row 1
        f1 = ttk.Frame(self.advanced_frame)
        f1.pack(fill=tk.X, pady=2)
        ttk.Label(f1, text="From:").pack(side=tk.LEFT)
        self.var_from_filter = tk.StringVar()
        ttk.Entry(f1, width=30, textvariable=self.var_from_filter).pack(side=tk.LEFT, padx=4)
        ttk.Label(f1, text="To:").pack(side=tk.LEFT, padx=(10,0))
        self.var_to_filter = tk.StringVar()
        ttk.Entry(f1, width=30, textvariable=self.var_to_filter).pack(side=tk.LEFT, padx=4)
        ttk.Label(f1, text="Subject:").pack(side=tk.LEFT, padx=(10,0))
        self.var_subject_filter = tk.StringVar()
        ttk.Entry(f1, width=30, textvariable=self.var_subject_filter).pack(side=tk.LEFT, padx=4)

        # Filters row 2
        f2 = ttk.Frame(self.advanced_frame)
        f2.pack(fill=tk.X, pady=2)
        ttk.Label(f2, text="After Date:").pack(side=tk.LEFT)
        self.var_after_date = tk.StringVar()
        ttk.Entry(f2, width=20, textvariable=self.var_after_date).pack(side=tk.LEFT, padx=4)
        ttk.Label(f2, text="(YYYY-MM-DD)", font=("Arial", 8), foreground="#666").pack(side=tk.LEFT)

        ttk.Label(f2, text="Before Date:").pack(side=tk.LEFT, padx=(10,0))
        self.var_before_date = tk.StringVar()
        ttk.Entry(f2, width=20, textvariable=self.var_before_date).pack(side=tk.LEFT, padx=4)
        ttk.Label(f2, text="(YYYY-MM-DD)", font=("Arial", 8), foreground="#666").pack(side=tk.LEFT)

        # Ranking controls
        f3 = ttk.Frame(self.advanced_frame)
        f3.pack(fill=tk.X, pady=(8,2))
        ttk.Label(f3, text="MMR Lambda (relevance vs diversity):").pack(side=tk.LEFT)
        self.var_mmr_lambda = tk.DoubleVar(value=self.settings.mmr_lambda)
        ttk.Scale(f3, from_=0.0, to=1.0, variable=self.var_mmr_lambda, orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT, padx=4)
        self.lbl_mmr = ttk.Label(f3, text=f"{self.settings.mmr_lambda:.2f}")
        self.lbl_mmr.pack(side=tk.LEFT)
        self.var_mmr_lambda.trace_add("write", lambda *_: self.lbl_mmr.config(text=f"{self.var_mmr_lambda.get():.2f}"))

        f4 = ttk.Frame(self.advanced_frame)
        f4.pack(fill=tk.X, pady=2)
        ttk.Label(f4, text="Rerank Alpha (boost vs summary):").pack(side=tk.LEFT)
        self.var_rerank_alpha = tk.DoubleVar(value=self.settings.rerank_alpha)
        ttk.Scale(f4, from_=0.0, to=1.0, variable=self.var_rerank_alpha, orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT, padx=4)
        self.lbl_rerank = ttk.Label(f4, text=f"{self.settings.rerank_alpha:.2f}")
        self.lbl_rerank.pack(side=tk.LEFT)
        self.var_rerank_alpha.trace_add("write", lambda *_: self.lbl_rerank.config(text=f"{self.var_rerank_alpha.get():.2f}"))

        # Results display
        results_frame = ttk.Frame(frm, padding=(8,0,8,8))
        results_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("score", "subject", "id")
        self.tree = ttk.Treeview(results_frame, columns=cols, show="headings", height=15)
        self.tree.heading("score", text="Score")
        self.tree.heading("subject", text="Subject")
        self.tree.heading("id", text="Doc ID")
        self.tree.column("score", width=70, anchor=tk.CENTER)
        self.tree.column("subject", width=600)
        self.tree.column("id", width=600)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._show_snippet)

        yscroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.tree.yview)
        yscroll.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.configure(yscrollcommand=yscroll.set)

        right = ttk.Frame(results_frame)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8,0))
        ttk.Label(right, text="Snippet:").pack(anchor="w")
        self.txt_snip = tk.Text(right, height=15, wrap="word", font=("Courier", 9), state="disabled")
        self.txt_snip.pack(fill=tk.BOTH, expand=True)

    def _toggle_advanced_search(self) -> None:
        if self.show_advanced.get():
            self.advanced_frame.pack(fill=tk.X, padx=8, pady=(0,8), before=self.advanced_frame.master.children[list(self.advanced_frame.master.children.keys())[2]])
        else:
            self.advanced_frame.pack_forget()

    # ------------- Reply Tab -------------

    def _build_reply_tab(self) -> None:
        frm = self.tab_reply

        row1 = ttk.Frame(frm, padding=8)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="Conversation ID:").pack(side=tk.LEFT)
        self.var_conv = tk.StringVar()
        self.cmb_conv = ttk.Combobox(row1, width=50, textvariable=self.var_conv)
        self.cmb_conv.pack(side=tk.LEFT, padx=6)
        ttk.Button(row1, text="Load Conversations", command=self._load_conversations).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(frm, padding=8)
        row2.pack(fill=tk.X)
        ttk.Label(row2, text="Query (optional):").pack(side=tk.LEFT)
        self.var_reply_q = tk.StringVar()
        ttk.Entry(row2, width=50, textvariable=self.var_reply_q).pack(side=tk.LEFT, padx=6)
        self.var_reply_tokens = tk.IntVar(value=self.settings.reply_tokens)
        LabeledSpinbox(row2, "Tokens:", 2000, 100000, 1000, 10, self.var_reply_tokens).pack(side=tk.LEFT, padx=4)
        ttk.Label(row2, text="Policy:").pack(side=tk.LEFT)
        self.var_reply_policy = tk.StringVar(value=self.settings.reply_policy)
        ttk.Combobox(row2, width=12, state="readonly", textvariable=self.var_reply_policy,
                     values=["reply_all", "smart", "sender_only"]).pack(side=tk.LEFT, padx=4)
        self.var_reply_attach = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="Include attachments", variable=self.var_reply_attach).pack(side=tk.LEFT, padx=12)

        row3 = ttk.Frame(frm, padding=8)
        row3.pack(fill=tk.X)
        self.btn_draft_reply = ttk.Button(row3, text="Generate Reply", command=partial(self._on_draft_reply))
        self.btn_draft_reply.pack(side=tk.LEFT)
        self.pb_reply = ttk.Progressbar(row3, mode="indeterminate", length=220)
        self.pb_reply.pack(side=tk.LEFT, padx=8)
        ttk.Button(row3, text="Save .eml…", command=self._save_eml_reply).pack(side=tk.LEFT, padx=8)

        self.txt_reply = tk.Text(frm, wrap="word", font=("Arial", 10))
        self.txt_reply.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

        self._last_reply_bytes = None
        self._last_reply_meta = None

    # ------------- Fresh Email Tab -------------

    def _build_fresh_tab(self) -> None:
        frm = self.tab_fresh

        r1 = ttk.Frame(frm, padding=8); r1.pack(fill=tk.X)
        ttk.Label(r1, text="To:").pack(side=tk.LEFT)
        self.var_to = tk.StringVar(value=self.settings.last_to)
        ttk.Entry(r1, width=45, textvariable=self.var_to).pack(side=tk.LEFT, padx=6)

        ttk.Label(r1, text="Cc:").pack(side=tk.LEFT)
        self.var_cc = tk.StringVar(value=self.settings.last_cc)
        ttk.Entry(r1, width=45, textvariable=self.var_cc).pack(side=tk.LEFT, padx=6)

        r2 = ttk.Frame(frm, padding=8); r2.pack(fill=tk.X)
        ttk.Label(r2, text="Subject:").pack(side=tk.LEFT)
        self.var_subject = tk.StringVar(value=self.settings.last_subject)
        ttk.Entry(r2, width=80, textvariable=self.var_subject).pack(side=tk.LEFT, padx=6)

        r3 = ttk.Frame(frm, padding=8); r3.pack(fill=tk.X)
        ttk.Label(r3, text="Intent/Instructions:").pack(side=tk.LEFT)
        self.var_fresh_q = tk.StringVar()
        ttk.Entry(r3, width=70, textvariable=self.var_fresh_q).pack(side=tk.LEFT, padx=6)
        self.var_fresh_tokens = tk.IntVar(value=self.settings.fresh_tokens)
        LabeledSpinbox(r3, "Tokens:", 2000, 100000, 1000, 10, self.var_fresh_tokens).pack(side=tk.LEFT, padx=4)
        self.var_fresh_attach = tk.BooleanVar(value=True)
        ttk.Checkbutton(r3, text="Include attachments", variable=self.var_fresh_attach).pack(side=tk.LEFT, padx=12)

        r4 = ttk.Frame(frm, padding=8); r4.pack(fill=tk.X)
        self.btn_draft_fresh = ttk.Button(r4, text="Generate Fresh Email", command=partial(self._on_draft_fresh))
        self.btn_draft_fresh.pack(side=tk.LEFT)
        self.pb_fresh = ttk.Progressbar(r4, mode="indeterminate", length=220)
        self.pb_fresh.pack(side=tk.LEFT, padx=8)
        ttk.Button(r4, text="Save .eml…", command=self._save_eml_fresh).pack(side=tk.LEFT, padx=8)

        self.txt_fresh = tk.Text(frm, wrap="word", font=("Arial", 10))
        self.txt_fresh.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

        self._last_fresh_bytes = None
        self._last_fresh_meta = None

    # ------------- Chat Tab (Enhanced with Sessions) -------------

    def _build_chat_tab(self) -> None:
        frm = self.tab_chat

        # Session management panel
        session_frame = ttk.LabelFrame(frm, text="Session Management", padding=8)
        session_frame.pack(fill=tk.X, padx=8, pady=8)

        s1 = ttk.Frame(session_frame)
        s1.pack(fill=tk.X)
        ttk.Label(s1, text="Session ID:").pack(side=tk.LEFT)
        self.var_session_id = tk.StringVar(value=self.settings.chat_session_id)
        ttk.Entry(s1, width=30, textvariable=self.var_session_id).pack(side=tk.LEFT, padx=6)
        ttk.Button(s1, text="Load", command=self._load_chat_session).pack(side=tk.LEFT, padx=2)
        ttk.Button(s1, text="Save", command=self._save_chat_session).pack(side=tk.LEFT, padx=2)
        ttk.Button(s1, text="Reset", command=self._reset_chat_session).pack(side=tk.LEFT, padx=2)
        self.var_max_history = tk.IntVar(value=self.settings.max_chat_history)
        LabeledSpinbox(s1, "Max History:", 1, 10, 1, 5, self.var_max_history).pack(side=tk.LEFT, padx=(20,0))

        # Query panel
        query_frame = ttk.LabelFrame(frm, text="Question", padding=8)
        query_frame.pack(fill=tk.X, padx=8, pady=(0,8))

        r1 = ttk.Frame(query_frame)
        r1.pack(fill=tk.X)
        self.var_chat_q = tk.StringVar()
        ttk.Entry(r1, width=80, textvariable=self.var_chat_q).pack(side=tk.LEFT, padx=6)
        self.var_chat_k = tk.IntVar(value=self.settings.k)
        LabeledSpinbox(r1, "k:", 1, 100, 1, 6, self.var_chat_k).pack(side=tk.LEFT, padx=4)
        self.btn_chat = ttk.Button(r1, text="Ask", command=partial(self._on_chat))
        self.btn_chat.pack(side=tk.LEFT, padx=8)
        self.pb_chat = ttk.Progressbar(r1, mode="indeterminate", length=180)
        self.pb_chat.pack(side=tk.LEFT, padx=8)

        self.txt_chat = tk.Text(frm, wrap="word", font=("Arial", 10))
        self.txt_chat.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

    # ------------- Conversations Tab (Enhanced with Viewing/Opening) -------------

    def _build_conversations_tab(self) -> None:
        frm = self.tab_convs

        # Top control panel
        top = ttk.Frame(frm, padding=8)
        top.pack(fill=tk.X)
        self.btn_list_convs = ttk.Button(top, text="🔄 List Conversations", command=partial(self._on_list_convs), style='Action.TButton')
        self.btn_list_convs.pack(side=tk.LEFT, padx=2)
        self.pb_convs = ttk.Progressbar(top, mode="indeterminate", length=180)
        self.pb_convs.pack(side=tk.LEFT, padx=8)

        # Action buttons with icons
        ttk.Button(top, text="📖 View Conversation.txt", command=self._view_conversation_txt).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="📎 Open Attachments", command=self._open_attachments_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="📂 Open Folder", command=self._open_conversation_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="📋 Export List", command=self._export_conversation_list).pack(side=tk.LEFT, padx=2)

        # Main conversation list
        cols = ("conv_id", "subject", "first", "last", "count")
        self.tree_convs = ttk.Treeview(frm, columns=cols, show="headings", height=18)
        self.tree_convs.heading("conv_id", text="Conversation ID")
        self.tree_convs.heading("subject", text="Subject")
        self.tree_convs.heading("first", text="First Date")
        self.tree_convs.heading("last", text="Last Date")
        self.tree_convs.heading("count", text="Emails")
        self.tree_convs.column("conv_id", width=220)
        self.tree_convs.column("subject", width=520)
        self.tree_convs.column("first", width=140)
        self.tree_convs.column("last", width=140)
        self.tree_convs.column("count", width=70, anchor=tk.CENTER)

        # Add scrollbar
        conv_scroll = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=self.tree_convs.yview)
        self.tree_convs.configure(yscrollcommand=conv_scroll.set)
        self.tree_convs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        conv_scroll.pack(side=tk.LEFT, fill=tk.Y, pady=8)

        # Double-click to view conversation
        self.tree_convs.bind("<Double-Button-1>", self._view_conversation_txt)

        # Bottom action panel
        bottom_actions = ttk.Frame(frm, padding=8)
        bottom_actions.pack(fill=tk.X, padx=8, pady=(0,8))
        ttk.Button(bottom_actions, text="↩️ Use in Draft Reply", command=self._use_selected_conv, style='Primary.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom_actions, text="⚡ Add to Batch", command=self._add_selected_to_batch).pack(side=tk.LEFT, padx=2)

    # ------------- Index Tab (Enhanced with Progress) -------------

    def _build_index_tab(self) -> None:
        frm = self.tab_index

        # Main controls
        ctrl_frame = ttk.LabelFrame(frm, text="Index Configuration", padding=10)
        ctrl_frame.pack(fill=tk.X, padx=8, pady=8)

        r1 = ttk.Frame(ctrl_frame)
        r1.pack(fill=tk.X, pady=2)
        self.var_batch = tk.IntVar(value=int(os.getenv("EMBED_BATCH", "64")))
        LabeledSpinbox(r1, "Batch size:", 1, 250, 1, 8, self.var_batch).pack(side=tk.LEFT, padx=4)

        self.var_workers = tk.IntVar(value=self._detect_worker_count())
        LabeledSpinbox(r1, "Workers:", 1, 16, 1, 6, self.var_workers).pack(side=tk.LEFT, padx=(20,0))
        detected = self._detect_worker_count()
        ttk.Label(r1, text=f"(detected: {detected})", font=("Arial", 8), foreground="#666").pack(side=tk.LEFT)

        r2 = ttk.Frame(ctrl_frame)
        r2.pack(fill=tk.X, pady=(8,2))
        self.var_force = tk.BooleanVar(value=False)
        ttk.Checkbutton(r2, text="Force full re-index", variable=self.var_force).pack(side=tk.LEFT, padx=4)
        self.var_limit = tk.IntVar(value=0)
        LabeledSpinbox(r2, "Limit per conversation:", 0, 2000, 1, 8, self.var_limit).pack(side=tk.LEFT, padx=(20,0))
        ttk.Label(r2, text="(0 = unlimited)", font=("Arial", 8), foreground="#666").pack(side=tk.LEFT)

        # Action panel with progress
        action_frame = ttk.Frame(frm, padding=8)
        action_frame.pack(fill=tk.X, padx=8)
        self.btn_build = ttk.Button(action_frame, text="🔨 Build / Update Index", command=partial(self._on_build_index), style='Primary.TButton')
        self.btn_build.pack(side=tk.LEFT, padx=4)
        self.pb_index = ttk.Progressbar(action_frame, mode="indeterminate", length=260)
        self.pb_index.pack(side=tk.LEFT, padx=8)
        self.lbl_index_progress = ttk.Label(action_frame, text="")
        self.lbl_index_progress.pack(side=tk.LEFT, padx=8)

        # Helpful hints
        hint = ttk.Label(frm, text="💡 Tip: Parallel indexing uses 1 worker per GCP credential. Monitor logs for real-time progress.",
                        foreground=self.colors["info"], font=("Arial", 9))
        hint.pack(anchor="w", padx=8, pady=(0,8))

    def _view_conversation_txt(self, event: tk.Event | None = None) -> None:
        """View Conversation.txt content in a new window."""
        selection = self.tree_convs.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a conversation to view")
            return

        try:
            values = self.tree_convs.item(selection[0])["values"]
            conv_id = values[0]
            subject = values[1] if len(values) > 1 else "No Subject"

            if not self.settings.export_root:
                messagebox.showerror("Error", "Export root not set")
                return

            conv_path = Path(self.settings.export_root) / conv_id / "Conversation.txt"
            if not conv_path.exists():
                messagebox.showerror("Not Found", f"Conversation.txt not found:\n{conv_path}")
                return

            # Create viewer window
            viewer = tk.Toplevel(self)
            viewer.title(f"Conversation: {subject}")
            viewer.geometry("900x700")

            # Header with conversation info
            header = ttk.Frame(viewer, padding=10)
            header.pack(fill=tk.X)
            ttk.Label(header, text=f"ID: {conv_id}", font=("Arial", 10, "bold")).pack(anchor="w")
            ttk.Label(header, text=f"Subject: {subject}", font=("Arial", 10)).pack(anchor="w")

            # Text display with scrollbar
            text_frame = ttk.Frame(viewer)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

            text_widget = tk.Text(text_frame, wrap="word", font=("Courier", 9))
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text_widget.configure(yscrollcommand=scrollbar.set)

            # Load and display content
            content = ""
            try:
                content = conv_path.read_text(encoding="utf-8", errors="ignore")
                text_widget.insert("1.0", content)
                text_widget.config(state="disabled")
                module_logger.info(f"Viewing conversation: {conv_id}")
            except Exception as e:
                error_msg = f"Error loading file:\n{e!s}"
                text_widget.insert("1.0", error_msg)
                content = error_msg
                module_logger.error(f"Failed to load conversation text: {e}")

            # Action buttons
            btn_frame = ttk.Frame(viewer, padding=10)
            btn_frame.pack(fill=tk.X)
            ttk.Button(btn_frame, text="📂 Open Folder",
                      command=lambda p=conv_path.parent: self._open_path(p)).pack(side=tk.LEFT, padx=4)
            ttk.Button(btn_frame, text="💾 Save Copy",
                      command=lambda c=content, n=f"{conv_id}_conversation.txt": self._save_text_copy(c, n)).pack(side=tk.LEFT, padx=4)
            ttk.Button(btn_frame, text="Close", command=viewer.destroy).pack(side=tk.RIGHT, padx=4)

        except Exception as e:
            module_logger.error(f"Failed to view conversation: {e}")
            messagebox.showerror("Error", f"Failed to view conversation:\n{e!s}")

    def _open_attachments_folder(self) -> None:
        """Open the Attachments folder for selected conversation."""
        selection = self.tree_convs.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a conversation")
            return

        try:
            values = self.tree_convs.item(selection[0])["values"]
            conv_id = values[0]

            if not self.settings.export_root:
                messagebox.showerror("Error", "Export root not set")
                return

            attachments_path = Path(self.settings.export_root) / conv_id / "Attachments"
            if not attachments_path.exists():
                messagebox.showinfo("No Attachments", f"No Attachments folder found for:\n{conv_id}")
                return

            self._open_path(attachments_path)
            module_logger.info(f"Opened attachments folder: {attachments_path}")

        except Exception as e:
            module_logger.error(f"Failed to open attachments folder: {e}")
            messagebox.showerror("Error", f"Failed to open attachments folder:\n{e!s}")

    def _open_conversation_folder(self) -> None:
        """Open the conversation folder in file explorer."""
        selection = self.tree_convs.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a conversation")
            return

        try:
            values = self.tree_convs.item(selection[0])["values"]
            conv_id = values[0]

            if not self.settings.export_root:
                messagebox.showerror("Error", "Export root not set")
                return

            conv_path = Path(self.settings.export_root) / conv_id
            if not conv_path.exists():
                messagebox.showerror("Not Found", f"Conversation folder not found:\n{conv_path}")
                return

            self._open_path(conv_path)
            module_logger.info(f"Opened conversation folder: {conv_path}")

        except Exception as e:
            module_logger.error(f"Failed to open conversation folder: {e}")
            messagebox.showerror("Error", f"Failed to open folder:\n{e!s}")

    def _open_path(self, path: Path) -> None:
        """Open a file or folder in the system's default application."""
        try:
            if platform.system() == "Windows":
                os.startfile(str(path))
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(path)], check=True)
            else:  # Linux and others
                subprocess.run(["xdg-open", str(path)], check=True)
        except Exception as e:
            module_logger.error(f"Failed to open path {path}: {e}")
            messagebox.showerror("Error", f"Failed to open:\n{e!s}")

    def _save_text_copy(self, content: str, default_name: str) -> None:
        """Save text content to a user-selected file."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=default_name
            )
            if filename:
                Path(filename).write_text(content, encoding="utf-8")
                messagebox.showinfo("Success", f"Saved to {filename}")
                module_logger.info(f"Saved text copy to {filename}")
        except Exception as e:
            module_logger.error(f"Failed to save text copy: {e}")
            messagebox.showerror("Save Error", f"Failed to save:\n{e!s}")

    def _export_conversation_list(self) -> None:
        """Export the conversation list to CSV."""
        items = self.tree_convs.get_children()
        if not items:
            messagebox.showwarning("No Data", "No conversations to export")
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            if not filename:
                return

            from pathlib import Path
            with Path(filename).open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Conversation ID", "Subject", "First Date", "Last Date", "Email Count"])
                for item_id in items:
                    values = self.tree_convs.item(item_id)["values"]
                    writer.writerow(values)

            messagebox.showinfo("Success", f"Exported {len(items)} conversations to {filename}")
            module_logger.info(f"Exported conversation list to {filename}")
        except Exception as e:
            module_logger.error(f"Failed to export conversation list: {e}")
            messagebox.showerror("Export Error", f"Failed to export:\n{e!s}")

    # ------------- Configuration Tab (NEW) -------------

    def _build_config_tab(self) -> None:
        frm = self.tab_config

        # Scrollable frame setup
        canvas = tk.Canvas(frm)
        scrollbar = ttk.Scrollbar(frm, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # GCP Settings
        gcp_frame = ttk.LabelFrame(scrollable_frame, text="GCP Settings", padding=10)
        gcp_frame.pack(fill=tk.X, padx=8, pady=8)

        g1 = ttk.Frame(gcp_frame)
        g1.pack(fill=tk.X, pady=2)
        ttk.Label(g1, text="GCP Project:").grid(row=0, column=0, sticky="w", padx=(0,4))
        self.var_gcp_project = tk.StringVar(value=os.getenv("GCP_PROJECT", ""))
        ttk.Entry(g1, width=40, textvariable=self.var_gcp_project).grid(row=0, column=1, sticky="ew", padx=4)

        ttk.Label(g1, text="GCP Region:").grid(row=1, column=0, sticky="w", padx=(0,4), pady=(4,0))
        self.var_gcp_region = tk.StringVar(value=os.getenv("GCP_REGION", "us-central1"))
        ttk.Entry(g1, width=40, textvariable=self.var_gcp_region).grid(row=1, column=1, sticky="ew", padx=4, pady=(4,0))

        ttk.Label(g1, text="Vertex Location:").grid(row=2, column=0, sticky="w", padx=(0,4), pady=(4,0))
        self.var_vertex_location = tk.StringVar(value=os.getenv("VERTEX_LOCATION", "us-central1"))
        ttk.Entry(g1, width=40, textvariable=self.var_vertex_location).grid(row=2, column=1, sticky="ew", padx=4, pady=(4,0))

        g1.columnconfigure(1, weight=1)

        # Indexing Settings
        idx_frame = ttk.LabelFrame(scrollable_frame, text="Indexing Settings", padding=10)
        idx_frame.pack(fill=tk.X, padx=8, pady=8)

        i1 = ttk.Frame(idx_frame)
        i1.pack(fill=tk.X, pady=2)
        self.var_cfg_chunk_size = tk.IntVar(value=int(os.getenv("CHUNK_SIZE", "1600")))
        LabeledSpinbox(i1, "Chunk Size:", 100, 5000, 1, 10, self.var_cfg_chunk_size).grid(row=0, column=0, sticky="w", padx=(0,4))

        self.var_cfg_chunk_overlap = tk.IntVar(value=int(os.getenv("CHUNK_OVERLAP", "200")))
        LabeledSpinbox(i1, "Chunk Overlap:", 0, 1000, 1, 10, self.var_cfg_chunk_overlap).grid(row=0, column=1, sticky="w", padx=(20,4))

        self.var_cfg_batch = tk.IntVar(value=int(os.getenv("EMBED_BATCH", "64")))
        LabeledSpinbox(i1, "Batch Size:", 1, 250, 1, 10, self.var_cfg_batch).grid(row=1, column=0, sticky="w", padx=(0,4), pady=(4,0))

        self.var_cfg_workers = tk.IntVar(value=int(os.getenv("NUM_WORKERS", "4")))
        LabeledSpinbox(i1, "Default Workers:", 1, 16, 1, 10, self.var_cfg_workers).grid(row=1, column=1, sticky="w", padx=(20,4), pady=(4,0))

        # Email Settings
        email_frame = ttk.LabelFrame(scrollable_frame, text="Email Settings", padding=10)
        email_frame.pack(fill=tk.X, padx=8, pady=8)

        e1 = ttk.Frame(email_frame)
        e1.pack(fill=tk.X, pady=2)
        ttk.Label(e1, text="Sender Name:").grid(row=0, column=0, sticky="w", padx=(0,4))
        self.var_sender_name = tk.StringVar(value=os.getenv("SENDER_LOCKED_NAME", ""))
        ttk.Entry(e1, width=40, textvariable=self.var_sender_name).grid(row=0, column=1, sticky="ew", padx=4)

        ttk.Label(e1, text="Sender Email:").grid(row=1, column=0, sticky="w", padx=(0,4), pady=(4,0))
        self.var_sender_email = tk.StringVar(value=os.getenv("SENDER_LOCKED_EMAIL", ""))
        ttk.Entry(e1, width=40, textvariable=self.var_sender_email).grid(row=1, column=1, sticky="ew", padx=4, pady=(4,0))

        ttk.Label(e1, text="Message ID Domain:").grid(row=2, column=0, sticky="w", padx=(0,4), pady=(4,0))
        self.var_msg_id_domain = tk.StringVar(value=os.getenv("MESSAGE_ID_DOMAIN", ""))
        ttk.Entry(e1, width=40, textvariable=self.var_msg_id_domain).grid(row=2, column=1, sticky="ew", padx=4, pady=(4,0))

        e1.columnconfigure(1, weight=1)

        # Control buttons
        btn_frame = ttk.Frame(scrollable_frame, padding=10)
        btn_frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(btn_frame, text="Apply Configuration", command=self._apply_config).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Reset to Defaults", command=self._reset_config).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="View Current Config", command=self._view_config).pack(side=tk.LEFT, padx=4)

        self.config_status = ttk.Label(btn_frame, text="", foreground=self.colors["info"])
        self.config_status.pack(side=tk.LEFT, padx=20)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    # ------------- System Diagnostics Tab (NEW) -------------

    def _build_diagnostics_tab(self) -> None:
        frm = self.tab_diagnostics

        # Control panel
        ctrl_frame = ttk.Frame(frm, padding=8)
        ctrl_frame.pack(fill=tk.X)

        self.btn_run_diagnostics = ttk.Button(ctrl_frame, text="Run Full Diagnostics", command=partial(self._run_diagnostics))
        self.btn_run_diagnostics.pack(side=tk.LEFT, padx=4)
        self.btn_check_deps = ttk.Button(ctrl_frame, text="Check Dependencies", command=partial(self._check_deps))
        self.btn_check_deps.pack(side=tk.LEFT, padx=4)
        self.btn_check_index = ttk.Button(ctrl_frame, text="Check Index Health", command=partial(self._check_index))
        self.btn_check_index.pack(side=tk.LEFT, padx=4)
        self.btn_test_embeddings = ttk.Button(ctrl_frame, text="Test Embeddings", command=partial(self._test_embeddings))
        self.btn_test_embeddings.pack(side=tk.LEFT, padx=4)
        self.pb_diagnostics = ttk.Progressbar(ctrl_frame, mode="indeterminate", length=200)
        self.pb_diagnostics.pack(side=tk.LEFT, padx=8)

        # Results display with colored status
        results_frame = ttk.LabelFrame(frm, text="Diagnostic Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.txt_diagnostics = tk.Text(results_frame, wrap="word", font=("Courier", 9))
        self.txt_diagnostics.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Configure tags for colored output
        self.txt_diagnostics.tag_config("success", foreground=self.colors["success"])
        self.txt_diagnostics.tag_config("warning", foreground=self.colors["warning"])
        self.txt_diagnostics.tag_config("error", foreground=self.colors["error"])
        self.txt_diagnostics.tag_config("info", foreground=self.colors["info"])

        yscroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.txt_diagnostics.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_diagnostics.configure(yscrollcommand=yscroll.set)

    # ------------- Chunking Tab (Conversation Pre-Processing) -------------

    def _build_chunking_tab(self) -> None:
        """Build tab for managing conversation chunking BEFORE indexing."""
        frm = self.tab_chunking

        # Info panel
        info_frame = ttk.LabelFrame(frm, text="ℹ️ About Chunking", padding=10)
        info_frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(info_frame, text="Chunking must be done BEFORE indexing. This step prepares conversations for embedding.",
                 font=("Arial", 10), foreground=self.colors["info"], wraplength=1100).pack()

        # Configuration panel
        config_frame = ttk.LabelFrame(frm, text="Chunking Parameters (from Config)", padding=10)
        config_frame.pack(fill=tk.X, padx=8, pady=8)

        r1 = ttk.Frame(config_frame)
        r1.pack(fill=tk.X, pady=2)
        ttk.Label(r1, text="Chunk Size:").pack(side=tk.LEFT)
        self.var_chunk_size = tk.IntVar(value=int(os.getenv("CHUNK_SIZE", "1600")))
        ttk.Spinbox(r1, from_=100, to=5000, width=10, textvariable=self.var_chunk_size, state="readonly").pack(side=tk.LEFT, padx=4)

        ttk.Label(r1, text="Overlap:").pack(side=tk.LEFT, padx=(20, 0))
        self.var_chunk_overlap = tk.IntVar(value=int(os.getenv("CHUNK_OVERLAP", "200")))
        ttk.Spinbox(r1, from_=0, to=1000, width=10, textvariable=self.var_chunk_overlap, state="readonly").pack(side=tk.LEFT, padx=4)

        ttk.Label(r1, text="(Set in Configuration tab)", font=("Arial", 8), foreground="#666").pack(side=tk.LEFT, padx=(10,0))

        # Action panel
        action_frame = ttk.LabelFrame(frm, text="Chunking Operations", padding=10)
        action_frame.pack(fill=tk.X, padx=8, pady=8)

        btn_row = ttk.Frame(action_frame)
        btn_row.pack(fill=tk.X)

        self.btn_force_rechunk = ttk.Button(btn_row, text="🔥 Force Re-chunk All", command=partial(self._on_force_rechunk), style='Primary.TButton')
        self.btn_force_rechunk.pack(side=tk.LEFT, padx=4)
        self.btn_incremental_chunk = ttk.Button(btn_row, text="🔄 Incremental Update", command=partial(self._on_incremental_chunk), style='Action.TButton')
        self.btn_incremental_chunk.pack(side=tk.LEFT, padx=4)
        self.btn_surgical_rechunk = ttk.Button(btn_row, text="🎯 Surgical Re-chunk", command=partial(self._on_surgical_rechunk))
        self.btn_surgical_rechunk.pack(side=tk.LEFT, padx=4)

        self.pb_chunk = ttk.Progressbar(btn_row, mode="determinate", length=300)
        self.pb_chunk.pack(side=tk.LEFT, padx=8)

        self.lbl_chunk_progress = ttk.Label(btn_row, text="")
        self.lbl_chunk_progress.pack(side=tk.LEFT)

        # Results panel
        results_frame = ttk.LabelFrame(frm, text="Chunked Conversations", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        cols = ("conv_id", "chunks", "status", "last_chunked")
        self.tree_chunks = ttk.Treeview(results_frame, columns=cols, show="headings", height=15)
        self.tree_chunks.heading("conv_id", text="Conversation ID")
        self.tree_chunks.heading("chunks", text="Chunk Count")
        self.tree_chunks.heading("status", text="Status")
        self.tree_chunks.heading("last_chunked", text="Last Chunked")
        self.tree_chunks.column("conv_id", width=300)
        self.tree_chunks.column("chunks", width=100, anchor=tk.CENTER)
        self.tree_chunks.column("status", width=150)
        self.tree_chunks.column("last_chunked", width=200)

        scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.tree_chunks.yview)
        self.tree_chunks.configure(yscrollcommand=scroll.set)
        self.tree_chunks.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Bottom actions
        bottom = ttk.Frame(frm, padding=8)
        bottom.pack(fill=tk.X)
        ttk.Button(bottom, text="📋 List Chunked Conversations", command=self._list_chunked_convs).pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom, text="🗑️ Clear Chunks Directory", command=self._clear_chunks_dir).pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom, text="🖥️ Console: Show Chunk Count", command=self.print_total_chunks).pack(side=tk.LEFT, padx=4)

    # ------------- Analyze Tab (Enhanced) -------------

    def _build_analyze_tab(self) -> None:
        frm = self.tab_analyze

        # Input panel
        input_frame = ttk.LabelFrame(frm, text="Conversation Selection", padding=8)
        input_frame.pack(fill=tk.X, padx=8, pady=8)

        row = ttk.Frame(input_frame)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Folder:").pack(side=tk.LEFT)
        self.var_thread_dir = tk.StringVar(value="")
        ttk.Entry(row, width=70, textvariable=self.var_thread_dir).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Browse…", command=self._choose_thread_dir).pack(side=tk.LEFT, padx=4)

        # Options panel
        opts_frame = ttk.LabelFrame(frm, text="Analysis Options", padding=8)
        opts_frame.pack(fill=tk.X, padx=8, pady=(0,8))

        o1 = ttk.Frame(opts_frame)
        o1.pack(fill=tk.X)

        ttk.Label(o1, text="Output Format:").pack(side=tk.LEFT)
        self.var_analysis_format = tk.StringVar(value="json")
        ttk.Radiobutton(o1, text="JSON", variable=self.var_analysis_format, value="json").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(o1, text="Markdown", variable=self.var_analysis_format, value="markdown").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(o1, text="Both", variable=self.var_analysis_format, value="both").pack(side=tk.LEFT, padx=4)

        self.var_export_csv = tk.BooleanVar(value=False)
        ttk.Checkbutton(o1, text="Export Actions to CSV", variable=self.var_export_csv).pack(side=tk.LEFT, padx=20)

        self.var_merge_manifest = tk.BooleanVar(value=True)
        ttk.Checkbutton(o1, text="Merge Manifest Data", variable=self.var_merge_manifest).pack(side=tk.LEFT, padx=4)

        # Action buttons with export
        action_frame = ttk.Frame(frm, padding=8)
        action_frame.pack(fill=tk.X, padx=8)

        self.btn_analyze = ttk.Button(action_frame, text="📊 Analyze Thread", command=self._on_analyze_thread, style='Primary.TButton')
        self.btn_analyze.pack(side=tk.LEFT)
        self.pb_analyze = ttk.Progressbar(action_frame, mode="indeterminate", length=200)
        self.pb_analyze.pack(side=tk.LEFT, padx=8)
        ttk.Button(action_frame, text="💾 Save Analysis", command=self._save_analysis_results).pack(side=tk.LEFT, padx=4)

        # Results display with tabs
        results_notebook = ttk.Notebook(frm)
        results_notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

        # Text tab
        text_frame = ttk.Frame(results_notebook)
        results_notebook.add(text_frame, text="Analysis Output")
        self.txt_analyze = tk.Text(text_frame, wrap="word", font=("Courier", 9))
        self.txt_analyze.pack(fill=tk.BOTH, expand=True)

        # Preview tab
        preview_frame = ttk.Frame(results_notebook)
        results_notebook.add(preview_frame, text="Quick Preview")
        self.txt_analyze_preview = tk.Text(preview_frame, wrap="word", font=("Arial", 10), state="disabled")
        self.txt_analyze_preview.pack(fill=tk.BOTH, expand=True)

    # ------------- Logs Tab (Enhanced) -------------

    def _build_log_tab(self) -> None:
        frm = self.tab_logs

        # Log level control
        ctrl_frame = ttk.Frame(frm, padding=8)
        ctrl_frame.pack(fill=tk.X)

        ttk.Label(ctrl_frame, text="Log Level:").pack(side=tk.LEFT)
        self.var_log_level = tk.StringVar(value="INFO")
        log_combo = ttk.Combobox(ctrl_frame, width=10, state="readonly", textvariable=self.var_log_level,
                                 values=["DEBUG", "INFO", "WARNING", "ERROR"])
        log_combo.pack(side=tk.LEFT, padx=4)
        log_combo.bind("<<ComboboxSelected>>", self._change_log_level)

        ttk.Button(ctrl_frame, text="Clear Logs", command=lambda: self.txt_logs.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=20)
        ttk.Button(ctrl_frame, text="Save Logs…", command=self._save_logs).pack(side=tk.LEFT, padx=4)

        # Log display
        log_frame = ttk.Frame(frm, padding=(8,0,8,8))
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.txt_logs = tk.Text(log_frame, wrap="word", font=("Courier", 9))
        self.txt_logs.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Configure tags for log levels
        self.txt_logs.tag_config("DEBUG", foreground="#888")
        self.txt_logs.tag_config("INFO", foreground="#000")
        self.txt_logs.tag_config("WARNING", foreground=self.colors["warning"])
        self.txt_logs.tag_config("ERROR", foreground=self.colors["error"])

        yscroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.txt_logs.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_logs.configure(yscrollcommand=yscroll.set)

    # ------------------------------- Utility Methods -------------------------------

    def _detect_worker_count(self) -> int:
        """Auto-detect optimal workers from credentials."""
        try:
            from emailops.llm_runtime import DEFAULT_ACCOUNTS, load_validated_accounts
            accounts = load_validated_accounts(default_accounts=DEFAULT_ACCOUNTS)
            return len(accounts)
        except Exception:
            return 1

    def _with_root_and_index(self) -> tuple[Path | None, Path | None]:
        """Validate and return root and index directories."""
        root = Path(self.var_root.get().strip()).expanduser()
        ok, msg = validate_directory_path(root, must_exist=True, allow_parent_traversal=False)
        if not ok:
            messagebox.showerror("Invalid root", msg)
            return None, None

        # Use proper index directory name from metadata module
        try:
            from emailops.index_metadata import INDEX_DIRNAME_DEFAULT
            ix_dir = root / INDEX_DIRNAME_DEFAULT
        except ImportError:
            ix_dir = root / "_index"

        # Validate index directory exists
        if not ix_dir.exists():
            messagebox.showerror("Index Not Found",
                               f"Index directory not found:\n{ix_dir}\n\nBuild index first.")
            return None, None

        return root, ix_dir

    def _set_status(self, msg: str, color: str = "info") -> None:
        """Update status label with color."""
        self.status_var.set(msg)
        self.status_label.config(foreground=self.colors.get(color, "#555"))

    # ------------------------------- File Operations -------------------------------

    def _choose_root(self) -> None:
        d = filedialog.askdirectory(title="Choose export root")
        if d:
            self.var_root.set(d)
            self.settings.export_root = d
            self.settings.save()

    def _choose_thread_dir(self) -> None:
        d = filedialog.askdirectory(title="Choose conversation folder (contains Conversation.txt)")
        if d:
            self.var_thread_dir.set(d)

    def _save_settings(self, event: tk.Event | None = None) -> None:
        try:
            self._sync_settings_from_ui()
            self.settings.save()
            messagebox.showinfo("Settings Saved", f"Successfully saved to:\n{SETTINGS_FILE}")
            module_logger.info("Settings saved successfully")
        except Exception as e:
            module_logger.error(f"Failed to save settings: {e}")
            messagebox.showerror("Save Error", f"Failed to save settings:\n{e!s}\n\nCheck logs for details.")

    def _load_settings(self, event: tk.Event | None = None) -> None:
        self.settings = AppSettings.load()
        self.var_root.set(self.settings.export_root)
        self.var_provider.set(self.settings.provider)
        self.var_persona.set(self.settings.persona)
        self.var_temp.set(self.settings.temperature)
        self.var_sim.set(self.settings.sim_threshold)
        self.var_k.set(self.settings.k)
        self.var_reply_tokens.set(self.settings.reply_tokens)
        self.var_fresh_tokens.set(self.settings.fresh_tokens)
        self.var_reply_policy.set(self.settings.reply_policy)
        self.var_mmr_lambda.set(self.settings.mmr_lambda)
        self.var_rerank_alpha.set(self.settings.rerank_alpha)
        self.var_session_id.set(self.settings.chat_session_id)
        self.var_max_history.set(self.settings.max_chat_history)
        self.var_to.set(self.settings.last_to)
        self.var_cc.set(self.settings.last_cc)
        self.var_subject.set(self.settings.last_subject)
        messagebox.showinfo("Settings", f"Loaded from {SETTINGS_FILE}")

    # ------------------------------- Missing Method Implementations -------------------------------

    def _sync_settings_from_ui(self) -> None:
        """Sync all settings from UI controls to internal state."""
        try:
            # Basic settings
            self.settings.export_root = self.var_root.get().strip()
            self.settings.provider = self.var_provider.get().strip()
            self.settings.persona = self.var_persona.get().strip()
            self.settings.temperature = float(self.var_temp.get())

            # Search settings
            self.settings.k = int(self.var_k.get())
            self.settings.sim_threshold = float(self.var_sim.get())
            self.settings.mmr_lambda = float(self.var_mmr_lambda.get())
            self.settings.rerank_alpha = float(self.var_rerank_alpha.get())

            # Reply/Fresh settings
            self.settings.reply_tokens = int(self.var_reply_tokens.get())
            self.settings.fresh_tokens = int(self.var_fresh_tokens.get())
            self.settings.reply_policy = self.var_reply_policy.get().strip()

            # Chat settings
            self.settings.chat_session_id = self.var_session_id.get().strip()
            self.settings.max_chat_history = int(self.var_max_history.get())

            # Save last used values
            self.settings.last_to = self.var_to.get().strip()
            self.settings.last_cc = self.var_cc.get().strip()
            self.settings.last_subject = self.var_subject.get().strip()

            module_logger.info("✓ Settings synchronized from UI")
            module_logger.debug(f"Export root: {self.settings.export_root}, Temp: {self.settings.temperature}, K: {self.settings.k}")
        except Exception as e:
            module_logger.error(f"✗ Failed to sync settings: {e}", exc_info=True)
            messagebox.showerror("Settings Error", f"Failed to sync settings: {e!s}")
            raise

    @run_with_progress("search", "pb_search", "status_label", "btn_search")
    def _on_search(self, *, update_progress) -> None:
        """Handle search with advanced filters integration."""
        query = self.var_search_q.get().strip()
        if not query:
            self.after(0, lambda: messagebox.showwarning("Input Required", "Please enter a search query"))
            return

        self._sync_settings_from_ui()
        update_progress(0, 1, "Searching...")

        # Collect advanced filter values if shown
        filters_dict = {}
        if self.show_advanced.get():
            if self.var_from_filter.get():
                filters_dict["from_emails"] = {e.strip().lower() for e in self.var_from_filter.get().split(",") if e.strip()}
            if self.var_to_filter.get():
                filters_dict["to_emails"] = {e.strip().lower() for e in self.var_to_filter.get().split(",") if e.strip()}
            if self.var_subject_filter.get():
                filters_dict["subject_contains"] = [self.var_subject_filter.get().strip().lower()]

        from emailops.search_and_draft import SearchFilters
        # Import _search directly
        try:
            from emailops.search_and_draft import _search
        except ImportError:
            # Fallback if _search is not directly available
            from emailops import processor
            _search = getattr(processor, '_search', None)
            if _search is None:
                messagebox.showerror("Import Error", "Search function not available")
                return

        # Build SearchFilters object
        search_filters = None
        if filters_dict:
            search_filters = SearchFilters(**filters_dict)

        ix_dir = Path(self.settings.export_root) / os.getenv("INDEX_DIRNAME", "_index")

        results = _search(
            ix_dir=ix_dir,
            query=query,
            k=self.settings.k,
            provider=self.settings.provider,
            filters=search_filters,
            mmr_lambda=self.settings.mmr_lambda,
            rerank_alpha=self.settings.rerank_alpha
        )

        self.search_results = results

        def update_ui():
            # Update tree view
            self.tree.delete(*self.tree.get_children())
            for i, result in enumerate(results):
                score = result.get("score", 0)
                subject = result.get("subject", "No Subject")
                doc_id = result.get("id", f"doc_{i}")
                self.tree.insert("", "end", values=(f"{score:.3f}", subject, doc_id))

            self._set_status(f"Found {len(results)} results", "success")
            update_progress(1, 1, f"Found {len(results)} results")

        self.after(0, update_ui)

    def _show_snippet(self, event) -> None:
        """Display snippet of selected search result."""
        selection = self.tree.selection()
        if not selection or not self.search_results:
            return

        try:
            idx = self.tree.index(selection[0])
            if 0 <= idx < len(self.search_results):
                result = self.search_results[idx]
                text = result.get("text", "No text available")
                self.txt_snip.delete("1.0", tk.END)
                self.txt_snip.insert("1.0", text[:2000])
        except Exception as e:
            module_logger.error(f"Failed to show snippet: {e}")

    @run_with_progress("draft_reply", "pb_reply", "status_label", "btn_draft_reply")
    def _on_draft_reply(self, *, update_progress) -> None:
        """Generate reply email with all options."""
        conv_id = self.var_conv.get().strip()
        if not conv_id:
            self.after(0, lambda: messagebox.showwarning("Input Required", "Please select a conversation ID"))
            return

        self._sync_settings_from_ui()
        update_progress(0, 1, "Drafting reply...")

        from emailops.search_and_draft import draft_email_reply_eml

        result = draft_email_reply_eml(
            export_root=Path(self.settings.export_root),
            conv_id=conv_id,
            provider=self.settings.provider,
            query=self.var_reply_q.get() or None,
            sim_threshold=self.settings.sim_threshold,
            target_tokens=self.settings.reply_tokens,
            temperature=self.settings.temperature,
            include_attachments=self.var_reply_attach.get(),
            reply_policy=self.settings.reply_policy
        )

        # Store for saving
        self._last_reply_bytes = result["eml_bytes"]
        self._last_reply_meta = {
            "to": result.get("to", []),
            "cc": result.get("cc", []),
            "subject": result.get("subject", ""),
            "conv_id": conv_id
        }

        def update_ui():
            # Display draft
            draft_json = result.get("draft_json", {})
            final_draft = draft_json.get("final_draft", {})
            email_text = final_draft.get("email_draft", "")

            self.txt_reply.delete("1.0", tk.END)
            self.txt_reply.insert("1.0", f"To: {', '.join(result.get('to', []))}\n")
            self.txt_reply.insert(tk.END, f"Cc: {', '.join(result.get('cc', []))}\n")
            self.txt_reply.insert(tk.END, f"Subject: {result.get('subject', '')}\n\n")
            self.txt_reply.insert(tk.END, f"{email_text}\n\n")

            # Show metadata
            confidence = draft_json.get("confidence_score", 0)
            citations = final_draft.get("citations", [])
            self.txt_reply.insert(tk.END, "\n--- Metadata ---\n")
            self.txt_reply.insert(tk.END, f"Confidence: {confidence:.2%}\n")
            self.txt_reply.insert(tk.END, f"Citations: {len(citations)}\n")

            self._set_status(f"Reply generated (confidence: {confidence:.0%})", "success")
            update_progress(1, 1, f"Reply generated (confidence: {confidence:.0%})")

        self.after(0, update_ui)

    def _save_eml_reply(self) -> None:
        """Save generated reply as .eml file."""
        if not self._last_reply_bytes or not self._last_reply_meta:
            messagebox.showwarning("No Draft", "Generate a reply first")
            return

        try:
            conv_id = self._last_reply_meta.get("conv_id", "reply")
            filename = filedialog.asksaveasfilename(
                defaultextension=".eml",
                filetypes=[("Email files", "*.eml"), ("All files", "*.*")],
                initialfile=f"{conv_id}_reply.eml"
            )
            if filename:
                Path(filename).write_bytes(self._last_reply_bytes)
                messagebox.showinfo("Success", f"Saved to {filename}")
                module_logger.info(f"Reply saved to {filename}")
        except Exception as e:
            module_logger.error(f"Failed to save reply: {e}")
            messagebox.showerror("Save Error", f"Failed to save:\n{e!s}")

    @run_with_progress("draft_fresh", "pb_fresh", "status_label", "btn_draft_fresh")
    def _on_draft_fresh(self, *, update_progress) -> None:
        """Generate fresh email with all options."""
        to_list = [x.strip() for x in self.var_to.get().split(",") if x.strip()]
        if not to_list:
            self.after(0, lambda: messagebox.showwarning("Input Required", "Please enter To recipients"))
            return

        subject = self.var_subject.get().strip()
        if not subject:
            self.after(0, lambda: messagebox.showwarning("Input Required", "Please enter a subject"))
            return

        query = self.var_fresh_q.get().strip()
        if not query:
            self.after(0, lambda: messagebox.showwarning("Input Required", "Please enter intent/instructions"))
            return

        self._sync_settings_from_ui()
        cc_list = [x.strip() for x in self.var_cc.get().split(",") if x.strip()]
        update_progress(0, 1, "Drafting fresh email...")

        from emailops.search_and_draft import draft_fresh_email_eml

        result = draft_fresh_email_eml(
            export_root=Path(self.settings.export_root),
            provider=self.settings.provider,
            to_list=to_list,
            cc_list=cc_list,
            subject=subject,
            query=query,
            sim_threshold=self.settings.sim_threshold,
            target_tokens=self.settings.fresh_tokens,
            temperature=self.settings.temperature,
            include_attachments=self.var_fresh_attach.get()
        )

        # Store for saving
        self._last_fresh_bytes = result["eml_bytes"]
        self._last_fresh_meta = {
            "to": to_list,
            "cc": cc_list,
            "subject": subject
        }

        def update_ui():
            # Display draft
            draft_json = result.get("draft_json", {})
            final_draft = draft_json.get("final_draft", {})
            email_text = final_draft.get("email_draft", "")

            self.txt_fresh.delete("1.0", tk.END)
            self.txt_fresh.insert("1.0", f"To: {', '.join(to_list)}\n")
            self.txt_fresh.insert(tk.END, f"Cc: {', '.join(cc_list)}\n")
            self.txt_fresh.insert(tk.END, f"Subject: {subject}\n\n")
            self.txt_fresh.insert(tk.END, f"{email_text}\n\n")

            # Show metadata
            confidence = draft_json.get("confidence_score", 0)
            citations = final_draft.get("citations", [])
            self.txt_fresh.insert(tk.END, "\n--- Metadata ---\n")
            self.txt_fresh.insert(tk.END, f"Confidence: {confidence:.2%}\n")
            self.txt_fresh.insert(tk.END, f"Citations: {len(citations)}\n")

            self._set_status(f"Fresh email generated (confidence: {confidence:.0%})", "success")
            update_progress(1, 1, f"Fresh email generated (confidence: {confidence:.0%})")

        self.after(0, update_ui)

    def _save_eml_fresh(self) -> None:
        """Save generated fresh email as .eml file."""
        if not self._last_fresh_bytes or not self._last_fresh_meta:
            messagebox.showwarning("No Draft", "Generate a fresh email first")
            return

        try:
            subject = self._last_fresh_meta.get("subject", "fresh")
            filename = filedialog.asksaveasfilename(
                defaultextension=".eml",
                filetypes=[("Email files", "*.eml"), ("All files", "*.*")],
                initialfile=f"{subject[:30]}_fresh.eml"
            )
            if filename:
                Path(filename).write_bytes(self._last_fresh_bytes)
                messagebox.showinfo("Success", f"Saved to {filename}")
                module_logger.info(f"Fresh email saved to {filename}")
        except Exception as e:
            module_logger.error(f"Failed to save fresh email: {e}")
            messagebox.showerror("Save Error", f"Failed to save:\n{e!s}")

    @run_with_progress("chat", "pb_chat", "status_label", "btn_chat")
    def _on_chat(self, *, update_progress) -> None:
        """Handle chat with session management."""
        query = self.var_chat_q.get().strip()
        if not query:
            self.after(0, lambda: messagebox.showwarning("Input Required", "Please enter a question"))
            return

        self._sync_settings_from_ui()
        update_progress(0, 1, "Processing chat...")

        from emailops.search_and_draft import (
            ChatSession,
            _search,
            chat_with_context,
        )

        # Get or create session
        ix_dir = Path(self.settings.export_root) / os.getenv("INDEX_DIRNAME", "_index")
        session = ChatSession(
            base_dir=ix_dir,
            session_id=self.settings.chat_session_id,
            max_history=self.settings.max_chat_history
        )
        session.load()

        # Search for context
        ctx = _search(
            ix_dir=ix_dir,
            query=query,
            k=self.var_chat_k.get(),
            provider=self.settings.provider
        )

        # Chat with context
        chat_hist = session.recent() if session else []
        answer = chat_with_context(
            query=query,
            context_snippets=ctx,
            chat_history=chat_hist,
            temperature=self.settings.temperature
        )

        # Update session
        session.add_message("user", query)
        session.add_message("assistant", answer.get("answer", ""))
        session.save()

        def update_ui():
            # Display
            self.txt_chat.insert(tk.END, f"\n{'='*80}\n")
            self.txt_chat.insert(tk.END, f"User: {query}\n\n")
            self.txt_chat.insert(tk.END, f"Assistant: {answer.get('answer', '')}\n")

            citations = answer.get("citations", [])
            if citations:
                self.txt_chat.insert(tk.END, f"\nCitations ({len(citations)}):\n")
                for c in citations[:5]:
                    doc_id = c.get("document_id", "")
                    self.txt_chat.insert(tk.END, f"  - {doc_id}\n")

            self.txt_chat.see(tk.END)
            self._set_status(f"Chat complete ({len(ctx)} context snippets)", "success")
            update_progress(1, 1, f"Chat complete ({len(ctx)} context snippets)")

        self.after(0, update_ui)

    def _load_chat_session(self) -> None:
        """Load a chat session."""
        try:
            from emailops.search_and_draft import ChatSession

            ix_dir = Path(self.settings.export_root) / os.getenv("INDEX_DIRNAME", "_index")
            session = ChatSession(
                base_dir=ix_dir,
                session_id=self.var_session_id.get(),
                max_history=self.var_max_history.get()
            )
            session.load()

            self.txt_chat.delete("1.0", tk.END)
            self.txt_chat.insert("1.0", f"Loaded session: {session.session_id}\n")
            self.txt_chat.insert(tk.END, f"History: {len(session.messages)} messages\n\n")

            for msg in session.messages:
                self.txt_chat.insert(tk.END, f"[{msg.role}] {msg.content}\n\n")

            messagebox.showinfo("Success", f"Loaded session: {session.session_id}")
        except Exception as e:
            module_logger.error(f"Failed to load session: {e}")
            messagebox.showerror("Load Error", f"Failed to load session:\n{e!s}")

    def _save_chat_session(self) -> None:
        """Save current chat session."""
        try:
            from emailops.search_and_draft import ChatSession

            ix_dir = Path(self.settings.export_root) / os.getenv("INDEX_DIRNAME", "_index")
            session = ChatSession(
                base_dir=ix_dir,
                session_id=self.var_session_id.get(),
                max_history=self.var_max_history.get()
            )
            session.save()

            messagebox.showinfo("Success", f"Session saved: {session.session_id}")
        except Exception as e:
            module_logger.error(f"Failed to save session: {e}")
            messagebox.showerror("Save Error", f"Failed to save session:\n{e!s}")

    def _reset_chat_session(self) -> None:
        """Reset current chat session."""
        try:
            from emailops.search_and_draft import ChatSession

            ix_dir = Path(self.settings.export_root) / os.getenv("INDEX_DIRNAME", "_index")
            session = ChatSession(
                base_dir=ix_dir,
                session_id=self.var_session_id.get(),
                max_history=self.var_max_history.get()
            )
            session.reset()

            self.txt_chat.delete("1.0", tk.END)
            self.txt_chat.insert("1.0", f"Session reset: {session.session_id}\n")

            messagebox.showinfo("Success", "Chat session reset")
        except Exception as e:
            module_logger.error(f"Failed to reset session: {e}")
            messagebox.showerror("Reset Error", f"Failed to reset session:\n{e!s}")

    def _load_conversations(self) -> None:
        """Load list of conversations for selection."""
        if not self.settings.export_root:
            messagebox.showwarning("No Root", "Please select export root first")
            return

        def run_load():
            try:
                self._set_status("Loading conversations...", "info")

                from emailops.search_and_draft import list_conversations_newest_first

                ix_dir = Path(self.settings.export_root) / os.getenv("INDEX_DIRNAME", "_index")
                convs = list_conversations_newest_first(ix_dir)

                # Update combobox
                conv_ids = [c["conv_id"] for c in convs]
                self.cmb_conv.config(values=conv_ids)

                messagebox.showinfo("Success", f"Loaded {len(convs)} conversations")
                self._set_status(f"Loaded {len(convs)} conversations", "success")

            except Exception as e:
                module_logger.error(f"Failed to load conversations: {e}")
                messagebox.showerror("Load Error", f"Failed to load conversations:\n{e!s}")
                self._set_status(f"Load failed: {e!s}", "error")

        threading.Thread(target=run_load, daemon=True).start()

    @run_with_progress("list_convs", "pb_convs", "status_label", "btn_list_convs")
    def _on_list_convs(self, *, update_progress) -> None:
        """List all conversations in tree view."""
        if not self.settings.export_root:
            self.after(0, lambda: messagebox.showwarning("No Root", "Please select export root first"))
            return

        update_progress(0, 1, "Listing conversations...")
        from emailops.search_and_draft import list_conversations_newest_first

        ix_dir = Path(self.settings.export_root) / os.getenv("INDEX_DIRNAME", "_index")
        mapping_path = ix_dir / "mapping.json"
        if not ix_dir.exists():
            self.after(0, lambda: messagebox.showerror("Index Missing", f"Index directory not found:\n{ix_dir}\n\nRun indexing first."))
            self._set_status("Index directory missing", "error")
            update_progress(1, 1, "Index directory missing")
            return
        if not mapping_path.exists():
            self.after(0, lambda: messagebox.showerror("Mapping Missing", f"mapping.json not found in index directory:\n{mapping_path}\n\nRun indexing first."))
            self._set_status("mapping.json missing", "error")
            update_progress(1, 1, "mapping.json missing")
            return

        convs = list_conversations_newest_first(ix_dir)

        def update_ui():
            # Update tree
            self.tree_convs.delete(*self.tree_convs.get_children())
            for conv in convs:
                self.tree_convs.insert("", "end", values=(
                    conv["conv_id"],
                    conv.get("subject", ""),
                    conv.get("first_date_str", ""),
                    conv.get("last_date_str", ""),
                    conv.get("count", 0)
                ))

            self._set_status(f"Listed {len(convs)} conversations", "success")
            update_progress(1, 1, f"Listed {len(convs)} conversations")

        self.after(0, update_ui)

    def _use_selected_conv(self) -> None:
        """Use selected conversation for reply."""
        selection = self.tree_convs.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a conversation")
            return

        try:
            values = self.tree_convs.item(selection[0])["values"]
            conv_id = values[0]
            self.var_conv.set(conv_id)
            self.nb.select(self.tab_reply)
            messagebox.showinfo("Success", f"Selected conversation: {conv_id}")
        except Exception as e:
            module_logger.error(f"Failed to use selected conversation: {e}")
            messagebox.showerror("Error", f"Failed: {e!s}")

    # ------------- Batch Operations Tab (NEW) -------------

    def _build_batch_tab(self) -> None:
        frm = self.tab_batch

        # Controls Frame
        controls_frame = ttk.LabelFrame(frm, text="Batch Configuration", padding=10)
        controls_frame.pack(fill=tk.X, padx=8, pady=8)

        # Batch controls
        batch_controls = ttk.Frame(controls_frame)
        batch_controls.pack(fill=tk.X)

        ttk.Button(batch_controls, text="Add Selected Conversations", command=self._add_selected_to_batch).pack(side=tk.LEFT, padx=4)
        ttk.Button(batch_controls, text="Clear Batch List", command=self._clear_batch_list).pack(side=tk.LEFT, padx=4)

        # Batch actions
        action_frame = ttk.Frame(controls_frame)
        action_frame.pack(fill=tk.X, pady=(10,0))

        self.btn_batch_summarize = ttk.Button(action_frame, text="Batch Summarize", command=partial(self._on_batch_summarize))
        self.btn_batch_summarize.pack(side=tk.LEFT, padx=4)

        self.btn_batch_reply = ttk.Button(action_frame, text="Batch Generate Replies", command=partial(self._on_batch_replies))
        self.btn_batch_reply.pack(side=tk.LEFT, padx=4)

        self.pb_batch = ttk.Progressbar(action_frame, mode="determinate", length=300)
        self.pb_batch.pack(side=tk.LEFT, padx=8)

        self.lbl_batch_progress = ttk.Label(action_frame, text="")
        self.lbl_batch_progress.pack(side=tk.LEFT)

        # Batch list
        list_frame = ttk.LabelFrame(frm, text="Conversations for Batch Processing", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        cols = ("conv_id", "subject")
        self.tree_batch = ttk.Treeview(list_frame, columns=cols, show="headings", height=15)
        self.tree_batch.heading("conv_id", text="Conversation ID")
        self.tree_batch.heading("subject", text="Subject")
        self.tree_batch.column("conv_id", width=250)
        self.tree_batch.column("subject", width=600)
        self.tree_batch.pack(fill=tk.BOTH, expand=True)

    def _add_selected_to_batch(self) -> None:
        """Add selected conversations from the main list to the batch list."""
        selection = self.tree_convs.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select conversations from the 'Conversations' tab first.")
            return

        # Build set of existing batch conv_ids (first column only)
        current_batch_ids = {self.tree_batch.item(i)["values"][0] for i in self.tree_batch.get_children()}

        added_count = 0
        for item_id in selection:
            values = self.tree_convs.item(item_id)["values"]
            if not values or not isinstance(values, (list, tuple)):
                continue
            conv_id = values[0]  # First column is conv_id
            if conv_id not in current_batch_ids:
                # Add only conv_id and subject to batch tree
                batch_values = (conv_id, values[1] if len(values) > 1 else "")
                self.tree_batch.insert("", "end", values=batch_values)
                current_batch_ids.add(conv_id)
                added_count += 1

        if added_count > 0:
            self._set_status(f"Added {added_count} conversations to batch", "success")
            self.nb.select(self.tab_batch)
        else:
            messagebox.showinfo("Info", "Selected conversations are already in the batch list.")

    def _clear_batch_list(self) -> None:
        """Clear the batch list."""
        self.tree_batch.delete(*self.tree_batch.get_children())

    def _run_batch_operation(
        self,
        operation_name: str,
        process_func: Callable[..., None],
        update_progress: Callable,
        output_dir_prompt: str | None = None,
    ) -> None:
        """Generic executor for batch operations."""
        items = self.tree_batch.get_children()
        if not items:
            messagebox.showwarning("No Batch Items", "Please add conversations to the batch list.")
            return

        conv_ids = [self.tree_batch.item(item_id)["values"][0] for item_id in items if self.tree_batch.item(item_id)["values"]]
        if not conv_ids:
            messagebox.showwarning("No Valid Items", "No valid conversation IDs in batch list.")
            return

        output_dir = None
        if output_dir_prompt:
            output_dir = filedialog.askdirectory(title=output_dir_prompt)
            if not output_dir:
                return

        total = len(conv_ids)
        completed, failed = 0, 0
        failed_conversations = []

        for i, conv_id in enumerate(conv_ids):
            if self.task.cancelled():
                break
            try:
                update_progress(i, total, f"{operation_name} {i+1}/{total}: {conv_id}...")
                process_func(conv_id, output_dir=output_dir)
                completed += 1
                module_logger.info(f"Successfully processed {conv_id} for {operation_name}")
            except Exception as e:
                failed += 1
                failed_conversations.append((conv_id, str(e)))
                module_logger.error(f"Failed to process {conv_id} for {operation_name}: {e}")

        final_msg = f"Batch {operation_name} complete: {completed} succeeded, {failed} failed"
        update_progress(total, total, final_msg)
        self._set_status(final_msg, "success" if failed == 0 else "warning")

        if failed_conversations:
            details = "\n".join([f"• {name}: {err[:100]}" for name, err in failed_conversations])
            messagebox.showwarning(
                f"Batch {operation_name} Results",
                f"Completed: {completed} succeeded, {failed} failed\n\nFailed items:\n{details}\n\nCheck logs for details."
            )

    @run_with_progress("batch_summarize", "pb_batch", "lbl_batch_progress", "btn_batch_summarize", "btn_batch_reply")
    def _on_batch_summarize(self, *, update_progress) -> None:
        """Handle batch summarization of conversations."""
        def process_summarize(conv_id, **kwargs):
            conv_dir = Path(self.settings.export_root) / conv_id
            os.environ["EMBED_PROVIDER"] = self.settings.provider
            asyncio.run(summarizer.analyze_conversation_dir(
                thread_dir=conv_dir,
                temperature=self.settings.temperature
            ))
        self._run_batch_operation("summarize", process_summarize, update_progress)

    @run_with_progress("batch_reply", "pb_batch", "lbl_batch_progress", "btn_batch_summarize", "btn_batch_reply")
    def _on_batch_replies(self, *, update_progress) -> None:
        """Handle batch generation of replies."""
        def process_reply(conv_id, output_dir, **kwargs):
            from emailops.search_and_draft import draft_email_reply_eml
            result = draft_email_reply_eml(
                export_root=Path(self.settings.export_root),
                conv_id=conv_id,
                provider=self.settings.provider,
                sim_threshold=self.settings.sim_threshold,
                target_tokens=self.settings.reply_tokens,
                temperature=self.settings.temperature,
                reply_policy=self.settings.reply_policy
            )
            out_path = Path(output_dir) / f"{conv_id}_reply.eml"
            out_path.write_bytes(result["eml_bytes"])
        self._run_batch_operation("reply", process_reply, update_progress, output_dir_prompt="Select Directory to Save Batch Replies")

    def _export_batch_results(self) -> None:
        """Export the list of conversations in the batch view."""
        items = self.tree_batch.get_children()
        if not items:
            messagebox.showwarning("No Data", "The batch list is empty.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="batch_conversations.csv"
        )
        if not filename:
            return

        try:
            from pathlib import Path
            with Path(filename).open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Conversation ID", "Subject"])
                for item_id in items:
                    values = self.tree_batch.item(item_id)["values"]
                    writer.writerow(values)
            messagebox.showinfo("Success", f"Exported {len(items)} items to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {e}")

    @run_with_progress("build_index", "pb_index", "lbl_index_progress", "btn_build")
    def _on_build_index(self, *, update_progress: Callable) -> None:
        """Build or update the index in a non-blocking way."""
        if not self.settings.export_root:
            self.after(0, lambda: messagebox.showwarning("No Root", "Please select export root first"))
            return

        self._sync_settings_from_ui()
        update_progress(0, 1, "Starting index build...")

        args = [
            sys.executable, "-m", "emailops.email_indexer",
            "--root", str(self.settings.export_root),
            "--provider", self.settings.provider,
            "--batch", str(self.var_batch.get()),
            "--workers", str(self.var_workers.get()),
        ]
        if self.var_force.get():
            args.append("--force-reindex")
        if self.var_limit.get() > 0:
            args.extend(["--limit", str(self.var_limit.get())])

        def run_in_thread():
            try:
                process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
                
                # Monitor stdout for progress
                if process.stdout:
                    for line in iter(process.stdout.readline, ''):
                        # This is a placeholder for real progress parsing.
                        # A more robust solution would parse structured output from the indexer.
                        self.after(0, lambda l=line: self.lbl_index_progress.config(text=l.strip()))
                
                process.wait(timeout=3600) # 1-hour timeout

                if process.returncode == 0:
                    self.after(0, lambda: self._set_status("Index build complete", "success"))
                    self.after(0, lambda: messagebox.showinfo("Success", "Index built successfully"))
                    update_progress(1, 1, "Index build complete")
                else:
                    stderr = process.stderr.read() if process.stderr else ""
                    self.after(0, lambda: self._set_status("Index build failed", "error"))
                    self.after(0, lambda: messagebox.showerror("Error", f"Index build failed:\n{stderr[:500]}"))
                    update_progress(1, 1, "Index build failed")

            except subprocess.TimeoutExpired:
                self.after(0, lambda: self._set_status("Index build timed out", "error"))
                self.after(0, lambda: messagebox.showerror("Timeout", "Index build timed out after 1 hour"))
                update_progress(1, 1, "Index build timed out")
            except Exception as e:
                module_logger.error(f"Index build failed: {e}", exc_info=True)
                self.after(0, lambda err=e: self._set_status(f"Index build failed: {err!s}", "error"))
                self.after(0, lambda err=e: messagebox.showerror("Error", f"Index build failed:\n{err!s}"))
                update_progress(1, 1, "Index build failed")

        threading.Thread(target=run_in_thread, daemon=True).start()

    def _apply_config(self) -> None:
        """Apply configuration changes and save to GUI settings."""
        try:
            # Update settings from UI
            self.settings.gcp_project = self.var_gcp_project.get().strip()
            self.settings.gcp_region = self.var_gcp_region.get().strip()
            self.settings.vertex_location = self.var_vertex_location.get().strip()
            self.settings.chunk_size = int(self.var_cfg_chunk_size.get())
            self.settings.chunk_overlap = int(self.var_cfg_chunk_overlap.get())
            self.settings.batch_size = int(self.var_cfg_batch.get())
            self.settings.num_workers = int(self.var_cfg_workers.get())
            self.settings.sender_locked_name = self.var_sender_name.get().strip()
            self.settings.sender_locked_email = self.var_sender_email.get().strip()
            self.settings.message_id_domain = self.var_msg_id_domain.get().strip()

            # Update environment variables
            os.environ["GCP_PROJECT"] = self.settings.gcp_project
            os.environ["GCP_REGION"] = self.settings.gcp_region
            os.environ["VERTEX_LOCATION"] = self.settings.vertex_location
            os.environ["CHUNK_SIZE"] = str(self.settings.chunk_size)
            os.environ["CHUNK_OVERLAP"] = str(self.settings.chunk_overlap)
            os.environ["EMBED_BATCH"] = str(self.settings.batch_size)
            os.environ["NUM_WORKERS"] = str(self.settings.num_workers)
            os.environ["SENDER_LOCKED_NAME"] = self.settings.sender_locked_name
            os.environ["SENDER_LOCKED_EMAIL"] = self.settings.sender_locked_email
            os.environ["MESSAGE_ID_DOMAIN"] = self.settings.message_id_domain

            # Save settings and sync other UI elements
            self._sync_settings_from_ui()
            self.settings.save()

            # Update other UI variables for consistency
            self.var_chunk_size.set(self.settings.chunk_size)
            self.var_chunk_overlap.set(self.settings.chunk_overlap)
            self.var_workers.set(self.settings.num_workers)
            self.var_batch.set(self.settings.batch_size)

            self.config_status.config(text="✓ Configuration applied and saved", foreground=self.colors["success"])
            messagebox.showinfo("Success",
                "Configuration applied successfully.\n\n"
                "Environment variables updated.\n"
                f"Settings saved to:\n{SETTINGS_FILE}")
            module_logger.info("✓ Configuration applied and settings saved")
        except Exception as e:
            module_logger.error(f"✗ Failed to apply configuration: {e}", exc_info=True)
            self.config_status.config(text="✗ Failed to apply", foreground=self.colors["error"])
            messagebox.showerror("Error", f"Failed to apply configuration:\n{e!s}")

    def _view_config(self) -> None:
        """View current configuration."""
        try:
            cfg = get_config()
            config_dict = cfg.to_dict()

            # Create dialog
            dialog = tk.Toplevel(self)
            dialog.title("Current Configuration")
            dialog.geometry("600x400")

            text = tk.Text(dialog, wrap="word", font=("Courier", 9))
            text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            text.insert("1.0", json.dumps(config_dict, indent=2))
            text.config(state="disabled")

            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)

        except Exception as e:
            module_logger.error(f"Failed to view configuration: {e}")
            messagebox.showerror("Error", f"Failed to view configuration:\n{e!s}")

    @run_with_progress("diagnostics", "pb_diagnostics", "status_label", "btn_run_diagnostics", "btn_check_deps", "btn_check_index", "btn_test_embeddings")
    def _run_diagnostics(self, *, update_progress) -> None:
        """Run full system diagnostics."""
        def update_ui(message, tag):
            self.after(0, lambda: self.txt_diagnostics.insert(tk.END, message, tag))

        self.after(0, lambda: self.txt_diagnostics.delete("1.0", tk.END))
        update_ui("Running full diagnostics...\n\n", "info")

        # Check dependencies
        report = doctor.check_and_install_dependencies(
            provider=self.settings.provider,
            auto_install=False,
            pip_timeout=300
        )

        update_ui("=== Dependencies ===\n", "info")
        if report.missing_critical:
            update_ui(f"✗ Missing critical: {', '.join(report.missing_critical)}\n", "error")
        else:
            update_ui("✓ All critical dependencies installed\n", "success")

        if report.missing_optional:
            update_ui(f"⚠ Missing optional: {', '.join(report.missing_optional)}\n", "warning")

        update_ui(f"✓ Installed: {len(report.installed)} packages\n\n", "success")

        # Check index
        ix_dir = Path(self.settings.export_root) / os.getenv("INDEX_DIRNAME", "_index")
        if ix_dir.exists():
            stats = doctor._get_index_statistics(ix_dir)
            update_ui("=== Index Health ===\n", "info")
            update_ui(f"✓ Index exists at {ix_dir}\n", "success")
            update_ui(f"  Documents: {stats.get('num_documents', 0)}\n", "info")
            update_ui(f"  Conversations: {stats.get('num_conversations', 0)}\n", "info")
            update_ui(f"  Total chars: {stats.get('total_chars', 0):,}\n\n", "info")
        else:
            update_ui("✗ Index not found\n\n", "error")

        # Test embeddings
        success, dim = doctor._probe_embeddings(self.settings.provider)
        update_ui("=== Embeddings ===\n", "info")
        if success:
            update_ui(f"✓ Embeddings working (dimension: {dim})\n", "success")
        else:
            update_ui("✗ Embeddings test failed\n", "error")

        self.after(0, lambda: self._set_status("Diagnostics complete", "success"))
        update_progress(1, 1, "Diagnostics complete")

    @run_with_progress("check_deps", "pb_diagnostics", "status_label", "btn_run_diagnostics", "btn_check_deps", "btn_check_index", "btn_test_embeddings")
    def _check_deps(self, *, update_progress) -> None:
        """Check dependencies only."""
        self.after(0, lambda: self.txt_diagnostics.delete("1.0", tk.END))
        update_progress(0, 1, "Checking dependencies...")

        report = doctor.check_and_install_dependencies(
            provider=self.settings.provider,
            auto_install=False,
            pip_timeout=300
        )

        def update_ui():
            self.txt_diagnostics.insert("1.0", "=== Dependency Check ===\n\n", "info")

            if report.missing_critical:
                self.txt_diagnostics.insert(tk.END, "Missing Critical:\n", "error")
                for pkg in report.missing_critical:
                    self.txt_diagnostics.insert(tk.END, f"  ✗ {pkg}\n", "error")
                self.txt_diagnostics.insert(tk.END, "\n")

            if report.missing_optional:
                self.txt_diagnostics.insert(tk.END, "Missing Optional:\n", "warning")
                for pkg in report.missing_optional:
                    self.txt_diagnostics.insert(tk.END, f"  ⚠ {pkg}\n", "warning")
                self.txt_diagnostics.insert(tk.END, "\n")

            self.txt_diagnostics.insert(tk.END, f"Installed ({len(report.installed)}):\n", "success")
            for pkg in report.installed[:20]:  # Show first 20
                self.txt_diagnostics.insert(tk.END, f"  ✓ {pkg}\n", "success")

            update_progress(1, 1, "Dependency check complete")

        self.after(0, update_ui)

    @run_with_progress("check_index", "pb_diagnostics", "status_label", "btn_run_diagnostics", "btn_check_deps", "btn_check_index", "btn_test_embeddings")
    def _check_index(self, *, update_progress) -> None:
        """Check index health."""
        self.after(0, lambda: self.txt_diagnostics.delete("1.0", tk.END))
        update_progress(0, 1, "Checking index health...")

        ix_dir = Path(self.settings.export_root) / os.getenv("INDEX_DIRNAME", "_index")

        if not ix_dir.exists():
            self.after(0, lambda: self.txt_diagnostics.insert("1.0", f"✗ Index not found at {ix_dir}\n", "error"))
            update_progress(1, 1, "Index not found")
            return

        stats = doctor._get_index_statistics(ix_dir)

        def update_ui():
            self.txt_diagnostics.insert("1.0", "=== Index Health ===\n\n", "info")
            self.txt_diagnostics.insert(tk.END, f"✓ Index directory: {ix_dir}\n", "success")
            self.txt_diagnostics.insert(tk.END, f"\n  Documents: {stats.get('num_documents', 0)}\n", "info")
            self.txt_diagnostics.insert(tk.END, f"  Conversations: {stats.get('num_conversations', 0)}\n", "info")
            self.txt_diagnostics.insert(tk.END, f"  Total chars: {stats.get('total_chars', 0):,}\n", "info")
            update_progress(1, 1, "Index health check complete")

        self.after(0, update_ui)

    @run_with_progress("test_embeddings", "pb_diagnostics", "status_label", "btn_run_diagnostics", "btn_check_deps", "btn_check_index", "btn_test_embeddings")
    def _test_embeddings(self, *, update_progress) -> None:
        """Test embedding functionality."""
        self.after(0, lambda: self.txt_diagnostics.delete("1.0", tk.END))
        update_progress(0, 1, "Testing embeddings...")

        success, dim = doctor._probe_embeddings(self.settings.provider)

        def update_ui():
            if success:
                self.txt_diagnostics.insert(tk.END, "✓ Embeddings working\n", "success")
                self.txt_diagnostics.insert(tk.END, f"  Provider: {self.settings.provider}\n", "info")
                self.txt_diagnostics.insert(tk.END, f"  Dimension: {dim}\n", "info")
            else:
                self.txt_diagnostics.insert(tk.END, "✗ Embeddings test failed\n", "error")
            update_progress(1, 1, "Embedding test complete")

        self.after(0, update_ui)

    @run_with_progress("force_rechunk", "pb_chunk", "lbl_chunk_progress", "btn_force_rechunk", "btn_incremental_chunk", "btn_surgical_rechunk")
    def _on_force_rechunk(self, *, update_progress) -> None:
        """Force re-chunk all conversations (deletes existing chunks)."""
        if not self.settings.export_root:
            self.after(0, lambda: messagebox.showwarning("No Root", "Please set export root first"))
            return

        response = messagebox.askyesno(
            "Confirm Force Re-chunk",
            "This will DELETE all existing chunks and recreate them.\n\n"
            "This operation may take a long time.\n\n"
            "Continue?"
        )
        if not response:
            return

        import shutil

        from emailops.text_chunker import prepare_index_units
        from emailops.utils import find_conversation_dirs, load_conversation

        root = Path(self.settings.export_root)
        chunks_dir = root / "_chunks"

        if chunks_dir.exists():
            shutil.rmtree(chunks_dir)
        chunks_dir.mkdir(parents=True, exist_ok=True)

        conv_dirs = find_conversation_dirs(root)
        total = len(conv_dirs)
        update_progress(0, total, "Starting re-chunk...")

        for i, conv_dir in enumerate(conv_dirs):
            if self.task.cancelled():
                break
            update_progress(i, total, f"Chunking {i+1}/{total}: {conv_dir.name}")

            try:
                conv_data = load_conversation(conv_dir, include_attachment_text=True)
                text_to_chunk = conv_data.get("conversation_txt", "")
                from emailops.utils import scrub_json_string
                text_to_chunk = scrub_json_string(text_to_chunk)

                chunks = prepare_index_units(
                    text=text_to_chunk,
                    doc_id=conv_dir.name,
                    doc_path=str(conv_dir / "Conversation.txt"),
                    chunk_size=self.var_cfg_chunk_size.get(),
                    chunk_overlap=self.var_cfg_chunk_overlap.get()
                )

                # Save chunks to a file
                chunk_file = chunks_dir / f"{conv_dir.name}.json"
                chunk_file.write_text(json.dumps(chunks, indent=2), encoding="utf-8")

            except Exception as e:
                module_logger.error(f"Failed to chunk {conv_dir.name}: {e}")

        update_progress(total, total, "Re-chunking complete.")
        self.after(0, lambda: messagebox.showinfo("Success", "All conversations have been re-chunked."))

    @run_with_progress("incremental_chunk", "pb_chunk", "lbl_chunk_progress", "btn_force_rechunk", "btn_incremental_chunk", "btn_surgical_rechunk")
    def _on_incremental_chunk(self, *, update_progress) -> None:
        """Incrementally chunk only new/changed conversations."""
        if not self.settings.export_root:
            self.after(0, lambda: messagebox.showwarning("No Root", "Please set export root first"))
            return

        from emailops.text_chunker import prepare_index_units
        from emailops.utils import find_conversation_dirs, load_conversation

        root = Path(self.settings.export_root)
        chunks_dir = root / "_chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        last_chunk_file = chunks_dir / "_last_chunk.txt"
        last_chunk_time = 0.0
        if last_chunk_file.exists():
            try:
                last_chunk_time = float(last_chunk_file.read_text())
            except (ValueError, OSError):
                last_chunk_time = 0.0

        conv_dirs = find_conversation_dirs(root)

        def needs_update(conv_dir):
            chunk_file = chunks_dir / f"{conv_dir.name}.json"
            if not chunk_file.exists():
                return True

            try:
                # Check Conversation.txt
                conv_txt_path = conv_dir / "Conversation.txt"
                if conv_txt_path.exists() and conv_txt_path.stat().st_mtime > last_chunk_time:
                    return True

                # Check attachments
                attachments_dir = conv_dir / "Attachments"
                if attachments_dir.exists():
                    for att_path in attachments_dir.rglob("*"):
                        if att_path.is_file() and att_path.stat().st_mtime > last_chunk_time:
                            return True
            except OSError:
                return True # Re-chunk if we can't read metadata
            return False

        to_chunk = [d for d in conv_dirs if needs_update(d)]
        total = len(to_chunk)
        update_progress(0, total, "Starting incremental chunk...")

        import concurrent.futures

        def chunk_one(conv_dir):
            try:
                conv_data = load_conversation(conv_dir, include_attachment_text=True)
                text_to_chunk = conv_data.get("conversation_txt", "")
                from emailops.utils import scrub_json_string
                text_to_chunk = scrub_json_string(text_to_chunk)

                chunks = prepare_index_units(
                    text=text_to_chunk,
                    doc_id=conv_dir.name,
                    doc_path=str(conv_dir / "Conversation.txt"),
                    chunk_size=self.var_cfg_chunk_size.get(),
                    chunk_overlap=self.var_cfg_chunk_overlap.get()
                )

                # Save chunks to a file
                chunk_file = chunks_dir / f"{conv_dir.name}.json"
                chunk_file.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
                return None
            except Exception as e:
                module_logger.error(f"Failed to chunk {conv_dir.name}: {e}")
                return str(e)

        num_workers = self.var_cfg_workers.get() if hasattr(self, 'var_cfg_workers') else 4
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(chunk_one, conv_dir): idx for idx, conv_dir in enumerate(conv_dirs)}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                idx = futures[future]
                update_progress(i, total, f"Chunking {i+1}/{total}: {conv_dirs[idx].name}")
                if self.task.cancelled():
                    break
        self.after(0, lambda: messagebox.showinfo("Success", f"Updated {total} conversations."))

    @run_with_progress("surgical_rechunk", "pb_chunk", "lbl_chunk_progress", "btn_force_rechunk", "btn_incremental_chunk", "btn_surgical_rechunk")
    def _on_surgical_rechunk(self, *, update_progress) -> None:
        """Re-chunk specific selected conversations."""
        selection = self.tree_convs.selection()
        if not selection:
            self.after(0, lambda: messagebox.showwarning("No Selection", "Please select one or more conversations from the 'Conversations' tab."))
            return

        conv_ids = [self.tree_convs.item(item)["values"][0] for item in selection]

        response = messagebox.askyesno(
            "Confirm Surgical Re-chunk",
            f"This will re-chunk the {len(conv_ids)} selected conversation(s).\n\nContinue?"
        )
        if not response:
            return

        from emailops.text_chunker import prepare_index_units
        from emailops.utils import load_conversation

        root = Path(self.settings.export_root)
        chunks_dir = root / "_chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        total = len(conv_ids)
        update_progress(0, total, "Starting surgical re-chunk...")

        for i, conv_id in enumerate(conv_ids):
            if self.task.cancelled():
                break
            conv_dir = root / conv_id
            update_progress(i, total, f"Chunking {i+1}/{total}: {conv_dir.name}")
            try:
                conv_data = load_conversation(conv_dir, include_attachment_text=True)
                text_to_chunk = conv_data.get("conversation_txt", "")

                chunks = prepare_index_units(
                    text=text_to_chunk,
                    doc_id=conv_dir.name,
                    doc_path=str(conv_dir / "Conversation.txt"),
                    chunk_size=self.var_cfg_chunk_size.get(),
                    chunk_overlap=self.var_cfg_chunk_overlap.get()
                )

                chunk_file = chunks_dir / f"{conv_dir.name}.json"
                chunk_file.write_text(json.dumps(chunks, indent=2), encoding="utf-8")

            except Exception as e:
                module_logger.error(f"Failed to chunk {conv_dir.name}: {e}")

        update_progress(total, total, "Surgical re-chunking complete.")
        self.after(0, lambda: messagebox.showinfo("Success", f"Successfully re-chunked {total} conversations."))

    def _list_chunked_convs(self) -> None:
        """List conversations that have been chunked."""
        if not self.settings.export_root:
            messagebox.showwarning("No Root", "Please set export root first")
            return

        chunks_dir = Path(self.settings.export_root) / "_chunks"
        if not chunks_dir.exists():
            messagebox.showinfo("No Chunks", f"No chunks directory found at:\n{chunks_dir}")
            return

        self.tree_chunks.delete(*self.tree_chunks.get_children())

        chunk_files = sorted(chunks_dir.glob("*.json"), key=os.path.getmtime, reverse=True)

        for chunk_file in chunk_files:
            try:
                conv_id = chunk_file.stem
                chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
                num_chunks = len(chunks)
                last_mod = datetime.fromtimestamp(chunk_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')

                self.tree_chunks.insert("", "end", values=(conv_id, num_chunks, "Chunked", last_mod))
            except Exception as e:
                module_logger.warning(f"Could not process chunk file {chunk_file}: {e}")

    def _clear_chunks_dir(self) -> None:
        """Clear the chunks directory."""
        if not self.settings.export_root:
            messagebox.showwarning("No Root", "Please set export root first")
            return

        chunks_dir = Path(self.settings.export_root) / "_chunks"
        if not chunks_dir.exists():
            messagebox.showinfo("No Chunks", "No chunks directory exists")
            return

        response = messagebox.askyesno(
            "Confirm Delete",
            f"Delete all chunks in:\n{chunks_dir}\n\nThis cannot be undone!"
        )
        if response:
            try:
                import shutil
                shutil.rmtree(chunks_dir)
                messagebox.showinfo("Success", "Chunks directory deleted")
                module_logger.info(f"Cleared chunks directory: {chunks_dir}")
            except Exception as e:
                module_logger.error(f"Failed to clear chunks: {e}")
                messagebox.showerror("Error", f"Failed to delete:\n{e!s}")

    @run_with_progress("analyze_thread", "pb_analyze", "status_label", "btn_analyze")
    def _on_analyze_thread(self, *, update_progress: Callable) -> None:
        """Analyze email thread with format/CSV options."""
        thread_dir = self.var_thread_dir.get().strip()
        if not thread_dir:
            self.after(0, lambda: messagebox.showwarning("Input Required", "Please select a conversation folder"))
            return

        thread_path = Path(thread_dir)
        if not thread_path.exists() or not (thread_path / "Conversation.txt").exists():
            self.after(0, lambda: messagebox.showerror("Invalid Path", "Folder must contain Conversation.txt"))
            return

        output_format = self.var_analysis_format.get()
        export_csv = self.var_export_csv.get()
        merge_manifest = self.var_merge_manifest.get()
        update_progress(0, 1, "Analyzing thread...")

        # HIGH #26: Fix method signature mismatch - analyze_conversation_dir gets provider from env
        # Set environment variable before calling instead of passing as parameter
        os.environ["EMBED_PROVIDER"] = self.settings.provider
        analysis = asyncio.run(
            summarizer.analyze_conversation_dir(
                thread_dir=thread_path,
                temperature=self.settings.temperature,
                merge_manifest=merge_manifest
            )
        )

        def update_ui():
            # Display results
            self.txt_analyze.delete("1.0", tk.END)

            if output_format in ["json", "both"]:
                json_path = thread_path / "summary.json"
                json_path.write_text(
                    json.dumps(analysis, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
                self.txt_analyze.insert("1.0", f"✓ Saved JSON to: {json_path}\n\n")
                self.txt_analyze.insert(tk.END, json.dumps(analysis, indent=2)[:2000] + "\n...\n")

            if output_format in ["markdown", "both"]:
                md_content = summarizer.format_analysis_as_markdown(analysis)
                md_path = thread_path / "summary.md"
                md_path.write_text(md_content, encoding="utf-8")
                self.txt_analyze.insert(tk.END, f"\n✓ Saved Markdown to: {md_path}\n")

            if export_csv:
                todos = analysis.get("next_actions", [])
                if todos:
                    from emailops.summarize_email_thread import _append_todos_csv
                    _append_todos_csv(thread_path.parent, thread_path.name, todos)
                    self.txt_analyze.insert(tk.END, f"✓ Exported {len(todos)} actions to todo.csv\n")

            self._set_status("Analysis complete", "success")
            update_progress(1, 1, "Analysis complete")
            messagebox.showinfo("Success", "Thread analysis complete")

        self.after(0, update_ui)

    def _drain_logs(self) -> None:
        """Drain log queue and display in GUI."""
        try:
            while True:
                try:
                    msg = self.log_queue.get_nowait()
                    self.txt_logs.insert(tk.END, msg + "\n")

                    # Apply tag based on log level
                    if "ERROR" in msg:
                        line_start = self.txt_logs.index("end-2l")
                        line_end = self.txt_logs.index("end-1l")
                        self.txt_logs.tag_add("ERROR", line_start, line_end)
                    elif "WARNING" in msg:
                        line_start = self.txt_logs.index("end-2l")
                        line_end = self.txt_logs.index("end-1l")
                        self.txt_logs.tag_add("WARNING", line_start, line_end)
                    elif "DEBUG" in msg:
                        line_start = self.txt_logs.index("end-2l")
                        line_end = self.txt_logs.index("end-1l")
                        self.txt_logs.tag_add("DEBUG", line_start, line_end)

                    self.txt_logs.see(tk.END)
                except queue.Empty:
                    break
        except Exception:
            pass

        # Schedule next drain
        self.after(100, self._drain_logs)

    def _change_log_level(self, event: tk.Event | None = None) -> None:
        """Change logging level."""
        try:
            level_name = self.var_log_level.get()
            level = getattr(logging, level_name, logging.INFO)
            logging.getLogger().setLevel(level)
            module_logger.setLevel(level)
            module_logger.info(f"Log level changed to {level_name}")
        except Exception as e:
            module_logger.error(f"Failed to change log level: {e}")

    def _save_logs(self) -> None:
        """Save current logs to file."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".log",
                filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=f"emailops_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            if filename:
                log_content = self.txt_logs.get("1.0", tk.END)
                Path(filename).write_text(log_content, encoding="utf-8")
                messagebox.showinfo("Success", f"Logs saved to {filename}")
                module_logger.info(f"Logs saved to {filename}")
        except Exception as e:
            module_logger.error(f"Failed to save logs: {e}")
            messagebox.showerror("Save Error", f"Failed to save logs:\n{e!s}")

    def _show_about(self) -> None:
        """Show about dialog."""
        try:
            dialog = tk.Toplevel(self)
            dialog.title("About EmailOps")
            dialog.geometry("500x400")
            dialog.resizable(False, False)

            # Create frame with padding
            frame = ttk.Frame(dialog, padding=20)
            frame.pack(fill=tk.BOTH, expand=True)

            # Title
            title = ttk.Label(frame, text="EmailOps Professional Assistant", font=("Arial", 14, "bold"))
            title.pack(pady=(0, 10))

            # Version info
            info_text = """
Version: 2.0
Provider: Vertex AI (Google Gemini)

A comprehensive email processing system with:
• Advanced semantic search with filters
• AI-powered email drafting
• Chat with context
• Thread analysis and summarization
• Text chunking and processing
• System diagnostics

© 2025 EmailOps Project
            """

            info_label = ttk.Label(frame, text=info_text.strip(), justify=tk.LEFT)
            info_label.pack(pady=10)

            # Close button
            ttk.Button(frame, text="Close", command=dialog.destroy).pack(pady=10)

        except Exception as e:
            module_logger.error(f"Failed to show about: {e}")
            messagebox.showerror("Error", f"Failed to show about:\n{e!s}")

    def _show_docs(self) -> None:
        """Show documentation dialog."""
        try:
            dialog = tk.Toplevel(self)
            dialog.title("EmailOps Documentation")
            dialog.geometry("700x600")

            # Create scrollable text
            frame = ttk.Frame(dialog, padding=10)
            frame.pack(fill=tk.BOTH, expand=True)

            text = tk.Text(frame, wrap="word", font=("Arial", 10))
            text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

            scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text.configure(yscrollcommand=scrollbar.set)

            docs = """
EmailOps Quick Start Guide
===========================

1. SETUP
   - Set Export Root to your email export directory
   - Configure GCP settings in Configuration tab
   - Build index in Index tab

2. SEARCH
   - Enter query and adjust k and similarity threshold
   - Use Advanced Filters for precise results
   - Click search results to view snippets

3. DRAFT REPLY
   - Select conversation from list
   - Optionally add query for context
   - Generate and save as .eml file

4. DRAFT FRESH EMAIL
   - Enter To, Cc, Subject
   - Provide intent/instructions
   - Generate and save as .eml file

5. CHAT
   - Ask questions about your emails
   - Sessions persist chat history
   - Citations show sources

6. ANALYZE THREAD
   - Select conversation folder
   - Choose output format (JSON/Markdown)
   - Optionally export actions to CSV

7. DIAGNOSTICS
   - Check dependencies
   - Verify index health
   - Test embeddings

8. CONFIGURATION
   - Set GCP Project/Region
   - Configure chunking parameters
   - Set email sender details

For full documentation, see emailops_docs/ directory.
            """

            text.insert("1.0", docs.strip())
            text.config(state="disabled")

            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)

        except Exception as e:
            module_logger.error(f"Failed to show docs: {e}")
            messagebox.showerror("Error", f"Failed to show documentation:\n{e!s}")

    def _export_search_results(self) -> None:
        """Export search results to CSV."""
        if not self.search_results:
            messagebox.showwarning("No Results", "No search results to export")
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            if not filename:
                return

            from pathlib import Path
            with Path(filename).open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Score", "Document ID", "Subject", "Conv ID", "Type", "Date", "Text Preview"])
                for result in self.search_results:
                    writer.writerow([
                        f"{result.get('score', 0):.4f}",
                        result.get("id", ""),
                        result.get("subject", ""),
                        result.get("conv_id", ""),
                        result.get("type", ""),
                        result.get("date", ""),
                        result.get("text", "")[:120]
                    ])

            messagebox.showinfo("Success", f"Exported {len(self.search_results)} results to {filename}")
            module_logger.info(f"Search results exported to {filename}")
        except Exception as e:
            module_logger.error(f"Failed to export search results: {e}")
            messagebox.showerror("Export Error", f"Failed to export:\n{e!s}")

    def _export_chat_history(self) -> None:
        """Export chat history to text file."""
        chat_content = self.txt_chat.get("1.0", tk.END).strip()
        if not chat_content:
            messagebox.showwarning("No History", "No chat history to export")
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            if not filename:
                return

            Path(filename).write_text(chat_content, encoding="utf-8")
            messagebox.showinfo("Success", f"Chat history exported to {filename}")
            module_logger.info(f"Chat history exported to {filename}")
        except Exception as e:
            module_logger.error(f"Failed to export chat history: {e}")
            messagebox.showerror("Export Error", f"Failed to export:\n{e!s}")

    def _save_analysis_results(self) -> None:
        """Save thread analysis results."""
        analysis_content = self.txt_analyze.get("1.0", tk.END).strip()
        if not analysis_content:
            messagebox.showwarning("No Results", "No analysis results to save")
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[
                    ("Text files", "*.txt"),
                    ("JSON files", "*.json"),
                    ("Markdown files", "*.md"),
                    ("All files", "*.*")
                ],
                initialfile=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            if not filename:
                return

            Path(filename).write_text(analysis_content, encoding="utf-8")
            messagebox.showinfo("Success", f"Analysis saved to {filename}")
            module_logger.info(f"Analysis results saved to {filename}")
        except Exception as e:
            module_logger.error(f"Failed to save analysis: {e}")
            messagebox.showerror("Save Error", f"Failed to save:\n{e!s}")

    def _update_status(self, msg: str) -> None:
        """Backward compatibility alias for _set_status."""
        self._set_status(msg, "info")


# ------------------------------- Main Entry Point -------------------------------

def main() -> None:
    """Main entry point for the GUI application."""
    parser = argparse.ArgumentParser(description="EmailOps Professional GUI")
    parser.add_argument("--root", help="Initial export root directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        module_logger.setLevel(logging.DEBUG)

    app = EmailOpsApp()

    if args.root:
        app.var_root.set(args.root)
        app.settings.export_root = args.root
        app.settings.save()

    try:
        app.mainloop()
    except KeyboardInterrupt:
        module_logger.info("Application interrupted by user")
    except Exception as e:
        module_logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
