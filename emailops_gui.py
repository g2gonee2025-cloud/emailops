#!/usr/bin/env python3
from __future__ import annotations

"""
EmailOps GUI

A Tkinter/ttk GUI that exposes the main user-facing features:
- Search index (top-K with snippets)
- Draft Reply (.eml) for a selected conversation
- Draft Fresh Email (.eml)
- Chat over retrieved context
- List conversations (newest -> oldest)
- Build/Update index (email_indexer)
- Analyze thread (facts-ledger summary)
- Unified logging and progress bars

The GUI is resilient to both package and local-module import layouts.
"""

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ------------------------------- Robust imports -------------------------------

def _try_imports():
    """
    Try to import modules from 'emailops.*'; fall back to local files.
    Returns a namespace-like dict of resolved modules/objects.
    """
    ns: dict[str, Any] = {}
    # Processor (primary high-level API)
    try:
        from emailops import processor as _processor  # package
        ns["processor"] = _processor
    except Exception:
        from . import processor as _processor  # local
        ns["processor"] = _processor

    # Indexer (CLI-oriented; we'll call main() in a worker thread)
    try:
        from emailops import email_indexer as _indexer
    except Exception:
        import email_indexer as _indexer
    ns["indexer"] = _indexer

    # Summarizer (optional tab)
    try:
        from emailops import summarize_email_thread as _summarizer
    except Exception:
        import summarize_email_thread as _summarizer
    ns["summarizer"] = _summarizer

    # Validators (for path hygiene)
    try:
        from emailops.validators import validate_directory_path  # package
        ns["validate_directory_path"] = validate_directory_path
    except Exception:
        from validators import validate_directory_path  # local
        ns["validate_directory_path"] = validate_directory_path

    # Utils logger (attach our GUI handler to bubble up module logs)
    try:
        from emailops.utils import logger as _module_logger  # package
    except Exception:
        try:
            from utils import logger as _module_logger  # local
        except Exception:
            _module_logger = logging.getLogger("emailops")
    ns["module_logger"] = _module_logger

    return ns

NS = _try_imports()
processor = NS["processor"]
email_indexer = NS["indexer"]
summarizer = NS["summarizer"]
validate_directory_path = NS["validate_directory_path"]
module_logger = NS["module_logger"]

# ------------------------------- Logging plumbing -----------------------------

class QueueHandler(logging.Handler):
    """Log handler that writes records into a queue (consumed by the GUI)."""
    def __init__(self, log_queue: queue.Queue[str]):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.log_queue.put_nowait(msg)
        except Exception:
            # Never crash the app because logging failed
            pass


def configure_logging(log_queue: queue.Queue[str]) -> None:
    """
    Configure root + module loggers to feed the GUI.
    - Attach a QueueHandler
    - Keep default formatting concise
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s: %(message)s")
    qh = QueueHandler(log_queue)
    qh.setFormatter(fmt)

    # Avoid duplicate handlers on re-run
    for h in list(root.handlers):
        root.removeHandler(h)

    root.addHandler(qh)

    # Make sure the internal module logger propagates
    try:
        module_logger.propagate = True
        module_logger.setLevel(logging.INFO)
    except Exception:
        pass


# --------------------------------- App settings -------------------------------

SETTINGS_FILE = Path.home() / ".emailops_gui.json"

@dataclass
class AppSettings:
    export_root: str = ""  # Folder that contains conversation folders and _index
    provider: str = "vertex"
    persona: str = os.getenv("PERSONA", getattr(processor, "PERSONA_DEFAULT", "expert insurance CSR"))
    sim_threshold: float = getattr(processor, "SIM_THRESHOLD_DEFAULT", 0.30)
    reply_tokens: int = getattr(processor, "REPLY_TOKENS_TARGET_DEFAULT", 20000)
    fresh_tokens: int = getattr(processor, "FRESH_TOKENS_TARGET_DEFAULT", 10000)
    reply_policy: str = getattr(processor, "REPLY_POLICY_DEFAULT", "reply_all")
    temperature: float = 0.2
    k: int = 25  # default top-K for search/chat
    last_to: str = ""
    last_cc: str = ""
    last_subject: str = ""

    def save(self) -> None:
        try:
            SETTINGS_FILE.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def load() -> "AppSettings":
        try:
            if SETTINGS_FILE.exists():
                raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                return AppSettings(**raw)
        except Exception:
            pass
        return AppSettings()


# ------------------------------- Worker utilities -----------------------------

class TaskController:
    """Simple cancellation token and busy flag for long-running tasks."""
    def __init__(self) -> None:
        self._busy = False
        self._cancel = False
        self._lock = threading.Lock()

    def start(self) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            self._cancel = False
            return True

    def done(self) -> None:
        with self._lock:
            self._busy = False
            self._cancel = False

    def cancel(self) -> None:
        with self._lock:
            self._cancel = True

    def cancelled(self) -> bool:
        with self._lock:
            return self._cancel

    def busy(self) -> bool:
        with self._lock:
            return self._busy


# ----------------------------------- GUI App ----------------------------------

class EmailOpsApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("EmailOps — Assistant")
        self.geometry("1200x780")

        # State
        self.settings = AppSettings.load()
        self.task = TaskController()
        self.log_queue: queue.Queue[str] = queue.Queue()
        configure_logging(self.log_queue)

        # Top-level layout
        self._build_menu()
        self._build_header()
        self._build_tabs()
        self._build_log_tab()

        # Periodic log pump
        self.after(100, self._drain_logs)

    # ------------- UI scaffolding -------------

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save Settings", command=self._save_settings)
        filemenu.add_command(label="Load Settings", command=self._load_settings)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=helpmenu)

        self.config(menu=menubar)

    def _build_header(self) -> None:
        frm = ttk.Frame(self, padding=8)
        frm.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(frm, text="Export Root:").pack(side=tk.LEFT)
        self.var_root = tk.StringVar(value=self.settings.export_root)
        self.ent_root = ttk.Entry(frm, width=80, textvariable=self.var_root)
        self.ent_root.pack(side=tk.LEFT, padx=6)
        ttk.Button(frm, text="Browse…", command=self._choose_root).pack(side=tk.LEFT, padx=4)

        ttk.Label(frm, text="Provider:").pack(side=tk.LEFT, padx=(20, 0))
        self.var_provider = tk.StringVar(value=self.settings.provider)
        self.cmb_provider = ttk.Combobox(frm, width=10, state="readonly", textvariable=self.var_provider,
                                         values=["vertex"])
        self.cmb_provider.pack(side=tk.LEFT, padx=4)

        ttk.Label(frm, text="Temp:").pack(side=tk.LEFT, padx=(20,0))
        self.var_temp = tk.DoubleVar(value=self.settings.temperature)
        ttk.Spinbox(frm, from_=0.0, to=1.0, increment=0.05, width=6, textvariable=self.var_temp).pack(side=tk.LEFT)

        ttk.Label(frm, text="Persona:").pack(side=tk.LEFT, padx=(20, 0))
        self.var_persona = tk.StringVar(value=self.settings.persona)
        ttk.Entry(frm, width=28, textvariable=self.var_persona).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(frm, textvariable=self.status_var, foreground="#555").pack(side=tk.RIGHT)

    def _build_tabs(self) -> None:
        self.nb = ttk.Notebook(self)
        self.nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.tab_search = ttk.Frame(self.nb)
        self.tab_reply = ttk.Frame(self.nb)
        self.tab_fresh = ttk.Frame(self.nb)
        self.tab_chat = ttk.Frame(self.nb)
        self.tab_convs = ttk.Frame(self.nb)
        self.tab_index = ttk.Frame(self.nb)
        self.tab_analyze = ttk.Frame(self.nb)
        self.tab_logs = ttk.Frame(self.nb)

        self.nb.add(self.tab_search, text="Search")
        self.nb.add(self.tab_reply, text="Draft Reply")
        self.nb.add(self.tab_fresh, text="Draft Fresh")
        self.nb.add(self.tab_chat, text="Chat")
        self.nb.add(self.tab_convs, text="Conversations")
        self.nb.add(self.tab_index, text="Index")
        self.nb.add(self.tab_analyze, text="Analyze Thread")
        self.nb.add(self.tab_logs, text="Logs")

        self._build_search_tab()
        self._build_reply_tab()
        self._build_fresh_tab()
        self._build_chat_tab()
        self._build_conversations_tab()
        self._build_index_tab()
        self._build_analyze_tab()

    def _build_log_tab(self) -> None:
        frm = self.tab_logs
        top = ttk.Frame(frm, padding=8)
        top.pack(fill=tk.BOTH, expand=True)

        self.txt_logs = tk.Text(top, wrap="word")
        self.txt_logs.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        yscroll = ttk.Scrollbar(top, orient=tk.VERTICAL, command=self.txt_logs.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_logs.configure(yscrollcommand=yscroll.set)

        btns = ttk.Frame(frm, padding=8)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Clear Logs", command=lambda: self.txt_logs.delete("1.0", tk.END)).pack(side=tk.LEFT)
        ttk.Button(btns, text="Save Logs…", command=self._save_logs).pack(side=tk.LEFT, padx=6)

    # ------------- Search tab -------------

    def _build_search_tab(self) -> None:
        frm = self.tab_search

        row1 = ttk.Frame(frm, padding=8)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="Query:").pack(side=tk.LEFT)
        self.var_search_q = tk.StringVar()
        ttk.Entry(row1, width=80, textvariable=self.var_search_q).pack(side=tk.LEFT, padx=6)
        ttk.Label(row1, text="k:").pack(side=tk.LEFT)
        self.var_k = tk.IntVar(value=self.settings.k)
        ttk.Spinbox(row1, from_=1, to=250, width=6, textvariable=self.var_k).pack(side=tk.LEFT, padx=4)
        ttk.Label(row1, text="Sim ≥").pack(side=tk.LEFT)
        self.var_sim = tk.DoubleVar(value=self.settings.sim_threshold)
        ttk.Spinbox(row1, from_=0.0, to=1.0, increment=0.01, width=6, textvariable=self.var_sim).pack(side=tk.LEFT, padx=4)

        self.btn_search = ttk.Button(row1, text="Search", command=self._on_search)
        self.btn_search.pack(side=tk.LEFT, padx=8)
        self.pb_search = ttk.Progressbar(row1, mode="indeterminate", length=150)
        self.pb_search.pack(side=tk.LEFT, padx=8)

        row2 = ttk.Frame(frm, padding=(8,0,8,8))
        row2.pack(fill=tk.BOTH, expand=True)

        cols = ("score", "subject", "id")
        self.tree = ttk.Treeview(row2, columns=cols, show="headings", height=12)
        self.tree.heading("score", text="Score")
        self.tree.heading("subject", text="Subject")
        self.tree.heading("id", text="Doc ID")
        self.tree.column("score", width=70, anchor=tk.CENTER)
        self.tree.column("subject", width=500)
        self.tree.column("id", width=500)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._show_snippet)

        yscroll = ttk.Scrollbar(row2, orient=tk.VERTICAL, command=self.tree.yview)
        yscroll.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.configure(yscrollcommand=yscroll.set)

        right = ttk.Frame(row2, padding=(8,0,8,8))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(right, text="Snippet:").pack(anchor="w")
        self.txt_snip = tk.Text(right, height=14, wrap="word")
        self.txt_snip.pack(fill=tk.BOTH, expand=True)

        self.search_results: list[dict[str, Any]] = []

    # ------------- Draft Reply tab -------------

    def _build_reply_tab(self) -> None:
        frm = self.tab_reply

        row1 = ttk.Frame(frm, padding=8)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="Conversation ID:").pack(side=tk.LEFT)
        self.var_conv = tk.StringVar()
        self.cmb_conv = ttk.Combobox(row1, width=60, textvariable=self.var_conv)
        self.cmb_conv.pack(side=tk.LEFT, padx=6)
        ttk.Button(row1, text="Load Conversations", command=self._load_conversations).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(frm, padding=8)
        row2.pack(fill=tk.X)
        ttk.Label(row2, text="Query (optional):").pack(side=tk.LEFT)
        self.var_reply_q = tk.StringVar()
        ttk.Entry(row2, width=72, textvariable=self.var_reply_q).pack(side=tk.LEFT, padx=6)
        ttk.Label(row2, text="Tokens:").pack(side=tk.LEFT)
        self.var_reply_tokens = tk.IntVar(value=self.settings.reply_tokens)
        ttk.Spinbox(row2, from_=2000, to=100000, increment=1000, width=10, textvariable=self.var_reply_tokens).pack(side=tk.LEFT, padx=4)
        ttk.Label(row2, text="Policy:").pack(side=tk.LEFT)
        self.var_reply_policy = tk.StringVar(value=self.settings.reply_policy)
        ttk.Combobox(row2, width=12, state="readonly", textvariable=self.var_reply_policy,
                     values=["reply_all", "smart", "sender_only"]).pack(side=tk.LEFT, padx=4)
        self.var_reply_attach = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="Include attachments", variable=self.var_reply_attach).pack(side=tk.LEFT, padx=12)

        row3 = ttk.Frame(frm, padding=8)
        row3.pack(fill=tk.X)
        self.btn_draft_reply = ttk.Button(row3, text="Generate Reply", command=self._on_draft_reply)
        self.btn_draft_reply.pack(side=tk.LEFT)
        self.pb_reply = ttk.Progressbar(row3, mode="indeterminate", length=220)
        self.pb_reply.pack(side=tk.LEFT, padx=8)
        ttk.Button(row3, text="Save .eml…", command=self._save_eml_reply).pack(side=tk.LEFT, padx=8)

        self.txt_reply = tk.Text(frm, wrap="word")
        self.txt_reply.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

        self._last_reply_bytes: Optional[bytes] = None
        self._last_reply_meta: Optional[dict[str, Any]] = None

    # ------------- Draft Fresh tab -------------

    def _build_fresh_tab(self) -> None:
        frm = self.tab_fresh

        r1 = ttk.Frame(frm, padding=8); r1.pack(fill=tk.X)
        ttk.Label(r1, text="To:").pack(side=tk.LEFT)
        self.var_to = tk.StringVar(value=self.settings.last_to)
        ttk.Entry(r1, width=50, textvariable=self.var_to).pack(side=tk.LEFT, padx=6)

        ttk.Label(r1, text="Cc:").pack(side=tk.LEFT)
        self.var_cc = tk.StringVar(value=self.settings.last_cc)
        ttk.Entry(r1, width=50, textvariable=self.var_cc).pack(side=tk.LEFT, padx=6)

        r2 = ttk.Frame(frm, padding=8); r2.pack(fill=tk.X)
        ttk.Label(r2, text="Subject:").pack(side=tk.LEFT)
        self.var_subject = tk.StringVar(value=self.settings.last_subject)
        ttk.Entry(r2, width=92, textvariable=self.var_subject).pack(side=tk.LEFT, padx=6)

        r3 = ttk.Frame(frm, padding=8); r3.pack(fill=tk.X)
        ttk.Label(r3, text="Intent / Instructions:").pack(side=tk.LEFT)
        self.var_fresh_q = tk.StringVar()
        ttk.Entry(r3, width=90, textvariable=self.var_fresh_q).pack(side=tk.LEFT, padx=6)
        ttk.Label(r3, text="Tokens:").pack(side=tk.LEFT)
        self.var_fresh_tokens = tk.IntVar(value=self.settings.fresh_tokens)
        ttk.Spinbox(r3, from_=2000, to=100000, increment=1000, width=10, textvariable=self.var_fresh_tokens).pack(side=tk.LEFT, padx=4)
        self.var_fresh_attach = tk.BooleanVar(value=True)
        ttk.Checkbutton(r3, text="Include attachments", variable=self.var_fresh_attach).pack(side=tk.LEFT, padx=12)

        r4 = ttk.Frame(frm, padding=8); r4.pack(fill=tk.X)
        self.btn_draft_fresh = ttk.Button(r4, text="Generate Fresh Email", command=self._on_draft_fresh)
        self.btn_draft_fresh.pack(side=tk.LEFT)
        self.pb_fresh = ttk.Progressbar(r4, mode="indeterminate", length=220)
        self.pb_fresh.pack(side=tk.LEFT, padx=8)
        ttk.Button(r4, text="Save .eml…", command=self._save_eml_fresh).pack(side=tk.LEFT, padx=8)

        self.txt_fresh = tk.Text(frm, wrap="word")
        self.txt_fresh.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

        self._last_fresh_bytes: Optional[bytes] = None
        self._last_fresh_meta: Optional[dict[str, Any]] = None

    # ------------- Chat tab -------------

    def _build_chat_tab(self) -> None:
        frm = self.tab_chat

        r1 = ttk.Frame(frm, padding=8); r1.pack(fill=tk.X)
        ttk.Label(r1, text="Question:").pack(side=tk.LEFT)
        self.var_chat_q = tk.StringVar()
        ttk.Entry(r1, width=90, textvariable=self.var_chat_q).pack(side=tk.LEFT, padx=6)
        ttk.Label(r1, text="k:").pack(side=tk.LEFT)
        self.var_chat_k = tk.IntVar(value=self.settings.k)
        ttk.Spinbox(r1, from_=1, to=100, width=6, textvariable=self.var_chat_k).pack(side=tk.LEFT, padx=4)
        self.btn_chat = ttk.Button(r1, text="Ask", command=self._on_chat)
        self.btn_chat.pack(side=tk.LEFT, padx=8)
        self.pb_chat = ttk.Progressbar(r1, mode="indeterminate", length=180)
        self.pb_chat.pack(side=tk.LEFT, padx=8)

        self.txt_chat = tk.Text(frm, wrap="word")
        self.txt_chat.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

    # ------------- Conversations tab -------------

    def _build_conversations_tab(self) -> None:
        frm = self.tab_convs

        top = ttk.Frame(frm, padding=8); top.pack(fill=tk.X)
        ttk.Button(top, text="List Conversations", command=self._on_list_convs).pack(side=tk.LEFT)
        self.pb_convs = ttk.Progressbar(top, mode="indeterminate", length=180)
        self.pb_convs.pack(side=tk.LEFT, padx=8)

        cols = ("conv_id", "subject", "first", "last", "count")
        self.tree_convs = ttk.Treeview(frm, columns=cols, show="headings", height=20)
        for k, w in (("conv_id", 220), ("subject", 520), ("first", 140), ("last", 140), ("count", 70)):
            self.tree_convs.heading(k, text=k)
            self.tree_convs.column(k, width=w)
        self.tree_convs.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Button(frm, text="Use Selected in Draft Reply", command=self._use_selected_conv).pack(padx=8, pady=(0,8), anchor="w")

    # ------------- Index tab -------------

    def _build_index_tab(self) -> None:
        frm = self.tab_index

        r1 = ttk.Frame(frm, padding=8); r1.pack(fill=tk.X)
        ttk.Label(r1, text="Batch size:").pack(side=tk.LEFT)
        self.var_batch = tk.IntVar(value=int(os.getenv("EMBED_BATCH", "64")))
        ttk.Spinbox(r1, from_=1, to=250, width=8, textvariable=self.var_batch).pack(side=tk.LEFT, padx=4)
        self.var_force = tk.BooleanVar(value=False)
        ttk.Checkbutton(r1, text="Force full re-index", variable=self.var_force).pack(side=tk.LEFT, padx=12)
        ttk.Label(r1, text="Limit per conversation:").pack(side=tk.LEFT, padx=(20,0))
        self.var_limit = tk.IntVar(value=0)
        ttk.Spinbox(r1, from_=0, to=2000, width=8, textvariable=self.var_limit).pack(side=tk.LEFT, padx=4)
        self.btn_build = ttk.Button(r1, text="Build / Update Index", command=self._on_build_index)
        self.btn_build.pack(side=tk.LEFT, padx=12)
        self.pb_index = ttk.Progressbar(r1, mode="indeterminate", length=260)
        self.pb_index.pack(side=tk.LEFT, padx=8)

        hint = ttk.Label(frm, text="Tip: Index builder logs progress here. The progress bar reflects overall activity.")
        hint.pack(anchor="w", padx=8, pady=(0,8))

    # ------------- Analyze tab -------------

    def _build_analyze_tab(self) -> None:
        frm = self.tab_analyze

        row = ttk.Frame(frm, padding=8); row.pack(fill=tk.X)
        ttk.Label(row, text="Conversation folder:").pack(side=tk.LEFT)
        self.var_thread_dir = tk.StringVar(value="")
        ttk.Entry(row, width=80, textvariable=self.var_thread_dir).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Browse…", command=self._choose_thread_dir).pack(side=tk.LEFT, padx=4)
        self.btn_analyze = ttk.Button(row, text="Analyze", command=self._on_analyze_thread)
        self.btn_analyze.pack(side=tk.LEFT, padx=8)
        self.pb_analyze = ttk.Progressbar(row, mode="indeterminate", length=200)
        self.pb_analyze.pack(side=tk.LEFT, padx=8)

        self.txt_analyze = tk.Text(frm, wrap="word")
        self.txt_analyze.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

    # ------------------------------- Actions ----------------------------------

    def _with_root_and_index(self) -> tuple[Optional[Path], Optional[Path]]:
        root = Path(self.var_root.get().strip()).expanduser()
        ok, msg = validate_directory_path(root, must_exist=True, allow_parent_traversal=False)
        if not ok:
            messagebox.showerror("Invalid root", msg)
            return None, None
        ix_dirname = getattr(processor, "INDEX_DIRNAME", getattr(processor, "INDEX_DIRNAME_DEFAULT", "_index"))
        return root, root / ix_dirname

    def _choose_root(self) -> None:
        d = filedialog.askdirectory(title="Choose export root")
        if d:
            self.var_root.set(d)
            self.settings.export_root = d
            self.settings.save()

    def _choose_thread_dir(self) -> None:
        d = filedialog.askdirectory(title="Choose a conversation folder (contains Conversation.txt)")
        if d:
            self.var_thread_dir.set(d)

    def _save_settings(self) -> None:
        self._sync_settings_from_ui()
        self.settings.save()
        messagebox.showinfo("Settings", f"Saved to {SETTINGS_FILE}")

    def _load_settings(self) -> None:
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
        self.var_to.set(self.settings.last_to)
        self.var_cc.set(self.settings.last_cc)
        self.var_subject.set(self.settings.last_subject)
        messagebox.showinfo("Settings", "Reloaded settings.")

    def _sync_settings_from_ui(self) -> None:
        self.settings.export_root = self.var_root.get().strip()
        self.settings.provider = self.var_provider.get().strip()
        self.settings.persona = self.var_persona.get().strip()
        self.settings.temperature = float(self.var_temp.get())
        self.settings.sim_threshold = float(self.var_sim.get())
        self.settings.k = int(self.var_k.get())
        self.settings.reply_tokens = int(self.var_reply_tokens.get())
        self.settings.fresh_tokens = int(self.var_fresh_tokens.get())
        self.settings.reply_policy = self.var_reply_policy.get().strip()
        self.settings.last_to = self.var_to.get().strip()
        self.settings.last_cc = self.var_cc.get().strip()
        self.settings.last_subject = self.var_subject.get().strip()

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About EmailOps GUI",
            "A simple GUI that exposes search, chat, drafting, indexing, and analysis.\n"
            "Backed by processor.py (draft/chat/search), email_indexer.py (index), and summarize_email_thread.py (analysis)."
        )

    def _save_logs(self) -> None:
        p = filedialog.asksaveasfilename(title="Save logs", defaultextension=".log", filetypes=[("Log files", "*.log"), ("Text", "*.txt"), ("All files", "*.*")])
        if not p:
            return
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(self.txt_logs.get("1.0", tk.END))
            messagebox.showinfo("Logs", f"Saved logs to:\n{p}")
        except Exception as e:
            messagebox.showerror("Logs", f"Failed to save logs: {e}")

    # ---- search

    def _on_search(self) -> None:
        if not self.task.start():
            messagebox.showwarning("Busy", "Another task is running.")
            return

        root, ix_dir = self._with_root_and_index()
        if not root:
            self.task.done()
            return

        q = self.var_search_q.get().strip()
        if not q:
            messagebox.showwarning("Search", "Please enter a query.")
            self.task.done()
            return

        provider = self.var_provider.get().strip() or "vertex"
        k = int(self.var_k.get())
        sim = float(self.var_sim.get())

        self.tree.delete(*self.tree.get_children())
        self.txt_snip.delete("1.0", tk.END)
        self.pb_search.start(10)
        self.status_var.set("Searching…")

        def worker():
            try:
                # processor._search is the same engine used by the CLI search. :contentReference[oaicite:5]{index=5}
                results = processor._search(ix_dir, q, k=k, provider=provider, conv_id_filter=None)
                self.search_results = results
                def update():
                    self.tree.delete(*self.tree.get_children())
                    for r in results:
                        self.tree.insert("", tk.END, values=(f"{r.get('score',0):.3f}", r.get("subject",""), r.get("id","")))
                    self.status_var.set(f"Found {len(results)} results (Sim ≥ {sim:.2f}).")
                self.after(0, update)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Search failed", str(e)))
            finally:
                self.after(0, lambda: self.pb_search.stop())
                self.task.done()

        threading.Thread(target=worker, daemon=True).start()

    def _show_snippet(self, _evt=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        if 0 <= idx < len(self.search_results):
            snip = self.search_results[idx].get("text", "")
            self.txt_snip.delete("1.0", tk.END)
            self.txt_snip.insert(tk.END, snip)

    # ---- conversations

    def _load_conversations(self) -> None:
        if not self.task.start():
            messagebox.showwarning("Busy", "Another task is running.")
            return
        root, ix_dir = self._with_root_and_index()
        if not root:
            self.task.done()
            return

        self.pb_convs.start(10)
        self.status_var.set("Loading conversations…")

        def worker():
            try:
                convs = processor.list_conversations_newest_first(ix_dir)  # :contentReference[oaicite:6]{index=6}
                items = [c.get("conv_id","") for c in convs]
                def update():
                    self.cmb_conv["values"] = items
                    if items:
                        self.cmb_conv.set(items[0])
                    self.status_var.set(f"Loaded {len(items)} conversations.")
                self.after(0, update)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Conversations", str(e)))
            finally:
                self.after(0, lambda: self.pb_convs.stop())
                self.task.done()

        threading.Thread(target=worker, daemon=True).start()

    def _on_list_convs(self) -> None:
        if not self.task.start():
            messagebox.showwarning("Busy", "Another task is running.")
            return
        root, ix_dir = self._with_root_and_index()
        if not root:
            self.task.done()
            return

        self.tree_convs.delete(*self.tree_convs.get_children())
        self.pb_convs.start(10)
        self.status_var.set("Listing conversations…")

        def worker():
            try:
                convs = processor.list_conversations_newest_first(ix_dir)  # :contentReference[oaicite:7]{index=7}
                def update():
                    for c in convs:
                        self.tree_convs.insert("", tk.END, values=(
                            c.get("conv_id",""),
                            c.get("subject",""),
                            c.get("first_date_str",""),
                            c.get("last_date_str",""),
                            c.get("count",""),
                        ))
                    self.status_var.set(f"{len(convs)} conversations.")
                self.after(0, update)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("List Conversations", str(e)))
            finally:
                self.after(0, lambda: self.pb_convs.stop())
                self.task.done()

        threading.Thread(target=worker, daemon=True).start()

    def _use_selected_conv(self) -> None:
        sel = self.tree_convs.selection()
        if not sel:
            messagebox.showinfo("Conversations", "Select a conversation first.")
            return
        conv_id = self.tree_convs.item(sel[0])["values"][0]
        self.nb.select(self.tab_reply)
        self.cmb_conv.set(conv_id)
        self.status_var.set(f"Selected conversation {conv_id} for Draft Reply.")

    # ---- draft reply

    def _on_draft_reply(self) -> None:
        if not self.task.start():
            messagebox.showwarning("Busy", "Another task is running.")
            return
        root, ix_dir = self._with_root_and_index()
        if not root:
            self.task.done()
            return

        conv_id = self.var_conv.get().strip()
        if not conv_id:
            messagebox.showwarning("Draft Reply", "Please select a Conversation ID (load conversations first).")
            self.task.done()
            return

        query = self.var_reply_q.get().strip() or None
        provider = self.var_provider.get().strip() or "vertex"
        tokens = int(self.var_reply_tokens.get())
        temp = float(self.var_temp.get())
        policy = self.var_reply_policy.get().strip() or getattr(processor, "REPLY_POLICY_DEFAULT", "reply_all")
        include_attachments = bool(self.var_reply_attach.get())

        self.pb_reply.start(10)
        self.txt_reply.delete("1.0", tk.END)
        self.status_var.set("Drafting reply… (see Logs for details)")

        def worker():
            try:
                # High-level orchestration; handles context, drafting (critic/audit), and eml compose. :contentReference[oaicite:8]{index=8}
                result = processor.draft_email_reply_eml(
                    export_root=root,
                    conv_id=conv_id,
                    provider=provider,
                    query=query,
                    sim_threshold=float(self.var_sim.get()),
                    target_tokens=tokens,
                    temperature=temp,
                    include_attachments=include_attachments,
                    sender=None,
                    reply_to=None,
                    reply_policy=policy,
                )
                eml_bytes = result.get("eml_bytes", b"")
                draft_json = result.get("draft_json", {})
                body = (draft_json.get("final_draft") or {}).get("email_draft", "")
                def update():
                    self._last_reply_bytes = eml_bytes
                    self._last_reply_meta = result
                    self.txt_reply.insert(tk.END, body or "(no body)")
                    self.status_var.set("Reply draft ready.")
                self.after(0, update)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Draft Reply failed", str(e)))
            finally:
                self.after(0, lambda: self.pb_reply.stop())
                self.task.done()

        threading.Thread(target=worker, daemon=True).start()

    def _save_eml_reply(self) -> None:
        if not self._last_reply_bytes:
            messagebox.showinfo("Save .eml", "Nothing to save. Generate a draft first.")
            return
        p = filedialog.asksaveasfilename(title="Save reply .eml", defaultextension=".eml",
                                         filetypes=[("Email files", "*.eml"), ("All files", "*.*")])
        if not p:
            return
        try:
            Path(p).write_bytes(self._last_reply_bytes)
            messagebox.showinfo("Save .eml", f"Saved:\n{p}")
        except Exception as e:
            messagebox.showerror("Save .eml", f"Failed to save: {e}")

    # ---- draft fresh

    def _on_draft_fresh(self) -> None:
        if not self.task.start():
            messagebox.showwarning("Busy", "Another task is running.")
            return
        root, ix_dir = self._with_root_and_index()
        if not root:
            self.task.done()
            return

        to = [x.strip() for x in self.var_to.get().split(",") if x.strip()]
        cc = [x.strip() for x in self.var_cc.get().split(",") if x.strip()]
        subject = self.var_subject.get().strip()
        query = self.var_fresh_q.get().strip()
        if not to or not subject or not query:
            messagebox.showwarning("Draft Fresh", "Please provide To, Subject, and Intent/Instructions.")
            self.task.done()
            return

        provider = self.var_provider.get().strip() or "vertex"
        tokens = int(self.var_fresh_tokens.get())
        temp = float(self.var_temp.get())
        include_attachments = bool(self.var_fresh_attach.get())

        self._sync_settings_from_ui()
        self.settings.save()

        self.pb_fresh.start(10)
        self.txt_fresh.delete("1.0", tk.END)
        self.status_var.set("Drafting fresh email… (see Logs for details)")

        def worker():
            try:
                # High-level fresh drafting; handles retrieval + eml compose. :contentReference[oaicite:9]{index=9}
                result = processor.draft_fresh_email_eml(
                    export_root=root,
                    provider=provider,
                    to_list=to,
                    cc_list=cc,
                    subject=subject,
                    query=query,
                    sim_threshold=float(self.var_sim.get()),
                    target_tokens=tokens,
                    temperature=temp,
                    include_attachments=include_attachments,
                    sender=None,
                    reply_to=None,
                )
                eml_bytes = result.get("eml_bytes", b"")
                draft_json = result.get("draft_json", {})
                body = (draft_json.get("final_draft") or {}).get("email_draft", "")
                def update():
                    self._last_fresh_bytes = eml_bytes
                    self._last_fresh_meta = result
                    self.txt_fresh.insert(tk.END, body or "(no body)")
                    self.status_var.set("Fresh email draft ready.")
                self.after(0, update)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Draft Fresh failed", str(e)))
            finally:
                self.after(0, lambda: self.pb_fresh.stop())
                self.task.done()

        threading.Thread(target=worker, daemon=True).start()

    def _save_eml_fresh(self) -> None:
        if not self._last_fresh_bytes:
            messagebox.showinfo("Save .eml", "Nothing to save. Generate a draft first.")
            return
        p = filedialog.asksaveasfilename(title="Save fresh .eml", defaultextension=".eml",
                                         filetypes=[("Email files", "*.eml"), ("All files", "*.*")])
        if not p:
            return
        try:
            Path(p).write_bytes(self._last_fresh_bytes)
            messagebox.showinfo("Save .eml", f"Saved:\n{p}")
        except Exception as e:
            messagebox.showerror("Save .eml", f"Failed to save: {e}")

    # ---- chat

    def _on_chat(self) -> None:
        if not self.task.start():
            messagebox.showwarning("Busy", "Another task is running.")
            return
        root, ix_dir = self._with_root_and_index()
        if not root:
            self.task.done()
            return

        q = self.var_chat_q.get().strip()
        if not q:
            messagebox.showwarning("Chat", "Please enter a question.")
            self.task.done()
            return

        provider = self.var_provider.get().strip() or "vertex"
        k = int(self.var_chat_k.get())
        temp = float(self.var_temp.get())

        self.pb_chat.start(10)
        self.txt_chat.delete("1.0", tk.END)
        self.status_var.set("Retrieving context…")

        def worker():
            try:
                # 1) Retrieve context (same search core)  :contentReference[oaicite:10]{index=10}
                ctx = processor._search(ix_dir, q, k=k, provider=provider, conv_id_filter=None)
                # 2) Chat over context (structured output)  :contentReference[oaicite:11]{index=11}
                ans = processor.chat_with_context(q, ctx, chat_history=None, temperature=temp)
                txt = ans.get("answer", "")
                cits = ans.get("citations", [])
                miss = ans.get("missing_information", [])
                def update():
                    out = [txt, ""]
                    if cits:
                        out.append("Citations:")
                        for c in cits:
                            out.append(f" - {c.get('document_id')}: {c.get('fact_cited')}")
                        out.append("")
                    if miss:
                        out.append("Missing Information:")
                        for m in miss:
                            out.append(f" - {m}")
                    self.txt_chat.insert(tk.END, "\n".join(out))
                    self.status_var.set("Answer ready.")
                self.after(0, update)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Chat failed", str(e)))
            finally:
                self.after(0, lambda: self.pb_chat.stop())
                self.task.done()

        threading.Thread(target=worker, daemon=True).start()

    # ---- index

    def _on_build_index(self) -> None:
        if not self.task.start():
            messagebox.showwarning("Busy", "Another task is running.")
            return
        root, ix_dir = self._with_root_and_index()
        if not root:
            self.task.done()
            return

        provider = self.var_provider.get().strip() or "vertex"
        batch = int(self.var_batch.get())
        force = bool(self.var_force.get())
        limit = int(self.var_limit.get() or 0)

        self.pb_index.start(10)
        self.status_var.set("Building/updating index… (see Logs for detailed progress)")

        # email_indexer exposes a CLI-oriented main(). We call it in-process with sys.argv
        def worker():
            try:
                argv = [
                    "email_indexer",
                    "--root", str(root),
                    "--provider", provider,
                    "--batch", str(batch),
                ]
                if force:
                    argv.append("--force-reindex")
                if limit and limit > 0:
                    argv.extend(["--limit", str(limit)])

                old_argv = sys.argv
                try:
                    sys.argv = argv
                    # This executes the build/update routine and logs its progress. :contentReference[oaicite:12]{index=12}
                    email_indexer.main()
                finally:
                    sys.argv = old_argv
                self.after(0, lambda: self.status_var.set("Index update complete."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Index", str(e)))
            finally:
                self.after(0, lambda: self.pb_index.stop())
                self.task.done()

        threading.Thread(target=worker, daemon=True).start()

    # ---- analyze thread

    def _on_analyze_thread(self) -> None:
        if not self.task.start():
            messagebox.showwarning("Busy", "Another task is running.")
            return

        thread_dir = Path(self.var_thread_dir.get().strip()).expanduser()
        ok, msg = validate_directory_path(thread_dir, must_exist=True, allow_parent_traversal=False)
        if not ok:
            messagebox.showerror("Analyze", msg)
            self.task.done()
            return

        provider = self.var_provider.get().strip() or "vertex"
        temp = float(self.var_temp.get())

        self.pb_analyze.start(10)
        self.txt_analyze.delete("1.0", tk.END)
        self.status_var.set("Analyzing thread… (see Logs for details)")

        def worker():
            try:
                # High-level analyzer: reads Conversation.txt and returns a rich dict. :contentReference[oaicite:13]{index=13}
                data = summarizer.analyze_conversation_dir(
                    thread_dir=thread_dir,
                    catalog=None,
                    provider=provider,
                    temperature=temp,
                    merge_manifest=True,
                )
                # Pretty-print summary in Markdown-ish text
                try:
                    md = summarizer.format_analysis_as_markdown(data)  # :contentReference[oaicite:14]{index=14}
                except Exception:
                    md = json.dumps(data, ensure_ascii=False, indent=2)
                self.after(0, lambda: self.txt_analyze.insert(tk.END, md))
                self.after(0, lambda: self.status_var.set("Analysis complete."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Analyze", str(e)))
            finally:
                self.after(0, lambda: self.pb_analyze.stop())
                self.task.done()

        threading.Thread(target=worker, daemon=True).start()

    # ----------------------------- Log pump ------------------------------------

    def _drain_logs(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.txt_logs.insert(tk.END, msg + "\n")
                self.txt_logs.see(tk.END)
        except queue.Empty:
            pass
        # Poll again
        self.after(100, self._drain_logs)


# --------------------------------- Entrypoint ---------------------------------

def main() -> None:
    # Optional: allow headless config or future flags
    _ = argparse.ArgumentParser(add_help=False)
    app = EmailOpsApp()
    app.mainloop()

if __name__ == "__main__":
    main()
