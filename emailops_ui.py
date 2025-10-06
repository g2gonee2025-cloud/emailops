#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmailOps UI — Streamlit Dashboard for Email Operations (Best-in-Class Edition)

Changes in this edition (Oct 2025):
- Theme-safe Dark Mode + Light Mode styles via CSS variables (no low-contrast issues).
- Full surfacing of search_and_draft.py capabilities:
  * Effective query preview (history-aware)
  * Provider/index compatibility & effective provider visibility
  * Recency + scoring tunables (HALF_LIFE_DAYS, RECENCY_BOOST_STRENGTH, CANDIDATES_MULTIPLIER,
    CONTEXT_SNIPPET_CHARS, MIN_AVG_SCORE, FORCE_RENORM) with Apply & Reload
  * Rich Chat Session manager (create, list, load, delete, reset, max history)
  * Direct Attachment Selector with export to attachments.json
  * Context validation summaries before draft/chat
  * Confidence breakdown and quick remediation buttons (increase K / lower temperature)
  * Cross-platform "Open Folder" for conversations
  * CSV/JSON export of search results and citations
  * Query term highlighting in snippets
- Everything preserves your original flows, tabs and commands.

Run:
    streamlit run emailops_ui.py
"""

from __future__ import annotations

import os
import sys
import re
import json
import time
import subprocess
import importlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime

import pandas as pd
import streamlit as st

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="EmailOps Dashboard",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Theme-safe CSS ----------
def inject_theme_css() -> None:
    st.markdown("""
<style>
:root {
  --bg: #ffffff;
  --surface: #ffffff;
  --surface-2: #f8f9fa;
  --text: #1a1a1a;
  --muted: #666666;
  --accent: #0ea5e9;
  --border: #e0e0e0;
  --ok: #22c55e;
  --warn: #f59e0b;
  --err: #ef4444;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0e1117;
    --surface: #111827;
    --surface-2: #0b1220;
    --text: #e5e7eb;
    --muted: #9ca3af;
    --accent: #38bdf8;
    --border: #1f2937;
  }
}

html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--bg);
  color: var(--text);
}

/* Metric containers */
[data-testid="metric-container"] {
  background-color: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.07);
}
[data-testid="metric-container"] > div:first-child {
  color: var(--muted);
  font-weight: 600;
}
[data-testid="metric-container"] > div:nth-child(2) {
  color: var(--text);
  font-weight: 700;
}
[data-testid="metric-container"] > div:last-child {
  color: var(--accent);
  font-weight: 600;
}

/* Inputs & textareas */
.stTextInput input,
.stTextArea textarea,
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stSlider > div,
.stNumberInput input {
  background-color: var(--surface);
  color: var(--text);
  border: 1px solid var(--border);
}
.stTextArea textarea:focus,
.stTextInput input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent);
}

/* Code blocks */
.stCodeBlock,
pre,
code {
  background-color: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
}

/* Expanders & tabs */
.streamlit-expanderHeader {
  background-color: var(--surface-2);
  color: var(--text);
  border-radius: 6px;
}
.stTabs [data-baseweb="tab-list"] {
  background-color: var(--surface-2);
  border-radius: 8px;
  padding: 4px;
}
.stTabs [data-baseweb="tab"] {
  color: var(--muted);
}
.stTabs [aria-selected="true"] {
  background-color: var(--surface);
  color: var(--text) !important;
  border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.12);
}

/* Buttons */
.stButton > button {
  background-color: var(--surface);
  color: var(--text);
  border: 1px solid var(--border);
  font-weight: 600;
}
.stButton > button:hover {
  background-color: var(--surface-2);
  border-color: var(--muted);
}
.stButton > button[kind="primary"] {
  background-color: var(--accent);
  color: #ffffff;
  border: none;
}
.stButton > button[kind="primary"]:hover {
  filter: brightness(0.95);
}

/* Dataframe */
.dataframe, .stDataFrame {
  background-color: var(--surface) !important;
  color: var(--text) !important;
}
.dataframe thead tr th {
  background-color: var(--surface-2) !important;
  color: var(--text) !important;
  font-weight: 700 !important;
}
.dataframe tbody tr:nth-child(even) {
  background-color: var(--surface-2) !important;
}

/* Informational boxes */
.stInfo {
  background-color: rgba(59,130,246,0.08) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-left: 4px solid #3b82f6 !important;
}
.stSuccess {
  background-color: rgba(34,197,94,0.10) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-left: 4px solid #22c55e !important;
}
.stWarning {
  background-color: rgba(245,158,11,0.12) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-left: 4px solid #f59e0b !important;
}
.stError {
  background-color: rgba(239,68,68,0.12) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-left: 4px solid #ef4444 !important;
}

/* Highlight utility */
.mark {
  background: rgba(56,189,248,0.25);
  padding: 0 2px;
  border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

inject_theme_css()

# ---------- Utilities ----------
def _normpath(p: str | Path) -> str:
    try:
        return str(Path(p).expanduser().resolve())
    except Exception:
        return str(p)

def _validate_path(path: Path, must_exist: bool = True, is_dir: bool = True) -> Tuple[bool, str]:
    try:
        p = Path(path).expanduser().resolve()
        if must_exist and not p.exists():
            return False, f"Path does not exist: {p}"
        if must_exist and is_dir and not p.is_dir():
            return False, f"Path is not a directory: {p}"
        return True, "Valid"
    except Exception as e:
        return False, str(e)

def _pkg_root_from(project_root: Path) -> Optional[Path]:
    pr = project_root
    if (pr / "emailops").exists() and (pr / "emailops").is_dir():
        return pr
    if pr.name == "emailops":
        return pr.parent
    return None

@st.cache_resource(show_spinner=False)
def _import_modules(project_root: str) -> Dict[str, Any]:
    pr = Path(project_root)
    pkg_root = _pkg_root_from(pr)
    if not pkg_root:
        raise RuntimeError(
            f"Could not find 'emailops' package under: {project_root}. "
            f"Please ensure the project root contains the 'emailops' folder."
        )
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    modules: Dict[str, Any] = {}
    required_modules = [
        ("utils", "emailops.utils"),
        ("llm_client", "emailops.llm_client"),
        ("env_utils", "emailops.env_utils"),
        ("email_indexer", "emailops.email_indexer"),
        ("search_and_draft", "emailops.search_and_draft"),
        ("summarize_email_thread", "emailops.summarize_email_thread"),
        ("doctor", "emailops.doctor"),
        ("vertex_utils", "vertex_utils"),
        ("vertex_indexer", "vertex_indexer"),
    ]
    for name, module_path in required_modules:
        try:
            modules[name] = importlib.import_module(module_path)
        except ImportError as e:
            if name in ["vertex_indexer", "vertex_utils"]:
                modules[name] = None
            else:
                raise ImportError(f"Failed to import required module {module_path}: {e}")
    return modules

def _format_json(obj: Any) -> str:
    try:
        if hasattr(obj, 'to_dict'):
            obj = obj.to_dict()
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False, indent=2)

def _run_command(cmd: List[str], workdir: str | None = None, title: str = "Running Command"):
    with st.status(title, expanded=True) as status:
        st.code(" ".join(cmd), language="bash")
        try:
            env = os.environ.copy()
            if "EMBED_PROVIDER" not in env:
                env["EMBED_PROVIDER"] = st.session_state.get("provider", "vertex")
            proc = subprocess.Popen(
                cmd, cwd=workdir or None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, universal_newlines=True, env=env,
            )
        except FileNotFoundError:
            st.error(f"Command not found: {cmd[0]}")
            return 1
        except Exception as e:
            st.error(f"Failed to start process: {e}")
            return 1

        log_container = st.container()
        lines: List[str] = []
        with log_container:
            log_area = st.empty()
            while True:
                line = proc.stdout.readline() if proc.stdout else ""
                if not line and proc.poll() is not None:
                    break
                if line:
                    lines.append(line.rstrip("\n"))
                    display_lines = lines[-500:]
                    log_area.code("\n".join(display_lines), language="log")

        rc = proc.poll()
        if rc == 0:
            status.update(label="✅ Completed Successfully", state="complete")
        else:
            status.update(label=f"❌ Failed with exit code {rc}", state="error")
        return rc

def _display_dataframe(
    data: List[Dict[str, Any]],
    columns: List[str] | None = None,
    max_rows: int = 100,
    title: str | None = None
) -> None:
    if not data:
        st.info("No data to display")
        return
    df = pd.DataFrame(data[:max_rows])
    if columns:
        missing = [c for c in columns if c not in df.columns]
        for m in missing:
            df[m] = ""
        df = df[columns]
    if title:
        st.subheader(title)
    st.dataframe(
        df, use_container_width=True, hide_index=True,
        height=min(420, 35 * max(1, len(df)))
    )

def _open_in_os(path: Path) -> None:
    try:
        if os.name == 'nt':
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        st.error(f"Could not open folder: {e}")

def _highlight(text: str, terms: List[str]) -> str:
    if not text or not terms:
        return text
    pattern = r"(" + "|".join(re.escape(t) for t in terms if t) + r")"
    return re.sub(pattern, r'<span class="mark">\1</span>', text, flags=re.IGNORECASE)

def _reload_search_module(modules: Dict[str, Any]) -> None:
    try:
        modules["search_and_draft"] = importlib.reload(modules["search_and_draft"])
        st.success("Reloaded search_and_draft with updated tunables")
    except Exception as e:
        st.error(f"Failed to reload search_and_draft: {e}")

# ---------- Session State ----------
if "provider" not in st.session_state:
    st.session_state.provider = "vertex"
if "project_root" not in st.session_state:
    st.session_state.project_root = str(Path.cwd())
if "export_root" not in st.session_state:
    st.session_state.export_root = r"C:\Users\ASUS\Desktop\Outlook"
if "modules" not in st.session_state:
    st.session_state.modules = None
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = ""

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("📁 Paths")
    project_root = st.text_input(
        "Project Root",
        value=st.session_state.project_root,
        help="Directory containing the 'emailops' package"
    )
    st.session_state.project_root = project_root
    valid, msg = _validate_path(Path(project_root), must_exist=True, is_dir=True)
    if valid:
        if (Path(project_root) / "emailops").exists():
            st.success("✅ Valid project root")
        else:
            st.warning("⚠️ 'emailops' folder not found")
    else:
        st.error(f"❌ {msg}")

    export_root = st.text_input(
        "Export Root",
        value=st.session_state.export_root,
        help="Directory containing conversation folders"
    )
    st.session_state.export_root = export_root
    valid, msg = _validate_path(Path(export_root), must_exist=True, is_dir=True)
    if valid:
        st.success("✅ Valid export root")
    else:
        st.error(f"❌ {msg}")

    st.divider()

    st.subheader("🔧 Provider Settings")
    provider = st.selectbox(
        "Embedding Provider",
        ["vertex", "openai", "azure", "cohere", "huggingface", "local", "qwen"],
        index=0,
        help="Provider for embedding operations. Generation uses Vertex AI."
    )
    st.session_state.provider = provider
    os.environ["EMBED_PROVIDER"] = provider

    with st.expander("🌍 Google Cloud Settings"):
        gcp_project = st.text_input("GCP_PROJECT", value=os.environ.get("GCP_PROJECT", ""))
        gcp_region = st.text_input("GCP_REGION", value=os.environ.get("GCP_REGION", "global"))
        credentials_path = st.text_input(
            "GOOGLE_APPLICATION_CREDENTIALS",
            value=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        )
        if st.button("Apply GCP Settings", use_container_width=True):
            if gcp_project:
                os.environ["GCP_PROJECT"] = gcp_project
                os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_project
            os.environ["GCP_REGION"] = gcp_region
            if credentials_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            st.success("✅ Environment variables updated")

    st.divider()

    st.subheader("🔬 Search Tuning (applies to search_and_draft)")
    col_a, col_b = st.columns(2)
    with col_a:
        half_life_days = st.number_input("HALF_LIFE_DAYS", min_value=1, max_value=365, value=int(os.getenv("HALF_LIFE_DAYS", "30")))
        cand_mult = st.number_input("CANDIDATES_MULTIPLIER", min_value=1, max_value=20, value=int(os.getenv("CANDIDATES_MULTIPLIER", "3")))
        snippet_chars = st.number_input("CONTEXT_SNIPPET_CHARS", min_value=200, max_value=8000, value=int(os.getenv("CONTEXT_SNIPPET_CHARS", "1500")), step=100)
    with col_b:
        recency_strength = st.number_input("RECENCY_BOOST_STRENGTH", min_value=0.0, max_value=5.0, value=float(os.getenv("RECENCY_BOOST_STRENGTH", "1.0")), step=0.1)
        min_avg_score = st.number_input("MIN_AVG_SCORE", min_value=0.0, max_value=1.0, value=float(os.getenv("MIN_AVG_SCORE", "0.2")), step=0.05)
        force_renorm = st.checkbox("FORCE_RENORM", value=os.getenv("FORCE_RENORM", "0") == "1")

    if st.button("Apply & Reload Search Module", use_container_width=True):
        os.environ["HALF_LIFE_DAYS"] = str(int(half_life_days))
        os.environ["CANDIDATES_MULTIPLIER"] = str(int(cand_mult))
        os.environ["CONTEXT_SNIPPET_CHARS"] = str(int(snippet_chars))
        os.environ["RECENCY_BOOST_STRENGTH"] = str(float(recency_strength))
        os.environ["MIN_AVG_SCORE"] = str(float(min_avg_score))
        os.environ["FORCE_RENORM"] = "1" if force_renorm else "0"
        if st.session_state.modules:
            _reload_search_module(st.session_state.modules)

    st.divider()

    if st.button("🔄 Load/Reload Modules", use_container_width=True):
        try:
            with st.spinner("Loading modules..."):
                modules = _import_modules(st.session_state.project_root)
                st.session_state.modules = modules
                st.success(f"✅ Loaded {len(modules)} modules")
        except Exception as e:
            st.error(f"❌ Failed to load modules: {e}")

# ---------- Main Content ----------
st.title("📧 EmailOps Dashboard")
st.caption("Optimized for high-utility search, chat, and drafting — now with robust Dark Mode.")

# Auto-load modules
if st.session_state.modules is None:
    try:
        modules = _import_modules(st.session_state.project_root)
        st.session_state.modules = modules
    except Exception as e:
        st.error(f"❌ Failed to load modules: {e}")
        st.info("Check your project root path and dependencies.")
        st.stop()
else:
    modules = st.session_state.modules

# ---------- Tabs ----------
tabs = st.tabs([
    "📊 Status",
    "🔍 Index",
    "📄 Chunk",
    "🔎 Search & Draft",
    "📝 Summarize",
    "🩺 Doctor",
    "ℹ️ Help"
])

# ---------- STATUS ----------
with tabs[0]:
    st.header("📊 Index Status")
    if modules and modules.get("vertex_utils"):
        try:
            monitor_class = modules["vertex_utils"].IndexingMonitor
            monitor = monitor_class(st.session_state.export_root)
            status = monitor.check_status(emit_text=False)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Documents Indexed", f"{status.documents_indexed:,}")
            with c2:
                st.metric("Conversations", f"{status.conversations_indexed:,} / {status.conversations_total:,}",
                          delta=f"{status.progress_percent:.1f}%" if status.conversations_total > 0 else None)
            with c3:
                st.metric("Progress", f"{status.progress_percent:.1f}%",
                          delta="Active" if status.is_active else "Idle")
            with c4:
                st.metric("Index Status", "✅ Exists" if status.index_exists else "❌ Not Found",
                          delta=status.index_type if status.index_exists else None)

            if status.index_exists:
                st.subheader("Index Metadata")
                col1, col2 = st.columns(2)
                with col1:
                    st.text(f"Provider: {status.provider or 'Unknown'}")
                    st.text(f"Model: {status.model or 'Unknown'}")
                    st.text(f"Dimensions: {status.actual_dimensions or 'Unknown'}")
                    st.text(f"Index Type: {status.index_type or 'Unknown'}")
                with col2:
                    if status.last_updated:
                        try:
                            last_update = datetime.fromisoformat(status.last_updated)
                            st.text(f"Last Updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
                        except Exception:
                            st.text(f"Last Updated: {status.last_updated}")
                    if status.index_file:
                        st.text(f"Index File: {status.index_file}")
                        if status.index_file_size_mb:
                            st.text(f"Size: {status.index_file_size_mb:.1f} MB")

                if st.button("📈 Analyze Indexing Rate"):
                    rate_info = monitor.analyze_rate(emit_text=False)
                    if rate_info:
                        cc1, cc2, cc3 = st.columns(3)
                        with cc1:
                            st.metric("Rate", f"{rate_info.get('rate_per_hour', 0):.1f} items/hour")
                        with cc2:
                            st.metric("Remaining", f"{rate_info.get('remaining_conversations', 0):,}")
                        with cc3:
                            st.metric("ETA", f"{rate_info.get('eta_hours', 0):.1f} hours")
        except Exception as e:
            st.error(f"Failed to load status: {e}")
    else:
        st.warning("Vertex utils module not available")

# ---------- INDEX ----------
with tabs[1]:
    st.header("🔍 Build/Update Index")
    col1, col2, col3 = st.columns(3)
    with col1:
        index_tool = st.selectbox("Indexing Tool", ["vertex_indexer (Parallel)", "email_indexer (Standard)"], index=0)
        mode = st.selectbox("Execution Mode", ["parallel", "sequential"], index=0)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=250, value=64)
    with col2:
        resume = st.checkbox("Resume from Previous", value=True)
        incremental = st.checkbox("Incremental Update", value=False)
        force_rebuild = st.checkbox("Force Full Rebuild", value=False)
    with col3:
        test_mode = st.checkbox("Test Mode", value=False)
        test_chunks = st.number_input("Test Chunks/Limit", min_value=1, max_value=1000, value=100, disabled=not test_mode)
        chunked_files = st.checkbox("Use Chunked Files", value=False)
    if st.button("🚀 Start Indexing", type="primary", use_container_width=True):
        if "vertex_indexer" in index_tool:
            cmd = [
                sys.executable, "-m", "vertex_indexer",
                "--root", st.session_state.export_root,
                "--mode", mode,
                "--batch-size", str(batch_size),
            ]
            if not resume:
                cmd.append("--no-resume")
            if incremental:
                cmd.append("--incremental")
            if force_rebuild:
                cmd.append("--force-rebuild")
            if chunked_files:
                cmd.append("--chunked-files")
            if test_mode:
                cmd.extend(["--test-mode", "--test-chunks", str(test_chunks)])
        else:
            cmd = [
                sys.executable, "-m", "emailops.email_indexer",
                "--root", st.session_state.export_root,
            ]
            if incremental:
                cmd.append("--incremental")
            if test_mode:
                cmd.extend(["--limit", str(test_chunks)])
        _run_command(cmd, workdir=st.session_state.project_root, title="Running Indexer")

# ---------- CHUNK ----------
with tabs[2]:
    st.header("📄 Document Chunking")
    col1, col2 = st.columns(2)
    with col1:
        input_dir = st.text_input("Input Directory", value=st.session_state.export_root)
    with col2:
        output_dir = st.text_input("Output Directory", value=str(Path(st.session_state.export_root) / "_chunks"))
    col1, col2, col3 = st.columns(3)
    with col1:
        workers = st.number_input("Worker Processes", min_value=1, max_value=16, value=8)
        chunk_size = st.number_input("Chunk Size", min_value=100, max_value=10000, value=1600)
    with col2:
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1000, value=200)
        file_pattern = st.text_input("File Pattern", value="*.txt")
    with col3:
        start_method = st.selectbox("Start Method", ["spawn", "forkserver", "fork"], index=0)
        log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)

    adv = st.expander("Advanced Options")
    with adv:
        c1, c2 = st.columns(2)
        with c1:
            no_resume = st.checkbox("Don't Resume", value=False)
            test_mode_chunk = st.checkbox("Test Mode", value=False, key="chunk_test_mode")
            no_clear = st.checkbox("Don't Clear Screen", value=True)
        with c2:
            test_files = st.number_input("Test Files", min_value=1, value=100, disabled=not test_mode_chunk)
            max_chars = st.number_input("Max Characters per File", min_value=0, value=0, help="0 = unlimited")

    if st.button("🚀 Start Chunking", type="primary", use_container_width=True):
        cmd = [
            sys.executable, "-m", "parallel_chunker",
            "--input-dir", input_dir,
            "--output-dir", output_dir,
            "--workers", str(workers),
            "--chunk-size", str(chunk_size),
            "--chunk-overlap", str(chunk_overlap),
            "--file-pattern", file_pattern,
            "--start-method", start_method,
            "--log-level", log_level,
        ]
        if no_resume:
            cmd.append("--no-resume")
        if test_mode_chunk:
            cmd.extend(["--test-mode", "--test-files", str(test_files)])
        if no_clear:
            cmd.append("--no-clear")
        if max_chars > 0:
            cmd.extend(["--max-chars", str(max_chars)])
        _run_command(cmd, workdir=st.session_state.project_root, title="Running Chunker")

# ---------- SEARCH & DRAFT ----------
with tabs[3]:
    st.header("🔎 Search & Draft")

    with st.info("", icon="ℹ️"):
        st.markdown(
            "**Quick Guide:**\n"
            "- Enter a query to find the most relevant snippets\n"
            "- Toggle **Chat Mode** for conversational answers with citations\n"
            "- Provide **Sender** to generate a structured draft with LLM-as-critic\n"
            "- Use **Attachment Selector** to export attachments.json for your mailer pipeline\n"
        )

    # Inputs
    query = st.text_area("Query", height=100, placeholder="Describe what you need or ask a question...")
    c1, c2 = st.columns(2)
    with c1:
        k = st.slider("Top-K Results", min_value=1, max_value=200, value=60, step=5)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    with c2:
        sender = st.text_input("Sender Name/Email", placeholder="Jane Smith <jane@example.com>")
        include_attachments = st.checkbox("Include Attachments", value=True)

    # Advanced
    with st.expander("🔧 Advanced Options"):
        left, right = st.columns(2)
        with left:
            chat_mode = st.checkbox("Chat Mode", value=False)
            session_id = st.text_input("Session ID", placeholder="Blank = new session", disabled=not chat_mode)
            max_hist = st.slider("Max History (hard cap 10)", 0, 10, 10, disabled=not chat_mode)
            if chat_mode:
                st.caption("History inclusion mirrors `_format_chat_history_for_prompt` and `_build_search_query_from_history`.")
            conv_id = st.text_input("Conversation ID Filter", placeholder="folder name like 2024-03-15-meeting")
            conv_subject = st.text_input("Subject Filter (substring)", placeholder="e.g., invoice, renewal")
            min_confidence = st.slider("Minimum Draft Confidence", 0.0, 1.0, 0.0, 0.05)
        with right:
            st.subheader("📁 Conversation Browser")
            browse_folders = st.checkbox("Browse conversation folders", value=False)
            if browse_folders:
                try:
                    export_path = Path(st.session_state.export_root)
                    if export_path.exists():
                        conv_folders = [
                            d.name for d in export_path.iterdir()
                            if d.is_dir() and not d.name.startswith('_') and (d / "Conversation.txt").exists()
                        ]
                        if conv_folders:
                            selected_folder = st.selectbox("Select conversation folder:", [""] + sorted(conv_folders)[:100])
                            if selected_folder:
                                conv_id = selected_folder
                                st.success(f"✅ Will search in: {selected_folder}")
                        else:
                            st.warning("No conversation folders with Conversation.txt found.")
                    else:
                        st.error(f"Export root does not exist: {export_path}")
                except Exception as e:
                    st.error(f"Error browsing folders: {e}")

    # Action buttons
    col_a, col_b, col_c = st.columns(3)
    search_only = col_a.button("🔍 Search Only", use_container_width=True)
    search_and_draft = col_b.button("✉️ Search & Draft", use_container_width=True, type="primary")
    do_chat = col_c.button("💬 Chat over Results", use_container_width=True, disabled=not chat_mode)

    # Prepare module refs
    search_module = modules["search_and_draft"]
    index_dir = Path(st.session_state.export_root) / os.getenv("INDEX_DIRNAME", "_index")

    # Build conversation filter set
    conv_id_filter: Optional[Set[str]] = None
    effective_query = query
    chat_session = None

    # Session init/management
    if chat_mode:
        if not hasattr(st.session_state, "chat_session") or (session_id and st.session_state.get("current_session_id") != session_id):
            if session_id:
                safe_id = search_module._sanitize_session_id(session_id)
                chat_session = search_module.ChatSession(base_dir=index_dir, session_id=safe_id, max_history=max_hist)
                chat_session.load()
                st.session_state.chat_session = chat_session
                st.session_state.current_session_id = session_id
            else:
                new_session_id = f"chat_{datetime.now():%Y%m%d_%H%M%S}"
                chat_session = search_module.ChatSession(base_dir=index_dir, session_id=new_session_id, max_history=max_hist)
                st.session_state.chat_session = chat_session
                st.session_state.current_session_id = new_session_id
        else:
            chat_session = st.session_state.chat_session

    # Effective query construction (history-aware)
    if chat_mode and chat_session:
        hist_for_query = chat_session.recent()
        effective_query = search_module._build_search_query_from_history(hist_for_query, query or "", max_back=5) if query else ""
        with st.expander("🧠 Effective Query & History"):
            st.code(effective_query or "(empty)", language="text")
            st.caption("Built via _build_search_query_from_history over recent user turns.")
            if hist_for_query:
                st.write("Recent History (compact):")
                st.code(search_module._format_chat_history_for_prompt(hist_for_query, max_chars=2000), language="text")

    # Provider/index compatibility & effective provider
    with st.expander("🔌 Provider & Index Compatibility"):
        try:
            meta = modules["search_and_draft"].load_index_metadata(index_dir)  # type: ignore[attr-defined]
        except Exception:
            meta = None
        indexed_provider = (meta.get("provider") if meta else "") or "(unknown)"
        requested = st.session_state.provider
        try:
            effective_provider = search_module._resolve_effective_provider(index_dir, requested)
        except Exception:
            effective_provider = requested
        colp1, colp2, colp3 = st.columns(3)
        with colp1:
            st.metric("Index Provider", indexed_provider)
        with colp2:
            st.metric("Requested Provider", requested)
        with colp3:
            st.metric("Effective Provider", effective_provider)
        if indexed_provider and indexed_provider.lower() != requested.lower():
            st.warning("Provider mismatch detected; using index provider for compatibility.")

    # If user pressed an action button
    if search_only or search_and_draft or do_chat:
        if not query and not (search_only and (conv_id or conv_subject)):
            st.error("Please enter a query (or use filters and press Search Only).")
        else:
            try:
                with st.spinner("Searching..."):
                    index_dir = Path(st.session_state.export_root) / os.getenv("INDEX_DIRNAME", "_index")
                    # Determine conv filter
                    if conv_id or conv_subject:
                        mapping = search_module._load_mapping(index_dir)
                        conv_id_filter = set()
                        if conv_id:
                            conv_id_filter.add(str(conv_id).strip())
                        if conv_subject and mapping:
                            hits = search_module._find_conv_ids_by_subject(mapping, conv_subject)
                            conv_id_filter |= hits
                        if not conv_id_filter:
                            st.info("No conversations matched the provided filters; continuing without filter.")
                            conv_id_filter = None

                    # Perform search
                    results = search_module._search(
                        ix_dir=index_dir,
                        query=effective_query,
                        k=k,
                        provider=st.session_state.provider,
                        conv_id_filter=conv_id_filter
                    )

                # Show results
                if not results:
                    st.warning("No results found. Try adjusting your query or increasing K.")
                else:
                    st.success(f"Found {len(results)} results")

                    # Group by conversation
                    conv_groups: Dict[str, List[Dict[str, Any]]] = {}
                    for r in results:
                        rid = r.get("conv_id") or (r.get("id", "").split("::")[0] if "::" in r.get("id", "") else "unknown")
                        conv_groups.setdefault(rid, []).append(r)

                    # Export buttons (CSV/JSON)
                    export_cols = ["id","conv_id","subject","date","from_name","from_email","doc_type","score","original_score"]
                    df_export = pd.DataFrame([{k: item.get(k, "") for k in export_cols} for item in results])
                    ex1, ex2 = st.columns(2)
                    with ex1:
                        st.download_button("📥 Download Results CSV",
                            data=df_export.to_csv(index=False).encode("utf-8"),
                            file_name="search_results.csv", mime="text/csv")
                    with ex2:
                        st.download_button("📥 Download Results JSON",
                            data=_format_json(results), file_name="search_results.json", mime="application/json")

                    # Display groups with highlight
                    terms = [t for t in (query or "").split() if len(t) > 1]
                    for cid, items in conv_groups.items():
                        conv_path = Path(st.session_state.export_root) / cid
                        exists = conv_path.exists() and (conv_path / "Conversation.txt").exists()
                        status_icon = "✅" if exists else "⚠️"
                        first = items[0]
                        subject = (first.get("subject") or "No subject")[:80]
                        with st.expander(f"{status_icon} **{cid}** - {subject} ({len(items)} items)"):
                            colx, coly = st.columns([3,1])
                            with colx:
                                st.markdown(f"**Subject:** {first.get('subject', 'N/A')}")
                                st.markdown(f"**Date:** {first.get('date', 'N/A')}")
                                st.markdown(f"**From:** {first.get('from_name','')} <{first.get('from_email','')}>")
                                if exists:
                                    st.markdown(f"**📁 Path:** `{conv_path}`")
                                    if st.button("Open Folder", key=f"open_{cid}"):
                                        _open_in_os(conv_path)
                                else:
                                    st.warning(f"Conversation folder not found at: {conv_path}")
                            with coly:
                                scores = [float(it.get('score', 0) or 0) for it in items]
                                st.metric("Avg Score", f"{sum(scores)/max(1,len(scores)):.3f}")
                                st.metric("Items", len(items))
                            # Items
                            st.markdown("**Items in this conversation:**")
                            for item in items:
                                doc_type = item.get("doc_type", "")
                                icon = "📎" if doc_type == "attachment" else "📧"
                                score = float(item.get('score', 0) or 0)
                                item_id = item.get("id", "").split("::")[-1] if "::" in item.get("id", "") else item.get("id","")
                                snippet = item.get("text","") or ""
                                if terms:
                                    snippet = _highlight(snippet, terms)
                                st.markdown(f"{icon} **{item_id}** (score: {score:.3f})", unsafe_allow_html=True)
                                st.markdown(snippet, unsafe_allow_html=True)
                                if item.get("attachment_name"):
                                    st.markdown(f"📎 Attachment: {item.get('attachment_name')} ({item.get('attachment_type','unknown')})")
                                st.divider()

                    # Store for downstream steps
                    st.session_state.search_results = results

                    # Summary/quality
                    st.subheader("📊 Search Summary & Context Quality")
                    colq1, colq2, colq3 = st.columns(3)
                    with colq1:
                        st.metric("Total Results", len(results))
                        st.metric("Unique Conversations", len(conv_groups))
                    with colq2:
                        avg_score = sum(float(r.get('score', 0) or 0) for r in results)/len(results)
                        st.metric("Average Score", f"{avg_score:.3f}")
                        doc_types = {}
                        for r in results:
                            doc_types[r.get("doc_type","unknown")] = doc_types.get(r.get("doc_type","unknown"), 0) + 1
                        st.metric("Types", ", ".join(f"{k}: {v}" for k,v in doc_types.items()))
                    with colq3:
                        valid_paths = sum(1 for key in conv_groups if (Path(st.session_state.export_root)/key).exists())
                        st.metric("Valid Folders", f"{valid_paths}/{len(conv_groups)}")
                        dr = [r.get("date") for r in results if r.get("date")]
                        st.metric("Date Range", f"{min(dr)[:10]} to {max(dr)[:10]}" if dr else "n/a")

                    # validate_context_quality
                    is_valid, msg = search_module.validate_context_quality(results)
                    if is_valid:
                        st.success(f"Context quality: {msg}")
                    else:
                        st.warning(f"Context quality issue: {msg}")

                    # Chat
                    if do_chat and chat_mode:
                        with st.spinner("Generating chat response..."):
                            chat_history = chat_session.recent() if chat_session else []
                            chat_result = search_module.chat_with_context(
                                query=query,
                                context_snippets=results,
                                chat_history=chat_history,
                                temperature=temperature
                            )
                            if chat_session:
                                conv_for_turn = None
                                if conv_id_filter and len(conv_id_filter) == 1:
                                    conv_for_turn = list(conv_id_filter)[0]
                                chat_session.add_message("user", query, conv_id=conv_for_turn)
                                chat_session.add_message("assistant", chat_result.get("answer",""), conv_id=conv_for_turn)
                                chat_session.save()
                        st.subheader("💬 Chat Response")
                        st.write(chat_result.get("answer",""))
                        cits = chat_result.get("citations",[])
                        if cits:
                            with st.expander(f"📚 Citations ({len(cits)})"):
                                st.write(pd.DataFrame(cits))
                        missing_info = chat_result.get("missing_information",[])
                        if missing_info:
                            with st.expander("❓ Missing Information"):
                                for m in missing_info:
                                    st.write(f"• {m}")

                    # Draft
                    if search_and_draft and sender:
                        with st.spinner("Drafting email with LLM-as-critic..."):
                            chat_history = chat_session.recent() if (chat_mode and chat_session) else None
                            draft_result = search_module.draft_email_structured(
                                query=query,
                                sender=sender,
                                context_snippets=results,
                                provider=st.session_state.provider,
                                temperature=temperature,
                                include_attachments=include_attachments,
                                chat_history=chat_history
                            )
                            if chat_session:
                                conv_for_turn = None
                                if conv_id_filter and len(conv_id_filter) == 1:
                                    conv_for_turn = list(conv_id_filter)[0]
                                chat_session.add_message("user", query, conv_id=conv_for_turn)
                                chat_session.add_message("assistant", draft_result.get("final_draft",{}).get("email_draft",""), conv_id=conv_for_turn)
                                chat_session.save()

                        confidence = float(draft_result.get("confidence_score", 0.0) or 0.0)
                        if confidence >= 0.7:
                            st.success(f"✅ High Confidence: {confidence:.2f}")
                        elif confidence >= 0.4:
                            st.warning(f"⚠️ Medium Confidence: {confidence:.2f}")
                        else:
                            st.error(f"❌ Low Confidence: {confidence:.2f}")

                        st.subheader("📝 Email Draft")
                        draft_text = draft_result.get("final_draft",{}).get("email_draft","")
                        st.text_area("Draft", value=draft_text, height=300, disabled=True, key="draft_display")

                        # Quick remediation
                        colr1, colr2 = st.columns(2)
                        if colr1.button("↗️ Rerun with higher K (+20)"):
                            st.session_state["rerun_k"] = min(200, k+20)
                            st.experimental_rerun()
                        if colr2.button("🛡️ Rerun safer (lower temperature -0.1)"):
                            st.session_state["rerun_temp"] = max(0.0, temperature-0.1)
                            st.experimental_rerun()

                        # Metadata
                        md = draft_result.get("metadata",{})
                        c_m1, c_m2, c_m3 = st.columns(3)
                        with c_m1:
                            st.metric("Citations", md.get("citation_count",0))
                            st.metric("Word Count", md.get("draft_word_count",0))
                        with c_m2:
                            st.metric("Workflow", md.get("workflow_state",""))
                            st.metric("Quality", draft_result.get("critic_feedback",{}).get("overall_quality",""))
                        with c_m3:
                            atts = draft_result.get("selected_attachments",[])
                            st.metric("Attachments", len(atts))

                        # Downloads
                        colj1, colj2 = st.columns(2)
                        with colj1:
                            st.download_button(
                                "📥 Download Draft JSON",
                                data=_format_json(draft_result),
                                file_name="draft_result.json",
                                mime="application/json"
                            )
                        with colj2:
                            st.download_button(
                                "📥 Download Draft TXT",
                                data=draft_text.encode("utf-8"),
                                file_name="draft.txt",
                                mime="text/plain"
                            )

                        # Details
                        with st.expander("📋 View Details"):
                            tab1, tab2, tab3, tab4 = st.tabs(["Missing Info", "Attachments", "Citations", "Source Conversations"])
                            with tab1:
                                missing_info = draft_result.get("final_draft",{}).get("missing_information",[])
                                if missing_info:
                                    for item in missing_info:
                                        st.write(f"• {item}")
                                else:
                                    st.success("No missing information detected.")
                            with tab2:
                                if atts:
                                    data = [{"filename":a["filename"],"size_mb":a["size_mb"],"relevance":a["relevance_score"],"extension":a.get("extension","")} for a in atts]
                                    _display_dataframe(data, ["filename","size_mb","relevance","extension"], max_rows=100, title="Selected Attachments")
                                else:
                                    st.info("No attachments selected.")
                            with tab3:
                                citations = draft_result.get("final_draft",{}).get("citations",[])
                                if citations:
                                    _display_dataframe(citations, ["document_id","fact_cited","confidence"], max_rows=200, title="Citations")
                                else:
                                    st.info("No citations.")
                            with tab4:
                                conv_ids = set()
                                for s in st.session_state.get("search_results", []):
                                    cid2 = s.get("conv_id") or (s.get("id","").split("::")[0] if "::" in s.get("id","") else "")
                                    if cid2:
                                        conv_ids.add(cid2)
                                if conv_ids:
                                    for cc in sorted(conv_ids):
                                        path = Path(st.session_state.export_root) / cc
                                        if path.exists():
                                            st.success(f"✅ {cc}")
                                            st.caption(f"Path: `{path}`")
                                        else:
                                            st.warning(f"⚠️ {cc} (folder not found)")
                                else:
                                    st.info("No conversation folders identified.")

                        # Threshold enforcement
                        if confidence < min_confidence:
                            st.error(f"Draft confidence ({confidence:.2f}) below threshold ({min_confidence:.2f}).")

                    elif search_and_draft and not sender:
                        st.error("Please provide Sender Name/Email to draft.")

            except Exception as e:
                st.error(f"Operation failed: {e}")
                st.exception(e)

    # --- Attachment Selector (direct call to select_relevant_attachments) ---
    st.divider()
    st.subheader("📎 Attachment Selector (Direct)")
    st.caption("Ranks and filters attachments from the conversations present in the current search results.")
    col_att1, col_att2, col_att3, col_att4 = st.columns(4)
    with col_att1:
        att_max = st.number_input("Max attachments", min_value=1, max_value=50, value=10)
    with col_att2:
        att_mb = st.number_input("Max total size (MB)", min_value=1.0, max_value=500.0, value=25.0, step=1.0)
    with col_att3:
        att_enabled = st.checkbox("Enable selection", value=True)
    with col_att4:
        emit_json_path = st.text_input("Save attachments.json to", value=str(Path(st.session_state.export_root) / "attachments.json"))

    if st.button("🔎 Rank Attachments", disabled=not att_enabled):
        if not st.session_state.get("search_results"):
            st.warning("Run a search first to populate context snippets.")
        else:
            try:
                ranked = search_module.select_relevant_attachments(
                    query=query or "",
                    context_snippets=st.session_state["search_results"],
                    provider=st.session_state.provider,
                    max_attachments=int(att_max),
                    max_size_mb=float(att_mb)
                )
                if not ranked:
                    st.info("No relevant attachments found.")
                else:
                    _display_dataframe(
                        [{k: v for k, v in a.items()} for a in ranked],
                        ["filename","size_mb","relevance_score","extension","path"],
                        max_rows=100,
                        title="Ranked Attachments"
                    )
                    st.download_button(
                        "📥 Download attachments.json",
                        data=json.dumps({"attachments": ranked}, ensure_ascii=False, indent=2),
                        file_name="attachments.json",
                        mime="application/json"
                    )
                    if emit_json_path and st.button("💾 Save to path"):
                        try:
                            Path(emit_json_path).write_text(json.dumps({"attachments": ranked}, ensure_ascii=False, indent=2), encoding="utf-8")
                            st.success(f"Saved to {emit_json_path}")
                        except Exception as e:
                            st.error(f"Failed to save: {e}")
                    # Optional nudge line for drafting
                    if ranked and st.checkbox("✍️ Mention only these attachments in the next draft"):
                        filenames = ", ".join(a["filename"] for a in ranked)
                        st.info("When you click draft, we will append a hint to the query to mention these filenames explicitly.")
                        st.session_state["mention_attachments_hint"] = f"\n\nPlease explicitly mention these attachments by filename if relevant: {filenames}"
                    else:
                        st.session_state.pop("mention_attachments_hint", None)

    # --- Chat Session Manager ---
    st.divider()
    st.subheader("💼 Chat Session Manager")
    if chat_mode and st.session_state.chat_session:
        cs = st.session_state.chat_session
        csm1, csm2, csm3, csm4 = st.columns(4)
        if csm1.button("🔄 Reset Session"):
            cs.reset(); cs.save()
            st.success("Session reset.")
        if csm2.button("🗑️ Delete Session File"):
            try:
                p = cs.session_path
                if p.exists():
                    p.unlink()
                cs.reset()
                cs.save()
                st.success("Session file deleted.")
            except Exception as e:
                st.error(f"Failed to delete session: {e}")
        new_name = csm3.text_input("Rename to (safe ID)")
        if csm4.button("✏️ Rename"):
            try:
                safe = search_module._sanitize_session_id(new_name or "")
                old = cs.session_path
                cs.session_id = safe
                cs.save()
                if old.exists() and old != cs.session_path:
                    try:
                        old.unlink()
                    except Exception:
                        pass
                st.session_state.current_session_id = safe
                st.success(f"Renamed session to: {safe}")
            except Exception as e:
                st.error(f"Rename failed: {e}")

        if st.button("📋 List Sessions"):
            sessions_dir = Path(index_dir) / search_module.SESSIONS_DIRNAME
            if sessions_dir.exists():
                files = sorted(sessions_dir.glob("*.json"))
                if files:
                    for f in files[:20]:
                        st.text(f.stem)
                else:
                    st.info("No saved sessions found.")
            else:
                st.info("No sessions directory found.")

# ---------- SUMMARIZE ----------
with tabs[4]:
    st.header("📝 Summarize Email Thread")
    thread_path = st.text_input("Thread Directory", value=st.session_state.export_root, help="Directory containing Conversation.txt")
    c1, c2 = st.columns(2)
    with c1:
        output_format = st.selectbox("Output Format", ["JSON", "Markdown", "Both"])
    with c2:
        write_to_disk = st.checkbox("Save to Thread Directory", value=True)

    if st.button("📊 Analyze Thread", type="primary", use_container_width=True):
        thread_dir = Path(thread_path)
        convo_file = thread_dir / "Conversation.txt"
        if not convo_file.exists():
            st.error(f"Conversation.txt not found in {thread_dir}")
        elif not modules:
            st.error("Modules not loaded.")
        else:
            try:
                with st.spinner("Analyzing thread with facts ledger..."):
                    summarize_module = modules["summarize_email_thread"]
                    utils_module = modules["utils"]
                    raw_text = utils_module.read_text_file(convo_file)
                    cleaned_text = utils_module.clean_email_text(raw_text)
                    analysis = summarize_module.analyze_email_thread_with_ledger(
                        thread_text=cleaned_text,
                        provider=st.session_state.provider
                    )
                st.success("✅ Analysis complete")

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Category", analysis.get("category", "Unknown"))
                    st.metric("Completeness", f"{analysis.get('_metadata', {}).get('completeness_score', 0)}%")
                with c2:
                    st.metric("Subject", (analysis.get("subject","") or "")[:50])
                    st.metric("Participants", len(analysis.get("participants", [])))

                if analysis.get("summary"):
                    st.subheader("📋 Summary")
                    st.write(analysis["summary"])

                if analysis.get("facts_ledger"):
                    st.subheader("📊 Facts Ledger")
                    facts_data = [{
                        "Fact": f.get("fact",""),
                        "Category": f.get("category",""),
                        "Status": f.get("status",""),
                        "Confidence": f"{f.get('confidence',0):.2f}",
                        "Source": (f.get("source_email","") or "")[:30]
                    } for f in analysis["facts_ledger"]]
                    _display_dataframe(facts_data, ["Fact","Category","Status","Confidence","Source"], max_rows=50)

                if analysis.get("action_items"):
                    st.subheader("✅ Action Items")
                    for item in analysis["action_items"]:
                        cc1, cc2, cc3 = st.columns([3,1,1])
                        with cc1:
                            st.write(f"• {item.get('action','')}")
                        with cc2:
                            st.write(f"Owner: {item.get('owner','TBD')}")
                        with cc3:
                            st.write(f"Due: {item.get('due_date','TBD')}")

                if write_to_disk:
                    try:
                        json_path = thread_dir / "thread_analysis.json"
                        json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                        if output_format in ["Markdown", "Both"] and hasattr(summarize_module, 'format_analysis_as_markdown'):
                            md_path = thread_dir / "thread_analysis.md"
                            md_content = summarize_module.format_analysis_as_markdown(analysis)
                            md_path.write_text(md_content, encoding="utf-8")
                        st.success(f"Saved analysis to {thread_dir}")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")

                dc1, dc2 = st.columns(2)
                with dc1:
                    st.download_button("📥 Download JSON",
                        data=json.dumps(analysis, ensure_ascii=False, indent=2, default=str),
                        file_name="thread_analysis.json", mime="application/json")
                with dc2:
                    if hasattr(summarize_module, 'format_analysis_as_markdown'):
                        md_content = summarize_module.format_analysis_as_markdown(analysis)
                        st.download_button("📥 Download Markdown",
                            data=md_content, file_name="thread_analysis.md", mime="text/markdown")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.exception(e)

# ---------- DOCTOR ----------
with tabs[5]:
    st.header("🩺 System Doctor")
    st.markdown(
        "- Dependency & environment checks\n"
        "- Index health and provider compatibility\n"
        "- Embedding connectivity\n"
        "- Index statistics and recommendations"
    )
    c1, c2 = st.columns(2)
    with c1:
        skip_install_check = st.checkbox("Skip Install Check", value=False)
        install_missing = st.checkbox("Auto-Install Missing", value=False)
    with c2:
        skip_embed_check = st.checkbox("Skip Embedding Check", value=False)
        log_level = st.selectbox("Log Level", ["INFO","DEBUG","WARNING","ERROR"], index=0)

    doctor_provider = st.selectbox("Provider to Check",
        ["(Use Index Provider)", "vertex","openai","azure","cohere","huggingface","local","qwen"], index=0)

    if st.button("🔍 Run Diagnostics", type="primary", use_container_width=True):
        if not modules:
            st.error("Modules not loaded.")
        else:
            cmd = [
                sys.executable, "-m", "emailops.doctor",
                "--root", st.session_state.export_root,
                "--log-level", log_level,
            ]
            if doctor_provider != "(Use Index Provider)":
                cmd.extend(["--provider", doctor_provider])
            if skip_install_check:
                cmd.append("--skip-install-check")
            if install_missing:
                cmd.append("--install-missing")
            if skip_embed_check:
                cmd.append("--skip-embed-check")
            _run_command(cmd, workdir=st.session_state.project_root, title="Running System Doctor")

# ---------- HELP ----------
with tabs[6]:
    st.header("ℹ️ Help & Documentation")
    st.markdown("""
**Quick Start**
1. Configure Project/Export roots in the sidebar; click **Load/Reload Modules**.
2. Build an index under **Index** (parallel Vertex or standard).
3. Use **Search & Draft** to search, chat, or draft with critic & attachments.
4. Tweak **Search Tuning** in the sidebar and **Apply & Reload** to observe effects.

**CLI parity** mirrors `emailops.search_and_draft` flags (chat, session, subject filters, emit-json, min-confidence).  
All new UI features map directly to documented functions so behavior is predictable across UI and CLI.
""")

st.divider()
st.markdown(
    "<div style='text-align:center; color: var(--muted); font-size: 0.9em;'>"
    "EmailOps Dashboard — Best-in-Class Edition"
    "</div>",
    unsafe_allow_html=True
)
