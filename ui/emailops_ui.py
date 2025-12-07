#!/usr/bin/env python3
# ruff: noqa: I001
"""
EmailOps UI — Enhanced Streamlit Dashboard for Email Operations

    from cortex.config.loader import get_config  # Import for configuration
    from cortex.orchestration.graphs import (build_draft_graph,  # Graph building functions
                                             build_summarize_graph)
    from cortex.retrieval.hybrid_search import (KBSearchInput,  # Hybrid search input
                                                tool_kb_search_hybrid)
- Index: One-click incremental or full reindex with streaming logs
- Chunk: Parallel document chunking with progress monitoring
- Search: Query top-K context with boosted recency and scored snippets
- Draft: Structured email draft with critic pass, attachments, and confidence scoring
- Summarize: Facts-ledger analysis for email threads
- Doctor: Comprehensive dependency and compatibility checks

Requirements:
    pip install streamlit pandas

Run:
    streamlit run emailops_ui.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

# ---------- Path Setup ----------
# Add project roots to path to allow importing cortex modules
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root / "backend" / "src"))
sys.path.append(str(project_root / "cli" / "src"))
sys.path.append(str(project_root / "workers" / "src"))

# ---------- Cortex Imports ----------
try:
    from cortex.config.loader import get_config
    from cortex.orchestration.graphs import build_draft_graph, build_summarize_graph
    from cortex.retrieval.hybrid_search import KBSearchInput, tool_kb_search_hybrid

    config = get_config()
    CORTEX_AVAILABLE = True
except ImportError as e:
    logging.getLogger("emailops.ui").error(f"Failed to import Cortex modules: {e}")
    st.error(f"❌ Failed to import Cortex modules: {e}")
    st.info(
        "This usually means a dependency is missing. Try running: pip install langgraph"
    )
    CORTEX_AVAILABLE = False

    # Fallback config
    class FallbackConfig:
        class Core:
            env = "dev"
            provider = "vertex"

        class Index:
            dirname = "_index"

        core = Core()
        INDEX_DIRNAME = "_index"
        CHUNK_DIRNAME = "_chunks"
        DEFAULT_BATCH_SIZE = 64
        DEFAULT_CHUNK_SIZE = 1600
        DEFAULT_CHUNK_OVERLAP = 200

    config = FallbackConfig()


# ---------- Page Configuration ----------
def _load_env_defaults(dotenv_path: Path | None = None) -> None:
    """Populate os.environ with entries from a .env file without overriding existing values."""
    path = dotenv_path or (Path.cwd() / ".env")
    try:
        if not path.exists():
            return
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or key.startswith("#"):
                continue
            if "#" in value:
                value = value.split("#", 1)[0].rstrip()
            if value and (
                (value.startswith('"') and value.endswith('"'))
                or (value.startswith("'") and value.endswith("'"))
            ):
                value = value[1:-1]
            if key not in os.environ:
                os.environ[key] = value
    except Exception as exc:
        logging.getLogger("emailops.ui").warning(
            "Failed to load .env defaults: %s", exc
        )


_load_env_defaults()


st.set_page_config(
    page_title="EmailOps Dashboard",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS for Better UI ----------
st.markdown(
    """
<style>
    /* Quick start card styling */
    .quick-start-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 24px;
        color: white;
        margin-bottom: 20px;
    }

    .quick-start-step {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
        transition: all 0.2s ease;
    }

    .quick-start-step:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }

    .quick-start-step.completed {
        border-color: #22c55e;
        background-color: #f0fdf4;
    }

    .step-number {
        display: inline-block;
        width: 28px;
        height: 28px;
        background-color: #667eea;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 28px;
        font-weight: bold;
        margin-right: 12px;
    }

    .step-number.completed {
        background-color: #22c55e;
    }

    /* Improved metric styling with better contrast */
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Make metric labels more visible */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    [data-testid="metric-container"] > div:first-child {
        color: #666666;
        font-weight: 500;
        font-size: 0.9rem;
    }

    [data-testid="metric-container"] > div:nth-child(2) {
        color: #1a1a1a;
        font-weight: 600;
        font-size: 1.5rem;
    }

    [data-testid="metric-container"] > div:last-child {
        color: #0ea5e9;
        font-weight: 500;
    }

    /* Success/Warning/Error boxes with better visibility */
    .success-box {
        background-color: #f0fdf4;
        border: 1px solid #86efac;
        border-left: 4px solid #22c55e;
        border-radius: 5px;
        padding: 12px;
        margin: 10px 0;
        color: #14532d;
    }

    .warning-box {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #f59e0b;
        border-radius: 5px;
        padding: 12px;
        margin: 10px 0;
        color: #451a03;
    }

    .error-box {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 4px solid #ef4444;
        border-radius: 5px;
        padding: 12px;
        margin: 10px 0;
        color: #450a0a;
    }

    /* Improve text area readability */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #d4d4d4 !important;
    }

    .stTextArea textarea:focus {
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 0 1px #0ea5e9 !important;
    }

    /* Better code block visibility */
    .stCodeBlock {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
    }

    /* Improve selectbox and input visibility */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .stSelectbox > div > div > div {
        color: #000000 !important;
    }
    li[data-baseweb="menu-item"] > div {
        color: black !important;
    }

    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }

    /* Better visibility for expanders */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 5px;
        color: #1a1a1a;
    }

    .streamlit-expanderHeader:hover {
        background-color: #e9ecef;
    }

    /* Better tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #666666;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Improve button visibility */
    .stButton > button {
        background-color: #ffffff;
        color: #1a1a1a;
        border: 1px solid #d4d4d4;
        font-weight: 500;
    }

    .stButton > button:hover {
        background-color: #f8f9fa;
        border-color: #a3a3a3;
    }

    .stButton > button[kind="primary"] {
        background-color: #0ea5e9;
        color: #ffffff;
        border: none;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #0284c7;
    }

    /* Command box styling */
    .command-box {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 16px;
        font-family: 'Consolas', 'Monaco', monospace;
        color: #d4d4d4;
        margin: 10px 0;
    }

    .command-prompt {
        color: #4ec9b0;
    }

    .command-text {
        color: #ce9178;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Logging Utilities ----------
LOG_BUFFER_KEY = "debug_log_buffer"
LOG_BUFFER_MAXLEN = 2000


class StreamlitLogHandler(logging.Handler):
    """Streamlit-aware log handler that buffers records in session state."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            entry = {
                "timestamp": datetime.fromtimestamp(record.created).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "level": record.levelname,
                "logger": record.name,
                "message": message,
            }
            buffer: deque = st.session_state.setdefault(
                LOG_BUFFER_KEY,
                deque(maxlen=LOG_BUFFER_MAXLEN),
            )
            buffer.append(entry)
        except Exception:
            pass


def _ensure_log_handler() -> None:
    """Attach the Streamlit log handler once per session."""
    if LOG_BUFFER_KEY not in st.session_state:
        st.session_state[LOG_BUFFER_KEY] = deque(maxlen=LOG_BUFFER_MAXLEN)

    if st.session_state.get("log_handler_attached"):
        return

    handler = StreamlitLogHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    st.session_state["log_handler_attached"] = True


logger = logging.getLogger("emailops.ui")


# ---------- Utility Functions ----------
def _normpath(p: str | Path) -> str:
    """Normalize and resolve path."""
    try:
        return str(Path(p).expanduser().resolve())
    except Exception:
        return str(p)


def _validate_path(
    path: Path, must_exist: bool = True, is_dir: bool = True
) -> tuple[bool, str]:
    """Validate a path and return status with message."""
    try:
        p = Path(path).expanduser().resolve()
        if must_exist and not p.exists():
            return False, f"Path does not exist: {p}"
        if must_exist and is_dir and not p.is_dir():
            return False, f"Path is not a directory: {p}"
        return True, "Valid"
    except Exception as e:
        return False, str(e)


def _open_path(path: Path) -> None:
    """Open a file/folder cross-platform."""
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        st.error(f"Could not open folder: {e}")


def _format_json(obj: Any) -> str:
    """Format object as JSON string."""
    try:
        if hasattr(obj, "to_dict"):
            obj = obj.to_dict()
        elif hasattr(obj, "model_dump"):
            obj = obj.model_dump()
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False, indent=2)


def _run_command(
    cmd: list[str], workdir: str | None = None, title: str = "Running Command"
):
    """Run a command and stream output."""
    with st.status(title, expanded=True) as status:
        if not cmd or not cmd[0]:
            st.error("Invalid command: empty command provided")
            return 1

        # Display command
        safe_cmd_display = " ".join(cmd)
        st.code(safe_cmd_display, language="bash")

        try:
            env = os.environ.copy()
            # Ensure PYTHONPATH includes our source directories
            python_path = env.get("PYTHONPATH", "")
            paths_to_add = [
                str(project_root / "backend" / "src"),
                str(project_root / "cli" / "src"),
                str(project_root / "workers" / "src"),
            ]
            env["PYTHONPATH"] = os.pathsep.join([*paths_to_add, python_path])

            proc = subprocess.Popen(
                cmd,
                cwd=workdir or None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                env=env,
                shell=False,
            )
        except FileNotFoundError:
            st.error(f"❌ Command not found: {cmd[0]}")
            return 1
        except Exception as e:
            st.error(f"❌ Failed to start process: {e}")
            return 1

        log_container = st.container()
        lines: deque[str] = deque(maxlen=10000)

        with log_container:
            log_area = st.empty()
            try:
                while True:
                    line = proc.stdout.readline() if proc.stdout else ""
                    if not line and proc.poll() is not None:
                        break
                    if line:
                        lines.append(line.rstrip("\n"))
                        display_lines = list(lines)[-500:]
                        log_area.code("\n".join(display_lines), language="log")
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait(timeout=5)
                status.update(label="⚠️ Interrupted by user", state="error")
                return 130
            except Exception as e:
                st.error(f"Error reading output: {e}")

        rc = proc.poll()
        if rc == 0:
            status.update(label="✅ Completed Successfully", state="complete")
        else:
            status.update(label=f"❌ Failed with exit code {rc}", state="error")

        return rc


def _check_setup_status() -> dict[str, bool]:
    """Check the current setup status for Quick Start guidance."""
    status = {
        "project_root_valid": False,
        "export_root_valid": False,
        "gcp_configured": False,
        "cortex_available": False,
        "index_exists": False,
    }

    # Check project root
    pr = st.session_state.get("project_root", "")
    if pr and Path(pr).exists():
        status["project_root_valid"] = True

    # Check export root
    er = st.session_state.get("export_root", "")
    if er and Path(er).exists():
        status["export_root_valid"] = True

    # Check GCP config
    if os.environ.get("GCP_PROJECT") and os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS"
    ):
        status["gcp_configured"] = True

    # Check modules
    if CORTEX_AVAILABLE:
        status["cortex_available"] = True

    # Check index
    if er:
        index_dir = Path(er) / config.INDEX_DIRNAME
        if index_dir.exists() and (index_dir / "mapping.json").exists():
            status["index_exists"] = True

    return status


# ---------- Session State Initialization ----------
if "provider" not in st.session_state:
    st.session_state.provider = "vertex"
if "project_root" not in st.session_state:
    st.session_state.project_root = str(project_root)
if "export_root" not in st.session_state:
    export_env = os.getenv("EMAILOPS_EXPORT_ROOT")
    st.session_state.export_root = (
        export_env.strip() if export_env else r"C:\Users\ASUS\Desktop\OUTLOOK"
    )
if "index_root" not in st.session_state:
    index_env = os.getenv("EMAILOPS_INDEX_ROOT")
    if index_env:
        st.session_state.index_root = index_env.strip()
    else:
        st.session_state.index_root = st.session_state.export_root
if "show_quick_start" not in st.session_state:
    st.session_state.show_quick_start = True

_ensure_log_handler()

# ---------- Sidebar Configuration ----------
with st.sidebar:
    st.header("⚙️ Configuration")

    # Quick Start Toggle
    st.session_state.show_quick_start = st.checkbox(
        "📚 Show Quick Start Guide",
        value=st.session_state.show_quick_start,
        help="Toggle the quick start panel on the main page",
    )

    st.divider()

    # Path Configuration
    st.subheader("📁 Paths")
    project_root_input = st.text_input(
        "Code Project Root",
        value=st.session_state.project_root,
        help="Directory containing the project",
        key="project_root_input",
    )
    st.session_state.project_root = project_root_input

    valid, msg = _validate_path(Path(project_root_input), must_exist=True, is_dir=True)
    if valid:
        st.success("✅ Valid project root")
    else:
        st.error(f"❌ {msg}")

    export_root = st.text_input(
        "Outlook Export Root",
        value=st.session_state.export_root,
        help="Directory containing conversation folders",
        key="export_root_input",
    )
    st.session_state.export_root = export_root

    if export_root.strip():
        valid, msg = _validate_path(Path(export_root), must_exist=True, is_dir=True)
        if valid:
            st.success("✅ Valid export root")
        else:
            st.error(f"❌ {msg}")
    else:
        st.warning("⚠️ Set export root before running jobs")

    index_root = st.text_input(
        "Index Output Root",
        value=st.session_state.get("index_root", export_root),
        help=f"Directory where {config.INDEX_DIRNAME} folder will be created",
        key="index_root_input",
    )
    st.session_state.index_root = index_root

    st.divider()

    # Provider Configuration
    st.subheader("🔧 Provider Settings")
    provider = st.selectbox(
        "Embedding Provider",
        ["vertex", "openai", "cohere", "huggingface", "local", "qwen"],
        index=0,
        help="Provider for embedding operations",
        key="provider_select",
    )
    st.session_state.provider = provider
    os.environ["EMBED_PROVIDER"] = provider

    st.divider()

    # Environment Variables
    st.subheader("🌍 Environment Variables")

    with st.expander("Google Cloud Settings"):
        gcp_project = st.text_input(
            "GCP_PROJECT",
            value=os.environ.get("GCP_PROJECT", ""),
            help="Your Google Cloud project ID",
        )
        gcp_region = st.text_input(
            "GCP_REGION",
            value=os.environ.get("GCP_REGION", "global"),
            help="GCP region for Vertex AI",
        )
        credentials_path = st.text_input(
            "GOOGLE_APPLICATION_CREDENTIALS",
            value=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
            help="Path to service account JSON file",
        )

        if st.button("Apply GCP Settings", use_container_width=True):
            if gcp_project:
                os.environ["GCP_PROJECT"] = gcp_project
                os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_project
            os.environ["GCP_REGION"] = gcp_region
            if credentials_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            st.success("✅ Environment variables updated")

# ---------- Main Content ----------
st.title("📧 EmailOps Dashboard")
st.markdown("*Comprehensive email operations management system*")

if not CORTEX_AVAILABLE:
    st.error(
        "❌ Cortex modules not found. Please check your installation and PYTHONPATH."
    )
    st.stop()

# ---------- Quick Start Panel ----------
if st.session_state.show_quick_start:
    setup_status = _check_setup_status()
    completed_steps = sum(setup_status.values())
    total_steps = len(setup_status)

    st.markdown("---")
    st.subheader("🚀 Quick Start Guide")

    # Progress bar
    progress = completed_steps / total_steps
    st.progress(
        progress,
        text=f"Setup Progress: {completed_steps}/{total_steps} steps completed",
    )

    # Setup steps in columns
    col1, col2 = st.columns(2)

    with col1:
        # Step 1: Project Root
        icon1 = "✅" if setup_status["project_root_valid"] else "1️⃣"
        with st.container():
            st.markdown(f"**{icon1} Configure Project Root**")
            if setup_status["project_root_valid"]:
                st.success("Project root is configured correctly!")
            else:
                st.info("Set the path to your EmailOps project in the sidebar")

        # Step 2: Export Root
        icon2 = "✅" if setup_status["export_root_valid"] else "2️⃣"
        with st.container():
            st.markdown(f"**{icon2} Set Export Root**")
            if setup_status["export_root_valid"]:
                st.success("Export root is configured!")
            else:
                st.info("Point to your Outlook conversation exports folder")

        # Step 3: GCP Config
        icon3 = "✅" if setup_status["gcp_configured"] else "3️⃣"
        with st.container():
            st.markdown(f"**{icon3} Configure GCP (for Vertex AI)**")
            if setup_status["gcp_configured"]:
                st.success("GCP credentials configured!")
            else:
                st.info("Set GCP_PROJECT and GOOGLE_APPLICATION_CREDENTIALS")

    with col2:
        # Step 4: Cortex Available
        icon4 = "✅" if setup_status["cortex_available"] else "4️⃣"
        with st.container():
            st.markdown(f"**{icon4} Cortex Backend**")
            if setup_status["cortex_available"]:
                st.success("Cortex backend loaded successfully!")
            else:
                st.error("Cortex modules missing")

        # Step 5: Build Index
        icon5 = "✅" if setup_status["index_exists"] else "5️⃣"
        with st.container():
            st.markdown(f"**{icon5} Build Search Index**")
            if setup_status["index_exists"]:
                st.success("Index exists and ready!")
            else:
                st.info("Go to Index tab and click 'Start Indexing'")

        # Ready to use
        if all(setup_status.values()):
            st.balloons()
            st.success("🎉 **All set!** You're ready to search and draft emails!")

    st.markdown("---")

# ---------- Main Tabs ----------
tabs = st.tabs(
    [
        "📊 Status",
        "🔍 Index",
        "📄 Chunk",
        "🔎 Search & Draft",
        "📝 Summarize",
        "🩺 Doctor",
        "🪵 Logs",
        "Info Help",
    ]
)

# ---------- STATUS TAB ----------
with tabs[0]:
    st.header("📊 Index Status")

    export_display = st.session_state.export_root or "<not set>"
    index_display = (
        st.session_state.get("index_root")
        or st.session_state.export_root
        or "<not set>"
    )
    st.caption(f"Export root: `{export_display}` | Index root: `{index_display}`")

    try:
        export_value = (
            st.session_state.export_root.strip() if st.session_state.export_root else ""
        )
        index_value = (st.session_state.get("index_root") or "").strip()

        if not export_value:
            st.info("👈 Set the Outlook export root in the sidebar to view status.")
        else:
            export_path = Path(export_value)
            index_base = Path(index_value or export_value)
            index_dir = index_base / config.INDEX_DIRNAME

            if index_dir.exists():
                index_file = index_dir / "index.faiss"
                mapping_file = index_dir / "mapping.json"
                embeddings_file = index_dir / "embeddings.npy"
                meta_file = index_dir / "meta.json"

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    doc_count = 0
                    if mapping_file.exists():
                        try:
                            with mapping_file.open(encoding="utf-8") as f:
                                mapping = json.load(f)
                                if isinstance(mapping, list | dict):
                                    doc_count = len(mapping)
                        except Exception:
                            pass
                    st.metric("📄 Documents Indexed", f"{doc_count:,}")

                with col2:
                    index_exists = index_file.exists() or embeddings_file.exists()
                    st.metric(
                        "🗂️ Index Status", "✅ Ready" if index_exists else "❌ Missing"
                    )

                with col3:
                    total_size_mb = 0
                    for f in [index_file, mapping_file, embeddings_file, meta_file]:
                        if f.exists():
                            total_size_mb += f.stat().st_size / (1024 * 1024)
                    st.metric("💾 Index Size", f"{total_size_mb:.1f} MB")

                with col4:
                    last_modified = None
                    for f in [index_file, mapping_file, embeddings_file]:
                        if f.exists():
                            mtime = datetime.fromtimestamp(f.stat().st_mtime)
                            if last_modified is None or mtime > last_modified:
                                last_modified = mtime
                    if last_modified:
                        st.metric(
                            "🕐 Last Updated", last_modified.strftime("%Y-%m-%d %H:%M")
                        )
                    else:
                        st.metric("🕐 Last Updated", "N/A")

                if meta_file.exists():
                    try:
                        with meta_file.open(encoding="utf-8") as f:
                            meta = json.load(f)

                        st.subheader("📋 Index Metadata")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(
                                f"**Provider:** `{meta.get('provider', 'Unknown')}`"
                            )
                            st.markdown(f"**Model:** `{meta.get('model', 'Unknown')}`")
                            st.markdown(
                                f"**Dimensions:** `{meta.get('dimensions', 'Unknown')}`"
                            )

                        with col2:
                            st.markdown(
                                f"**Index Type:** `{meta.get('index_type', 'FAISS')}`"
                            )
                            st.markdown(
                                f"**Created:** `{meta.get('created_at', 'Unknown')}`"
                            )
                            st.markdown(
                                f"**Version:** `{meta.get('version', 'Unknown')}`"
                            )
                    except Exception as e:
                        st.warning(f"Could not read metadata: {e}")

                st.subheader("📁 Conversation Folders")
                conv_folders = [
                    d
                    for d in export_path.iterdir()
                    if d.is_dir() and not d.name.startswith("_")
                ]
                if conv_folders:
                    st.success(f"Found **{len(conv_folders)}** conversation folders")
                    with st.expander("Show folders"):
                        for folder in conv_folders[:50]:
                            st.text(f"• {folder.name}")
                        if len(conv_folders) > 50:
                            st.info(f"... and {len(conv_folders) - 50} more")
                else:
                    st.warning("No conversation folders found")
            else:
                st.warning("⚠️ Index directory not found. Please run indexing first.")
                st.info(f"Expected location: `{index_dir}`")

                if st.button("🚀 Go to Index Tab", type="primary"):
                    st.info(
                        "Click on the 'Index' tab above to build your search index!"
                    )

    except Exception as e:
        st.error(f"Failed to load index status: {e}")

# ---------- INDEX TAB ----------
with tabs[1]:
    st.header("🔍 Build/Update Index")

    st.info(
        "💡 **Tip:** Start with default settings for your first index. You can adjust parameters later for optimization."
    )

    col1, col2, col3 = st.columns(3)

    provider_options = [
        "vertex",
        "openai",
        "cohere",
        "huggingface",
        "local",
        "qwen",
    ]
    default_provider = st.session_state.get("provider", "vertex")
    if default_provider not in provider_options:
        default_provider = "vertex"
    batch_default = int(os.getenv("EMBED_BATCH", config.DEFAULT_BATCH_SIZE))

    with col1:
        provider = st.selectbox(
            "Embedding Provider",
            provider_options,
            index=provider_options.index(default_provider),
            help="Provider must match credentials in your environment",
        )
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=250,
            value=batch_default,
            help="Number of documents to process at once",
        )

    with col2:
        model_override = st.text_input(
            "Model Override (optional)",
            value=st.session_state.get("embed_model", "") or "",
            help="Leave empty to use provider default",
        )
        force_reindex = st.checkbox(
            "Force Full Reindex", value=False, help="Rebuild entire index from scratch"
        )

    with col3:
        limit_enabled = st.checkbox(
            "Test Mode (limit documents)", value=False, help="Good for testing setup"
        )
        limit_chunks = st.number_input(
            "Max Chunks per Conversation",
            min_value=1,
            max_value=1000,
            value=100,
            disabled=not limit_enabled,
        )

    if st.button("🚀 Start Indexing", type="primary", use_container_width=True):
        st.session_state.provider = provider
        st.session_state.embed_model = model_override

        export_root_value = (
            st.session_state.export_root.strip() if st.session_state.export_root else ""
        )
        if not export_root_value:
            st.error("Please set the Outlook export root before indexing.")
            st.stop()

        export_path = Path(export_root_value)
        if not export_path.exists():
            st.error(f"Export root does not exist: {export_path}")
            st.stop()

        # Use cortex CLI
        cmd = [
            sys.executable,
            "-m",
            "cortex_cli.main",
            "index",
            "--root",
            export_root_value,
            "--provider",
            provider,
            "--workers",
            str(int(os.cpu_count() or 4)),
        ]

        if limit_enabled:
            cmd.extend(["--limit", str(int(limit_chunks))])
        if force_reindex:
            cmd.append("--force")

        _run_command(
            cmd, workdir=st.session_state.project_root, title="Running Indexer"
        )

# ---------- CHUNK TAB ----------
with tabs[2]:
    st.header("📄 Document Chunking")
    st.info(
        "Chunking is now handled automatically during indexing. This tab is for debugging."
    )

# ---------- SEARCH & DRAFT TAB ----------
with tabs[3]:
    st.header("🔎 Search & Draft")

    st.info(
        """
    💡 **How to use:**
    1. Enter your search query below
    2. Click **Search Only** to find relevant emails
    3. Add sender info and click **Search & Draft** to generate a response
    """
    )

    query = st.text_area(
        "Query",
        placeholder="What emails discuss the Q4 budget proposal?",
        height=100,
        help="Enter your search query or draft request",
    )

    col1, col2 = st.columns(2)

    with col1:
        k = st.slider(
            "Top-K Results",
            min_value=1,
            max_value=250,
            value=50,
            step=5,
            help="Number of results to retrieve",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Higher = more creative",
        )

    with col2:
        sender = st.text_input(
            "Sender Name/Email",
            value="",
            placeholder="John Doe <john@example.com>",
            help="Required for drafting responses",
        )
        include_attachments = st.checkbox("Include Attachments", value=True)

    col1, col2 = st.columns(2)

    with col1:
        search_only = st.button("🔍 Search Only", use_container_width=True)

    with col2:
        search_and_draft = st.button(
            "✉️ Search & Draft", use_container_width=True, type="primary"
        )

    if search_only or search_and_draft:
        if not query:
            st.error("Please enter a query")
        else:
            try:
                with st.spinner("Searching..."):
                    # Use Cortex Hybrid Search
                    search_input = KBSearchInput(
                        tenant_id="default", user_id="ui-user", query=query, k=k
                    )
                    search_results = tool_kb_search_hybrid(search_input)
                    results = search_results.results

                    if not results:
                        st.warning("No results found. Try adjusting your query.")
                    else:
                        st.success(f"Found **{len(results)}** results")

                        # Group results by conversation
                        conv_groups = {}
                        for r in results:
                            conv_key = r.thread_id or "unknown"
                            if conv_key not in conv_groups:
                                conv_groups[conv_key] = []
                            conv_groups[conv_key].append(r)

                        st.subheader("📋 Search Results")
                        for conv_key, items in conv_groups.items():
                            conv_path = Path(st.session_state.export_root) / conv_key
                            path_exists = (
                                conv_path.exists()
                                and (conv_path / "Conversation.txt").exists()
                            )

                            status_icon = "✅" if path_exists else "⚠️"
                            first_item = items[0]
                            # Try to get subject from metadata or snippet
                            subject = first_item.metadata.get("subject", "No subject")

                            with st.expander(
                                f"{status_icon} **{conv_key}** - {subject} ({len(items)} items)"
                            ):
                                st.markdown(f"**Score:** {first_item.score:.3f}")
                                st.markdown(f"**Snippet:** {first_item.snippet}")

                                if path_exists:
                                    st.markdown(f"**📁 Path:** `{conv_path}`")
                                    if st.button("Open Folder", key=f"open_{conv_key}"):
                                        _open_path(conv_path)

                        if search_and_draft and sender:
                            with st.spinner("Drafting email..."):

                                async def run_draft():
                                    graph = build_draft_graph().compile()
                                    initial_state = {
                                        "tenant_id": "default",
                                        "user_id": "ui-user",
                                        "thread_id": None,  # Could infer from search results?
                                        "explicit_query": query,
                                        "draft_query": None,
                                        "retrieval_results": None,
                                        "assembled_context": None,
                                        "draft": None,
                                        "critique": None,
                                        "iteration_count": 0,
                                        "error": None,
                                    }
                                    return await graph.ainvoke(initial_state)

                                final_state = asyncio.run(run_draft())
                                draft_result = final_state.get("draft")

                            if draft_result:
                                st.subheader("📝 Email Draft")
                                st.text_area(
                                    "Subject", value=draft_result.subject, disabled=True
                                )
                                st.text_area(
                                    "Body",
                                    value=draft_result.body,
                                    height=300,
                                    disabled=True,
                                )

                                st.download_button(
                                    "📥 Download Draft JSON",
                                    data=_format_json(draft_result),
                                    file_name="draft_result.json",
                                    mime="application/json",
                                )
                            else:
                                st.error("Failed to generate draft.")

                        elif search_and_draft and not sender:
                            st.error("Please provide sender name/email for drafting")

            except Exception as e:
                st.error(f"Operation failed: {e}")
                st.exception(e)

# ---------- SUMMARIZE TAB ----------
with tabs[4]:
    st.header("📝 Summarize Email Thread")

    thread_path = st.text_input(
        "Thread Directory",
        value=st.session_state.export_root,
        help="Directory containing Conversation.txt",
    )

    if st.button("📊 Analyze Thread", type="primary", use_container_width=True):
        thread_dir = Path(thread_path)
        convo_file = thread_dir / "Conversation.txt"

        if not convo_file.exists():
            st.error(f"Conversation.txt not found in {thread_dir}")
        else:
            try:
                with st.spinner("Analyzing thread..."):
                    # We need a thread_id. For local files, we might need to ingest first or mock it.
                    # But the graph expects a thread_id to load from DB.
                    # If we want to summarize a local file, we might need a different entry point or ingest it first.
                    # For now, let's assume the user provides a thread ID or we use the folder name as ID if it was ingested.

                    thread_id = thread_dir.name

                    async def run_summary():
                        graph = build_summarize_graph().compile()
                        initial_state = {
                            "tenant_id": "default",
                            "user_id": "ui-user",
                            "thread_id": thread_id,
                            "thread_context": None,
                            "facts_ledger": None,
                            "critique": None,
                            "iteration_count": 0,
                            "summary": None,
                            "error": None,
                        }
                        return await graph.ainvoke(initial_state)

                    final_state = asyncio.run(run_summary())
                    summary = final_state.get("summary")

                    if summary:
                        st.success("✅ Analysis complete")
                        st.subheader("📋 Summary")
                        st.write(summary.content)

                        if summary.key_facts:
                            st.subheader("Key Facts")
                            for fact in summary.key_facts:
                                st.write(f"- {fact}")

                        st.download_button(
                            "📥 Download JSON",
                            data=_format_json(summary),
                            file_name="thread_analysis.json",
                            mime="application/json",
                        )
                    else:
                        st.error(
                            "Failed to generate summary. Ensure the thread is ingested first."
                        )

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.exception(e)

# ---------- DOCTOR TAB ----------
with tabs[5]:
    st.header("🩺 System Doctor")

    st.markdown(
        """
    Run comprehensive diagnostics to check:
    - ✅ Dependency installation status
    - ✅ Environment variables and API keys
    - ✅ Index health and compatibility
    - ✅ Embedding provider connectivity
    """
    )

    col1, col2 = st.columns(2)

    with col1:
        install_missing = st.checkbox(
            "Auto-Install Missing",
            value=False,
            help="Automatically install missing dependencies",
        )

    with col2:
        log_level = st.selectbox(
            "Log Level", ["INFO", "DEBUG", "WARNING", "ERROR"], index=0
        )

    doctor_provider = st.selectbox(
        "Provider to Check",
        [
            "(Use Index Provider)",
            "vertex",
            "openai",
            "cohere",
            "huggingface",
            "local",
            "qwen",
        ],
        index=0,
    )

    if st.button("🔍 Run Diagnostics", type="primary", use_container_width=True):
        cmd = [
            sys.executable,
            "-m",
            "cortex_cli.main",
            "doctor",
            "--root",
            st.session_state.export_root,
        ]

        if doctor_provider != "(Use Index Provider)":
            cmd.extend(["--provider", doctor_provider])

        if install_missing:
            cmd.append("--auto-install")

        _run_command(
            cmd, workdir=st.session_state.project_root, title="Running System Doctor"
        )

# ---------- LOGS TAB ----------
with tabs[6]:
    st.header("🪵 Debug Logs")

    if LOG_BUFFER_KEY not in st.session_state:
        st.session_state[LOG_BUFFER_KEY] = deque(maxlen=LOG_BUFFER_MAXLEN)

    level_col, search_col = st.columns([2, 1])
    with level_col:
        selected_levels = st.multiselect(
            "Log Levels",
            options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default=["INFO", "WARNING", "ERROR", "CRITICAL"],
        )
    with search_col:
        search_term = st.text_input("Search", placeholder="Filter logs...")

    buffer_snapshot = list(st.session_state[LOG_BUFFER_KEY])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Refresh Logs", use_container_width=True)
    with col2:
        if st.button("Clear Logs", use_container_width=True):
            st.session_state[LOG_BUFFER_KEY].clear()
            buffer_snapshot = []
    with col3:
        download_text = "\n".join(
            f"{e['timestamp']} | {e['level']} | {e['logger']} | {e['message']}"
            for e in buffer_snapshot
        )
        st.download_button(
            "Download Logs",
            data=download_text or "",
            file_name="logs.txt",
            mime="text/plain",
            use_container_width=True,
        )

    logs_to_show = []
    for entry in buffer_snapshot:
        if entry["level"] not in selected_levels:
            continue
        if search_term and (
            search_term.lower()
            not in " ".join([entry["message"], entry["logger"], entry["level"]]).lower()
        ):
            continue
        logs_to_show.append(entry)

    if logs_to_show:
        formatted = "\n".join(
            f"{e['timestamp']} | {e['level']:<7} | {e['logger']} | {e['message']}"
            for e in logs_to_show
        )
        st.code(formatted, language="log")
    else:
        st.info("No log entries match the current filters.")

# ---------- HELP TAB ----------
with tabs[7]:
    st.header("Help & Documentation")

    st.markdown(
        """
    ## 🚀 Quick Start Guide

    ### Step 1: Configure Paths
    Set your **Project Root** (where EmailOps code lives) and **Export Root** (where your Outlook exports are) in the sidebar.

    ### Step 2: Set Up GCP (for Vertex AI)
    ```bash
    # Set environment variables
    GCP_PROJECT=your-project-id
    GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
    ```

    ### Step 3: Build Index
    Go to **Index** tab → Click **Start Indexing**

    ### Step 4: Search & Draft
    Go to **Search & Draft** tab → Enter query → Click **Search Only** or **Search & Draft**

    ---

    ## 🔧 CLI Quick Reference

    The UI runs these commands under the hood:

    ```bash
    # Build/update search index
    cortex index --root /path/to/exports --provider vertex

    # System diagnostics
    cortex doctor --root /path/to/exports

    # Search and draft (programmatic)
    cortex search "your query"
    ```

    ---

    ## 🩺 Troubleshooting

    | Problem | Solution |
    |---------|----------|
    | Cortex modules missing | Check PYTHONPATH and project root |
    | Indexing fails | Verify export root has conversation folders |
    | No search results | Confirm index exists (check Status tab) |
    | Draft errors | Ensure Vertex AI credentials are set |

    """
    )

# ---------- Footer ----------
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
    EmailOps Dashboard v2.0 | Powered by Cortex & Vertex AI
    </div>
    """,
    unsafe_allow_html=True,
)
