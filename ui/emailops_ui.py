#!/usr/bin/env python3
"""
EmailOps UI — Enhanced Streamlit Dashboard for Email Operations

Features:
- Status: Live index health monitoring with detailed metrics
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

import importlib
import json
import logging
import os
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Import centralized configuration
try:
    from emailops.config import get_config
    config = get_config()
except ImportError:
    # Fallback if config module not available - use defaults
    class FallbackConfig:
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

    /* Improve dataframe visibility */
    .dataframe {
        background-color: #ffffff !important;
    }

    .dataframe thead tr th {
        background-color: #f8f9fa !important;
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }

    .dataframe tbody tr:nth-child(even) {
        background-color: #f8f9fa !important;
    }

    .dataframe tbody tr:hover {
        background-color: #e9ecef !important;
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

    /* Info box styling */
    .stInfo {
        background-color: #eff6ff !important;
        color: #1e3a8a !important;
        border: 1px solid #93c5fd !important;
        border-left: 4px solid #3b82f6 !important;
    }

    /* Success message styling */
    .stSuccess {
        background-color: #f0fdf4 !important;
        color: #14532d !important;
        border: 1px solid #86efac !important;
        border-left: 4px solid #22c55e !important;
    }

    /* Warning message styling */
    .stWarning {
        background-color: #fffbeb !important;
        color: #451a03 !important;
        border: 1px solid #fde68a !important;
        border-left: 4px solid #f59e0b !important;
    }

    /* Error message styling */
    .stError {
        background-color: #fef2f2 !important;
        color: #450a0a !important;
        border: 1px solid #fecaca !important;
        border-left: 4px solid #ef4444 !important;
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
            # Never raise from a logging handler
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


def _pkg_root_from(project_root: Path) -> Path | None:
    """Find the package root for importing emailops modules."""
    pr = project_root
    if (pr / "emailops").exists() and (pr / "emailops").is_dir():
        return pr
    if pr.name == "emailops":
        return pr.parent
    return None


@st.cache_resource(show_spinner=False)
def _import_modules(project_root: str) -> dict[str, Any]:
    """Import required modules with proper error handling."""
    pr = Path(project_root)
    pkg_root = _pkg_root_from(pr)
    if not pkg_root:
        raise RuntimeError(
            f"Could not find 'emailops' package under: {project_root}. "
            f"Please ensure the project root contains the 'emailops' folder."
        )

    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    modules: dict[str, Any] = {}

    required_modules = [
        ("utils", "emailops.utils"),
        ("llm_client", "emailops.llm_client"),
        ("env_utils", "emailops.env_utils"),
        ("email_indexer", "emailops.email_indexer"),
        ("text_chunker", "emailops.text_chunker"),
    ]

    optional_modules = [
        ("search_and_draft", "emailops.search_and_draft"),
        ("summarize_email_thread", "emailops.summarize_email_thread"),
    ]

    for name, module_path in required_modules:
        try:
            modules[name] = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(f"Failed to import required module {module_path}: {e}")

    for name, module_path in optional_modules:
        try:
            modules[name] = importlib.import_module(module_path)
        except ImportError:
            modules[name] = None

    return modules


def _format_json(obj: Any) -> str:
    """Format object as JSON string."""
    try:
        if hasattr(obj, "to_dict"):
            obj = obj.to_dict()
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False, indent=2)


def _run_command(
    cmd: list[str], workdir: str | None = None, title: str = "Running Command"
):
    """Run a command and stream output with security validation and better error handling."""
    # Import validators
    try:
        from emailops.validators import (
            quote_shell_arg,
            validate_command_args,
            validate_directory_path,
        )
    except ImportError:
        st.error("Security module not available. Cannot execute commands.")
        logger.error("Failed to import emailops.validators")
        return 1

    with st.status(title, expanded=True) as status:
        # Validate command is not empty
        if not cmd or not cmd[0]:
            st.error("Invalid command: empty command provided")
            logger.error("Attempted to run empty command")
            return 1

        # Security: Validate command and arguments
        # Whitelist of allowed commands for EmailOps
        allowed_commands = [
            "python", "python3", "py",  # Python interpreters
            sys.executable,  # Current Python executable
        ]

        command = cmd[0]
        args = cmd[1:] if len(cmd) > 1 else []

        # Check if using python with -m flag (module execution)
        is_valid_cmd = False
        if command in allowed_commands or (os.path.isabs(command) and Path(command).resolve() == Path(sys.executable).resolve()):
            is_valid_cmd = True

        if not is_valid_cmd:
            st.error(f"❌ Security: Command '{command}' not in allowed list")
            logger.error("Blocked execution of non-whitelisted command: %s", command)
            return 1

        # Validate arguments for injection attempts
        is_valid, msg = validate_command_args(command, args, allowed_commands)
        if not is_valid:
            st.error(f"❌ Security: {msg}")
            logger.error("Command validation failed: %s", msg)
            return 1

        # Validate working directory if provided
        if workdir:
            is_valid, msg = validate_directory_path(workdir, must_exist=True)
            if not is_valid:
                st.error(f"❌ Invalid working directory: {msg}")
                logger.error("Working directory validation failed: %s", msg)
                return 1

        # Display sanitized command
        safe_cmd_display = " ".join(quote_shell_arg(c) for c in cmd)
        st.code(safe_cmd_display, language="bash")

        try:
            env = os.environ.copy()
            # Ensure EMBED_PROVIDER is set for search operations
            if "EMBED_PROVIDER" not in env:
                env["EMBED_PROVIDER"] = st.session_state.get("provider", "vertex")
            logger.info("Starting validated command: %s", safe_cmd_display)

            proc = subprocess.Popen(
                cmd,  # Pass as list, not shell string
                cwd=workdir or None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                env=env,
                shell=False,  # CRITICAL: Never use shell=True with user input
            )
        except FileNotFoundError:
            st.error(f"❌ Command not found: {cmd[0]}")
            logger.error("Command not found: %s", cmd[0])
            return 1
        except PermissionError as e:
            st.error(f"❌ Permission denied: {e}")
            logger.error("Permission denied executing command: %s", e)
            return 1
        except Exception as e:
            st.error(f"❌ Failed to start process: {e}")
            logger.error("Failed to start process: %s", e)
            return 1

        log_container = st.container()
        lines: list[str] = []

        with log_container:
            log_area = st.empty()
            try:
                while True:
                    line = proc.stdout.readline() if proc.stdout else ""
                    if not line and proc.poll() is not None:
                        break
                    if line:
                        lines.append(line.rstrip("\n"))
                        # Show last 500 lines to keep UI responsive
                        display_lines = lines[-500:]
                        log_area.code("\n".join(display_lines), language="log")
            except KeyboardInterrupt:
                logger.warning("Command interrupted by user")
                proc.terminate()
                proc.wait(timeout=5)
                status.update(label="⚠️ Interrupted by user", state="error")
                return 130  # Standard exit code for SIGINT
            except Exception as e:
                logger.error("Error reading command output: %s", e)
                st.error(f"Error reading output: {e}")

        rc = proc.poll()
        if rc == 0:
            status.update(label="✅ Completed Successfully", state="complete")
            logger.info("Command completed successfully")
        else:
            status.update(label=f"❌ Failed with exit code {rc}", state="error")
            logger.error("Command failed with exit code %s", rc)

        return rc


def _display_dataframe(
    data: list[dict[str, Any]],
    columns: list[str] | None = None,
    max_rows: int = 100,
    title: str | None = None,
):
    """Display data in a nicely formatted dataframe."""
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
        df,
        use_container_width=True,
        hide_index=True,
        height=min(400, 35 * len(df)),  # Adaptive height
    )


# ---------- Session State Initialization ----------
if "provider" not in st.session_state:
    st.session_state.provider = "vertex"
if "project_root" not in st.session_state:
    st.session_state.project_root = str(Path.cwd())
if "export_root" not in st.session_state:
    export_env = os.getenv("EMAILOPS_EXPORT_ROOT")
    st.session_state.export_root = export_env.strip() if export_env else ""
if "index_root" not in st.session_state:
    index_env = os.getenv("EMAILOPS_INDEX_ROOT")
    if index_env:
        st.session_state.index_root = index_env.strip()
    else:
        st.session_state.index_root = st.session_state.export_root
if "modules" not in st.session_state:
    st.session_state.modules = None

_ensure_log_handler()

# ---------- Sidebar Configuration ----------
with st.sidebar:
    st.header("⚙️ Configuration")

    # Path Configuration
    st.subheader("📁 Paths")
    project_root = st.text_input(
        "Code Project Root",
        value=st.session_state.project_root,
        help="Directory containing the 'emailops' package",
        key="project_root_input",
    )
    st.session_state.project_root = project_root

    # Validate project root
    valid, msg = _validate_path(Path(project_root), must_exist=True, is_dir=True)
    if valid:
        if (Path(project_root) / "emailops").exists():
            st.success("✅ Valid project root")
        else:
            st.warning("⚠️ 'emailops' folder not found")
    else:
        st.error(f"❌ {msg}")

    export_root = st.text_input(
        "Outlook Export Root",
        value=st.session_state.export_root,
        help="Directory containing conversation folders (e.g. raw Outlook export)",
        key="export_root_input",
    )
    st.session_state.export_root = export_root

    # Validate export root
    if export_root.strip():
        valid, msg = _validate_path(Path(export_root), must_exist=True, is_dir=True)
        if valid:
            st.success("✅ Valid export root")
        else:
            st.error(f"❌ {msg}")
    else:
        st.warning("⚠️ Provide the path to your Outlook export root before running jobs")

    index_root = st.text_input(
        "Index Output Root",
        value=st.session_state.get("index_root", export_root),
        help="Directory where the _index folder should live",
        key="index_root_input",
    )
    st.session_state.index_root = index_root

    index_root_value = index_root.strip() if index_root else ""
    if index_root_value:
        valid, msg = _validate_path(
            Path(index_root_value), must_exist=True, is_dir=True
        )
        if valid:
            st.success("✅ Valid index output root")
        else:
            st.error(f"❌ {msg}")
    else:
        st.info("ℹ️ Leave blank to reuse the export root for index output")

    st.divider()

    # Provider Configuration
    st.subheader("🔧 Provider Settings")
    provider = st.selectbox(
        "Embedding Provider",
        ["vertex", "openai", "azure", "cohere", "huggingface", "local", "qwen"],
        index=0,
        help="Provider for embedding operations. Generation uses Vertex AI.",
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

    st.divider()

    # Module Loading
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
st.markdown("Comprehensive email operations management system")

# Auto-load modules on first run if not already loaded
if st.session_state.modules is None:
    try:
        modules = _import_modules(st.session_state.project_root)
        st.session_state.modules = modules
    except Exception as e:
        st.error(f"❌ Failed to load modules: {e}")
        st.info(
            "Please check your project root path in the sidebar and ensure all dependencies are installed."
        )
        st.stop()
else:
    modules = st.session_state.modules

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
        "ℹ️ Help",
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
    st.caption(f"Export root: {export_display}\nIndex output root: {index_display}")

    # Since vertex_utils doesn't exist, provide basic index status from the index directory
    try:
        export_value = (
            st.session_state.export_root.strip() if st.session_state.export_root else ""
        )
        index_value = (st.session_state.get("index_root") or "").strip()

        if not export_value:
            st.info("Set the Outlook export root in the sidebar to view status.")
            st.stop()

        export_path = Path(export_value)
        index_base = Path(index_value or export_value)
        index_dir = index_base / "_index"

        if index_dir.exists():
            # Check for index files
            index_file = index_dir / "index.faiss"
            mapping_file = index_dir / "mapping.json"
            embeddings_file = index_dir / "embeddings.npy"
            meta_file = index_dir / "meta.json"

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                # Count documents in mapping
                doc_count = 0
                if mapping_file.exists():
                    try:
                        with open(mapping_file, encoding="utf-8") as f:
                            mapping = json.load(f)
                            doc_count = len(mapping) if isinstance(mapping, list) else 0
                    except Exception:
                        pass
                st.metric("Documents Indexed", f"{doc_count:,}")

            with col2:
                # Check index existence
                index_exists = index_file.exists() or embeddings_file.exists()
                st.metric(
                    "Index Status", "✅ Exists" if index_exists else "❌ Not Found"
                )

            with col3:
                # Get file sizes
                total_size_mb = 0
                for f in [index_file, mapping_file, embeddings_file, meta_file]:
                    if f.exists():
                        total_size_mb += f.stat().st_size / (1024 * 1024)
                st.metric("Index Size", f"{total_size_mb:.1f} MB")

            with col4:
                # Last modified
                last_modified = None
                for f in [index_file, mapping_file, embeddings_file]:
                    if f.exists():
                        mtime = datetime.fromtimestamp(f.stat().st_mtime)
                        if last_modified is None or mtime > last_modified:
                            last_modified = mtime
                if last_modified:
                    st.metric("Last Updated", last_modified.strftime("%Y-%m-%d %H:%M"))
                else:
                    st.metric("Last Updated", "N/A")

            # Read metadata if available
            if meta_file.exists():
                try:
                    with open(meta_file, encoding="utf-8") as f:
                        meta = json.load(f)

                    st.subheader("Index Metadata")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.text(f"Provider: {meta.get('provider', 'Unknown')}")
                        st.text(f"Model: {meta.get('model', 'Unknown')}")
                        st.text(f"Dimensions: {meta.get('dimensions', 'Unknown')}")

                    with col2:
                        st.text(f"Index Type: {meta.get('index_type', 'FAISS')}")
                        st.text(f"Created: {meta.get('created_at', 'Unknown')}")
                        st.text(f"Version: {meta.get('version', 'Unknown')}")
                except Exception as e:
                    st.warning(f"Could not read metadata: {e}")

            # Show conversation folders
            st.subheader("📁 Conversation Folders")
            conv_folders = [
                d
                for d in export_path.iterdir()
                if d.is_dir() and not d.name.startswith("_")
            ]
            if conv_folders:
                st.info(f"Found {len(conv_folders)} conversation folders")
                with st.expander("Show folders"):
                    for folder in conv_folders[:50]:  # Show first 50
                        st.text(f"• {folder.name}")
            else:
                st.warning("No conversation folders found")
        else:
            st.warning("Index directory not found. Please run indexing first.")
            st.info(f"Expected index location: {index_dir}")

    except Exception as e:
        st.error(f"Failed to load index status: {e}")
        st.info("Please check your export root path in the sidebar.")

# ---------- INDEX TAB ----------
with tabs[1]:
    st.header("🔍 Build/Update Index")

    col1, col2, col3 = st.columns(3)

    provider_options = [
        "vertex",
        "openai",
        "azure",
        "cohere",
        "huggingface",
        "local",
        "qwen",
    ]
    default_provider = st.session_state.get("provider", "vertex")
    if default_provider not in provider_options:
        default_provider = "vertex"
    # Use config for default batch size
    batch_default = max(1, min(250, config.DEFAULT_BATCH_SIZE))

    with col1:
        provider = st.selectbox(
            "Embedding Provider",
            provider_options,
            index=provider_options.index(default_provider),
            help="Provider must match credentials available in your environment",
        )
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=250,
            value=batch_default,
            help="Email indexer clamps this value to the provider's maximum",
        )

    with col2:
        model_override = st.text_input(
            "Model Override",
            value=st.session_state.get("embed_model", "") or "",
            help="Optional provider-specific model or deployment name",
        )
        force_reindex = st.checkbox("Force Full Reindex", value=False)

    with col3:
        limit_enabled = st.checkbox(
            "Limit Documents", value=False, help="Use for smoke tests"
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

        cmd = [
            sys.executable,
            "-m",
            "emailops.email_indexer",
            "--root",
            export_root_value,
            "--provider",
            provider,
            "--batch",
            str(int(batch_size)),
        ]

        index_root_clean = (st.session_state.get("index_root") or "").strip()
        if index_root_clean:
            cmd.extend(["--index-root", index_root_clean])

        model_override_clean = model_override.strip()
        if model_override_clean:
            cmd.extend(["--model", model_override_clean])
        if force_reindex:
            cmd.append("--force-reindex")
        if limit_enabled:
            cmd.extend(["--limit", str(int(limit_chunks))])

        _run_command(
            cmd, workdir=st.session_state.project_root, title="Running Indexer"
        )

# ---------- CHUNK TAB ----------
with tabs[2]:
    st.header("📄 Document Chunking")

    col1, col2 = st.columns(2)

    with col1:
        input_dir = st.text_input(
            "Input Directory",
            value=st.session_state.export_root or "",
            help="Directory containing documents to chunk",
        )

    with col2:
        output_root = st.text_input(
            "Chunk Root",
            value=st.session_state.export_root or "",
            help=f"Chunked files will be saved under this path in a '{config.CHUNK_DIRNAME}' subfolder",
        )

    col1, col2 = st.columns(2)

    with col1:
        workers = st.number_input(
            "Worker Processes", min_value=1, max_value=16, value=6
        )
        chunk_size = st.number_input(
            "Chunk Size", min_value=100, max_value=10000, value=config.DEFAULT_CHUNK_SIZE
        )

    with col2:
        chunk_overlap = st.number_input(
            "Chunk Overlap", min_value=0, max_value=1000, value=config.DEFAULT_CHUNK_OVERLAP
        )
        file_pattern = st.text_input("File Pattern", value="*.txt")

    advanced = st.expander("Advanced Options")
    with advanced:
        test_mode_chunk = st.checkbox(
            "Sample Mode (process up to 10 files)", value=False, key="chunk_test_mode"
        )

    if st.button("🚀 Start Chunking", type="primary", use_container_width=True):
        # Note: processing.processor might be the correct path
        # Check if it exists first
        processor_path = (
            Path(st.session_state.project_root) / "processing" / "processor.py"
        )
        if processor_path.exists():
            cmd = [
                sys.executable,
                "-m",
                "processing.processor",
                "chunk",
            ]
        else:
            # Fallback to text_chunker if processor doesn't exist
            cmd = [
                sys.executable,
                "-m",
                "emailops.text_chunker",
            ]

        input_dir_value = input_dir.strip()
        output_root_value = output_root.strip()

        if not input_dir_value or not output_root_value:
            st.error(
                "Please set both input and chunk root directories before running the chunker."
            )
            st.stop()

        if not Path(input_dir_value).exists():
            st.error(f"Input directory does not exist: {input_dir_value}")
            st.stop()

        cmd.extend(
            [
                "--input",
                input_dir_value,
                "--output",
                output_root_value,
                "--workers",
                str(int(workers)),
                "--chunk-size",
                str(int(chunk_size)),
                "--chunk-overlap",
                str(int(chunk_overlap)),
                "--pattern",
                file_pattern,
            ]
        )

        if test_mode_chunk:
            cmd.append("--test")

        _run_command(
            cmd, workdir=st.session_state.project_root, title="Running Chunker"
        )

# ---------- SEARCH & DRAFT TAB ----------
with tabs[3]:
    st.header("🔎 Search & Draft")

    # Add helpful info box
    with st.info("", icon="ℹ️"):
        st.markdown("""
        **Quick Guide:**
        - Enter your search query to find relevant conversations
        - Results show actual conversation folders with full paths
        - Click folder names to view in file explorer (Windows)
        - Provide sender info to generate draft responses
        """)

    # Main search area
    query = st.text_area(
        "Query", placeholder="Enter your search query or draft request...", height=100
    )

    # Basic settings
    col1, col2 = st.columns(2)

    with col1:
        k = st.slider(
            "Top-K Results",
            min_value=1,
            max_value=200,
            value=60,
            step=5,
            help="Number of relevant conversations to retrieve",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Higher = more creative, Lower = more factual",
        )

    with col2:
        sender = st.text_input(
            "Sender Name/Email",
            placeholder="John Doe <john@example.com>",
            help="Required for drafting email responses",
        )
        include_attachments = st.checkbox(
            "Include Attachments",
            value=True,
            help="Automatically select relevant attachments",
        )

    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)

        with col1:
            chat_mode = st.checkbox(
                "Chat Mode",
                value=False,
                help="Enable conversational Q&A instead of email drafting",
            )
            session_id = st.text_input(
                "Session ID",
                placeholder="Leave empty for new session",
                disabled=not chat_mode,
                help="Maintain conversation context across queries",
            )

            # Add session management buttons
            if chat_mode:
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🔄 Reset Session", disabled=not session_id):
                        if hasattr(st.session_state, "chat_session"):
                            st.session_state.chat_session.reset()
                            st.session_state.chat_session.save()
                            st.success("Session reset!")
                            del st.session_state.chat_session
                            del st.session_state.current_session_id

                with col_b:
                    if st.button("📋 List Sessions"):
                        sessions_dir = (
                            Path(st.session_state.export_root)
                            / config.INDEX_DIRNAME
                            / "_chat_sessions"
                        )
                        if sessions_dir.exists():
                            session_files = list(sessions_dir.glob("*.json"))
                            if session_files:
                                st.info(f"Found {len(session_files)} sessions:")
                                for sf in session_files[:10]:  # Show max 10
                                    st.text(f"  • {sf.stem}")
                            else:
                                st.info("No saved sessions found")
                        else:
                            st.info("No sessions directory found")

        with col2:
            conv_id = st.text_input(
                "Conversation ID Filter",
                placeholder="e.g., 2024-03-15-meeting",
                help="Search only within a specific conversation folder",
            )
            conv_subject = st.text_input(
                "Subject Filter",
                placeholder="e.g., contract, invoice",
                help="Filter by keywords in email subject",
            )

            # Add confidence threshold for drafting
            min_confidence = st.slider(
                "Minimum Draft Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Reject drafts below this confidence score",
            )

        # Add folder browser option
        st.subheader("📁 Conversation Browser")
        browse_folders = st.checkbox("Browse conversation folders", value=False)

        if browse_folders:
            try:
                export_path = Path(st.session_state.export_root)
                if export_path.exists():
                    # Get list of conversation folders
                    conv_folders = [
                        d.name
                        for d in export_path.iterdir()
                        if d.is_dir()
                        and not d.name.startswith("_")
                        and (d / "Conversation.txt").exists()
                    ]

                    if conv_folders:
                        selected_folder = st.selectbox(
                            "Select conversation folder:",
                            ["", *sorted(conv_folders)[:100]],  # Limit to 100 for performance
                            help="Select a specific conversation to search within",
                        )
                        if selected_folder:
                            conv_id = selected_folder
                            st.success(f"✅ Will search in: {selected_folder}")
                    else:
                        st.warning(
                            "No conversation folders found with Conversation.txt"
                        )
                else:
                    st.error(f"Export root does not exist: {export_path}")
            except Exception as e:
                st.error(f"Error browsing folders: {e}")

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        search_only = st.button("🔍 Search Only", use_container_width=True)

    with col2:
        search_and_draft = st.button(
            "✉️ Search & Draft", use_container_width=True, type="primary"
        )

    # Handle search operations
    if search_only or search_and_draft:
        if not query:
            st.error("Please enter a query")
        elif not modules:
            st.error("Modules not loaded. Please load modules from the sidebar.")
        else:
            try:
                with st.spinner("Searching..."):
                    search_module = modules["search_and_draft"]
                    # Use config for index directory name
                    index_dir = Path(st.session_state.export_root) / config.INDEX_DIRNAME

                    # Build conversation filter if needed
                    conv_id_filter = None
                    if conv_id or conv_subject:
                        # Load mapping for subject filtering
                        mapping = search_module._load_mapping(index_dir)
                        conv_id_filter = set()

                        if conv_id:
                            conv_id_filter.add(str(conv_id).strip())

                        if conv_subject and mapping:
                            # Find conversation IDs by subject
                            hits = search_module._find_conv_ids_by_subject(
                                mapping, conv_subject
                            )
                            conv_id_filter |= hits

                        if not conv_id_filter:
                            st.info("No conversations matched the provided filters.")
                            conv_id_filter = (
                                None  # Allow search to continue without filter
                            )

                    # Handle chat mode with session
                    if chat_mode:
                        # Initialize or load chat session
                        if not hasattr(st.session_state, "chat_session") or (
                            session_id
                            and st.session_state.get("current_session_id") != session_id
                        ):
                            if session_id:
                                safe_id = search_module._sanitize_session_id(session_id)
                                chat_session = search_module.ChatSession(
                                    base_dir=index_dir,
                                    session_id=safe_id,
                                    max_history=10,
                                )
                                chat_session.load()
                                st.session_state.chat_session = chat_session
                                st.session_state.current_session_id = session_id
                            else:
                                # Create new session with timestamp
                                from datetime import datetime

                                new_session_id = f"chat_{datetime.now():%Y%m%d_%H%M%S}"
                                chat_session = search_module.ChatSession(
                                    base_dir=index_dir,
                                    session_id=new_session_id,
                                    max_history=10,
                                )
                                st.session_state.chat_session = chat_session
                                st.session_state.current_session_id = new_session_id
                        else:
                            chat_session = st.session_state.chat_session

                        # Build effective query with history
                        hist_for_query = chat_session.recent()
                        effective_query = (
                            search_module._build_search_query_from_history(
                                hist_for_query, query, max_back=5
                            )
                            if query
                            else query
                        )
                    else:
                        effective_query = query
                        chat_session = None

                    # Perform search with correct parameters
                    results = search_module._search(
                        ix_dir=index_dir,
                        query=effective_query,
                        k=k,
                        provider=st.session_state.provider,
                        conv_id_filter=conv_id_filter,
                    )

                if not results:
                    st.warning(
                        "No results found. Try adjusting your query or increasing K."
                    )
                else:
                    st.success(f"Found {len(results)} results")

                    # Enhanced search results with folder information
                    st.subheader("📋 Search Results")

                    # Group results by conversation
                    conv_groups = {}
                    for r in results:
                        conv_id = r.get("conv_id", "")
                        if not conv_id:
                            # Extract from ID if not present
                            conv_id = (
                                r.get("id", "").split("::")[0]
                                if "::" in r.get("id", "")
                                else "unknown"
                            )

                        if conv_id not in conv_groups:
                            conv_groups[conv_id] = []
                        conv_groups[conv_id].append(r)

                    # Display grouped results
                    for conv_id, items in conv_groups.items():
                        # Build conversation folder path
                        conv_path = Path(st.session_state.export_root) / conv_id
                        path_exists = (
                            conv_path.exists()
                            and (conv_path / "Conversation.txt").exists()
                        )

                        # Create expander for each conversation
                        status_icon = "✅" if path_exists else "⚠️"
                        first_item = items[0]
                        subject = first_item.get("subject", "No subject")[:80]

                        with st.expander(
                            f"{status_icon} **{conv_id}** - {subject} ({len(items)} items)"
                        ):
                            # Show conversation info
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                st.markdown(
                                    f"**Subject:** {first_item.get('subject', 'N/A')}"
                                )
                                st.markdown(
                                    f"**Date:** {first_item.get('date', 'N/A')}"
                                )
                                st.markdown(
                                    f"**From:** {first_item.get('from_name', '')} <{first_item.get('from_email', '')}>"
                                )

                                # Show full path
                                if path_exists:
                                    st.markdown(f"**📁 Path:** `{conv_path}`")

                                    # Add button to open folder (Windows only)
                                    if os.name == "nt":
                                        if st.button(
                                            "Open Folder", key=f"open_{conv_id}"
                                        ):
                                            try:
                                                os.startfile(str(conv_path))
                                            except Exception as e:
                                                st.error(f"Could not open folder: {e}")
                                else:
                                    st.warning(
                                        f"⚠️ Conversation folder not found at: {conv_path}"
                                    )

                            with col2:
                                # Show score distribution
                                scores = [float(item.get("score", 0)) for item in items]
                                st.metric(
                                    "Avg Score", f"{sum(scores) / len(scores):.3f}"
                                )
                                st.metric("Items", len(items))

                            # Show individual items in this conversation
                            st.markdown("**Items in this conversation:**")
                            for item in items:
                                doc_type = item.get("doc_type", "")
                                icon = "📎" if doc_type == "attachment" else "📧"
                                score = float(item.get("score", 0))

                                # Format item display
                                item_id = (
                                    item.get("id", "").split("::")[-1]
                                    if "::" in item.get("id", "")
                                    else item.get("id", "")
                                )
                                snippet = (
                                    item.get("text", "")[:200] + "..."
                                    if len(item.get("text", "")) > 200
                                    else item.get("text", "")
                                )

                                st.markdown(
                                    f"{icon} **{item_id}** (score: {score:.3f})"
                                )
                                st.markdown(f"   {snippet}")

                                # Show attachment info if present
                                if item.get("attachment_name"):
                                    st.markdown(
                                        f"   📎 Attachment: {item.get('attachment_name')} ({item.get('attachment_type', 'unknown')})"
                                    )

                                st.divider()

                    # Store results for drafting
                    st.session_state.search_results = results

                    # Summary statistics
                    st.subheader("📊 Search Summary")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Results", len(results))
                        st.metric("Unique Conversations", len(conv_groups))

                    with col2:
                        avg_score = (
                            sum(float(r.get("score", 0)) for r in results)
                            / len(results)
                            if results
                            else 0
                        )
                        st.metric("Average Score", f"{avg_score:.3f}")

                        doc_types = {}
                        for r in results:
                            dtype = r.get("doc_type", "unknown")
                            doc_types[dtype] = doc_types.get(dtype, 0) + 1
                        st.metric(
                            "Document Types",
                            ", ".join(f"{k}: {v}" for k, v in doc_types.items()),
                        )

                    with col3:
                        valid_paths = sum(
                            1
                            for conv_id in conv_groups
                            if (Path(st.session_state.export_root) / conv_id).exists()
                        )
                        st.metric("Valid Folders", f"{valid_paths}/{len(conv_groups)}")

                        date_range = []
                        for r in results:
                            if r.get("date"):
                                date_range.append(r.get("date"))
                        if date_range:
                            st.metric(
                                "Date Range",
                                f"{min(date_range)[:10]} to {max(date_range)[:10]}",
                            )

                    # Handle chat mode or drafting
                    if chat_mode and not sender:
                        # Chat mode - Q&A without drafting
                        if results:
                            with st.spinner("Generating chat response..."):
                                # Get chat history for context
                                chat_history = (
                                    chat_session.recent() if chat_session else []
                                )

                                # Call chat_with_context function with correct import
                                if hasattr(search_module, "chat_with_context"):
                                    chat_result = search_module.chat_with_context(
                                        query=query,
                                        context_snippets=results,
                                        chat_history=chat_history,
                                        temperature=temperature,
                                    )
                                else:
                                    st.error(
                                        "chat_with_context function not found in search_and_draft module"
                                    )
                                    chat_result = {
                                        "answer": "Function not available",
                                        "citations": [],
                                        "missing_information": [
                                            "chat_with_context not found"
                                        ],
                                    }

                                # Save to session
                                if chat_session:
                                    # Determine conv_id if filter identifies single conversation
                                    conv_id_for_turn = None
                                    if conv_id_filter and len(conv_id_filter) == 1:
                                        conv_id_for_turn = next(iter(conv_id_filter))

                                    chat_session.add_message(
                                        "user", query, conv_id=conv_id_for_turn
                                    )
                                    chat_session.add_message(
                                        "assistant",
                                        chat_result.get("answer", ""),
                                        conv_id=conv_id_for_turn,
                                    )
                                    chat_session.save()

                                # Display chat response
                                st.subheader("💬 Chat Response")

                                # Show session info
                                if chat_session:
                                    st.info(
                                        f"Session ID: {st.session_state.current_session_id} | Messages: {len(chat_session.messages)}"
                                    )

                                # Display answer
                                st.markdown("### Answer")
                                st.write(chat_result.get("answer", ""))

                                # Display citations
                                citations = chat_result.get("citations", [])
                                if citations:
                                    with st.expander(
                                        f"📚 Citations ({len(citations)})"
                                    ):
                                        for i, cite in enumerate(citations, 1):
                                            st.markdown(
                                                f"**{i}. {cite.get('document_id', '')}**"
                                            )
                                            st.write(
                                                f"   Fact: {cite.get('fact_cited', '')}"
                                            )
                                            conf = cite.get("confidence", "low")
                                            if conf == "high":
                                                conf_color = "🟢"
                                            elif conf == "medium":
                                                conf_color = "🟡"
                                            else:
                                                conf_color = "🔴"
                                            st.write(
                                                f"   Confidence: {conf_color} {conf}"
                                            )

                                # Display missing information
                                missing_info = chat_result.get(
                                    "missing_information", []
                                )
                                if missing_info:
                                    with st.expander("❓ Missing Information"):
                                        for item in missing_info:
                                            st.write(f"• {item}")

                                # Show chat history
                                if chat_session and chat_session.messages:
                                    with st.expander("💬 Chat History"):
                                        for msg in chat_session.messages[
                                            -10:
                                        ]:  # Show last 10 messages
                                            role_icon = (
                                                "👤" if msg.role == "user" else "🤖"
                                            )
                                            st.markdown(
                                                f"**{role_icon} {msg.role.title()}**"
                                            )
                                            st.write(
                                                msg.content[:500]
                                            )  # Truncate long messages
                                            st.caption(f"Time: {msg.timestamp}")
                                            if msg.conv_id:
                                                st.caption(
                                                    f"Conversation: {msg.conv_id}"
                                                )
                                            st.divider()
                        else:
                            st.warning(
                                "No results found for chat. Try adjusting your query."
                            )

                    elif search_and_draft and sender:
                        # Email drafting mode
                        with st.spinner("Drafting email with LLM-as-critic..."):
                            # Get chat history if available
                            chat_history = None
                            if chat_session:
                                chat_history = chat_session.recent()

                            # Call draft_email_structured with correct parameters
                            if hasattr(search_module, "draft_email_structured"):
                                draft_result = search_module.draft_email_structured(
                                    query=query,
                                    sender=sender,
                                    context_snippets=results,
                                    provider=st.session_state.provider,
                                    temperature=temperature,
                                    include_attachments=include_attachments,
                                    chat_history=chat_history,
                                    # max_context_chars_per_snippet parameter is optional with default
                                )
                            else:
                                st.error(
                                    "draft_email_structured function not found in search_and_draft module"
                                )
                                draft_result = {
                                    "initial_draft": {
                                        "email_draft": "Function not available"
                                    },
                                    "critic_feedback": {},
                                    "final_draft": {
                                        "email_draft": "Function not available"
                                    },
                                    "selected_attachments": [],
                                    "confidence_score": 0.0,
                                    "metadata": {},
                                }

                            # Save to chat session if active
                            if chat_session:
                                conv_id_for_turn = None
                                if conv_id_filter and len(conv_id_filter) == 1:
                                    conv_id_for_turn = next(iter(conv_id_filter))

                                chat_session.add_message(
                                    "user", query, conv_id=conv_id_for_turn
                                )
                                chat_session.add_message(
                                    "assistant",
                                    draft_result.get("final_draft", {}).get(
                                        "email_draft", ""
                                    ),
                                    conv_id=conv_id_for_turn,
                                )
                                chat_session.save()

                        # Display draft with enhanced formatting
                        confidence = draft_result.get("confidence_score", 0.0)

                        # Confidence indicator with explanation
                        if confidence >= 0.7:
                            st.success(
                                f"✅ High Confidence: {confidence:.2f} - Draft is well-supported by context"
                            )
                        elif confidence >= 0.4:
                            st.warning(
                                f"⚠️ Medium Confidence: {confidence:.2f} - Review carefully before sending"
                            )
                        else:
                            st.error(
                                f"❌ Low Confidence: {confidence:.2f} - Consider refining query or adding context"
                            )

                        # Draft content with better formatting
                        st.subheader("📝 Email Draft")

                        # Add copy button
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            draft_text = draft_result["final_draft"]["email_draft"]
                            st.text_area(
                                "Draft",
                                value=draft_text,
                                height=300,
                                disabled=True,
                                key="draft_display",
                            )

                        with col2:
                            if st.button(
                                "📋 Copy to Clipboard", help="Copy draft to clipboard"
                            ):
                                # Note: Direct clipboard copy requires pyperclip or JavaScript
                                # For now, show the text in a code block for easy copying
                                st.info("Select and copy the text below:")

                            # Show abbreviated version for reference
                            st.caption("Preview:")
                            st.code(draft_text[:100] + "...", language="text")

                        # Metadata
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            meta = draft_result.get("metadata", {})
                            st.metric("Citations", meta.get("citation_count", 0))
                            st.metric("Word Count", meta.get("draft_word_count", 0))

                        with col2:
                            st.metric("Workflow", meta.get("workflow_state", ""))
                            st.metric(
                                "Quality",
                                draft_result.get("critic_feedback", {}).get(
                                    "overall_quality", ""
                                ),
                            )

                        with col3:
                            if draft_result.get("selected_attachments"):
                                st.metric(
                                    "Attachments",
                                    len(draft_result["selected_attachments"]),
                                )

                            # Download button
                            json_data = _format_json(draft_result)
                            st.download_button(
                                "📥 Download JSON",
                                data=json_data,
                                file_name="draft_result.json",
                                mime="application/json",
                            )

                        # Additional details in expander with better organization
                        with st.expander("📋 View Details"):
                            tab1, tab2, tab3, tab4 = st.tabs(
                                [
                                    "Missing Info",
                                    "Attachments",
                                    "Citations",
                                    "Source Conversations",
                                ]
                            )

                            with tab1:
                                missing_info = draft_result.get("final_draft", {}).get(
                                    "missing_information", []
                                )
                                if missing_info:
                                    st.subheader("❓ Missing Information")
                                    for item in missing_info:
                                        st.write(f"• {item}")
                                else:
                                    st.success(
                                        "✅ No missing information - all required details found"
                                    )

                            with tab2:
                                if draft_result.get("selected_attachments"):
                                    st.subheader("📎 Selected Attachments")
                                    for att in draft_result["selected_attachments"]:
                                        col1, col2, col3 = st.columns([3, 1, 1])
                                        with col1:
                                            st.write(f"📄 **{att['filename']}**")
                                            if "path" in att:
                                                st.caption(f"Path: `{att['path']}`")
                                        with col2:
                                            st.metric("Size", f"{att['size_mb']} MB")
                                        with col3:
                                            st.metric(
                                                "Relevance",
                                                f"{att['relevance_score']:.2f}",
                                            )
                                else:
                                    st.info("No attachments selected for this draft")

                            with tab3:
                                citations = draft_result.get("final_draft", {}).get(
                                    "citations", []
                                )
                                if citations:
                                    st.subheader("📚 Citations")
                                    for i, cite in enumerate(citations, 1):
                                        st.markdown(
                                            f"**{i}. {cite.get('document_id', '')}**"
                                        )
                                        st.write(
                                            f"   Fact: {cite.get('fact_cited', '')}"
                                        )
                                        conf = cite.get("confidence", "low")
                                        if conf == "high":
                                            conf_color = "🟢"
                                        elif conf == "medium":
                                            conf_color = "🟡"
                                        else:
                                            conf_color = "🔴"
                                        st.write(f"   Confidence: {conf_color} {conf}")
                                        st.divider()
                                else:
                                    st.info("No citations in this draft")

                            with tab4:
                                st.subheader("📁 Source Conversations Used")
                                # Extract unique conversation IDs from context
                                conv_ids = set()
                                for snippet in st.session_state.get(
                                    "search_results", []
                                ):
                                    conv_id = snippet.get("conv_id", "")
                                    if not conv_id and "::" in snippet.get("id", ""):
                                        conv_id = snippet["id"].split("::")[0]
                                    if conv_id:
                                        conv_ids.add(conv_id)

                                if conv_ids:
                                    for conv_id in sorted(conv_ids):
                                        conv_path = (
                                            Path(st.session_state.export_root) / conv_id
                                        )
                                        if conv_path.exists():
                                            st.success(f"✅ {conv_id}")
                                            st.caption(f"   Path: `{conv_path}`")
                                        else:
                                            st.warning(
                                                f"⚠️ {conv_id} (folder not found)"
                                            )
                                else:
                                    st.info("No conversation folders identified")

                        # Check confidence threshold (outside expander, inside draft block)
                        if draft_result.get("confidence_score", 0) < min_confidence:
                            st.error(
                                f"❌ Draft confidence ({draft_result.get('confidence_score', 0):.2f}) below threshold ({min_confidence:.2f})"
                            )
                            st.info(
                                "Try refining your query or adjusting search parameters."
                            )

                    elif search_and_draft and not sender:
                        st.error("Please provide sender name/email for drafting")

                    # Show validate context quality info
                    if results and not search_only:
                        is_valid, msg = search_module.validate_context_quality(results)
                        if is_valid:
                            st.success(f"✅ Context quality: {msg}")
                        else:
                            st.warning(f"⚠️ Context quality issue: {msg}")

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

    col1, col2 = st.columns(2)

    with col1:
        output_format = st.selectbox("Output Format", ["JSON", "Markdown", "Both"])

    with col2:
        write_to_disk = st.checkbox("Save to Thread Directory", value=True)

    if st.button("📊 Analyze Thread", type="primary", use_container_width=True):
        thread_dir = Path(thread_path)
        convo_file = thread_dir / "Conversation.txt"

        if not convo_file.exists():
            st.error(f"Conversation.txt not found in {thread_dir}")
        elif not modules:
            st.error("Modules not loaded. Please load modules from the sidebar.")
        else:
            try:
                with st.spinner("Analyzing thread with facts ledger..."):
                    summarize_module = modules["summarize_email_thread"]
                    utils_module = modules["utils"]

                    # Read and clean text
                    raw_text = utils_module.read_text_file(convo_file)
                    cleaned_text = utils_module.clean_email_text(raw_text)

                    # Analyze with correct parameters (catalog, temperature are optional)
                    if hasattr(summarize_module, "analyze_email_thread_with_ledger"):
                        analysis = summarize_module.analyze_email_thread_with_ledger(
                            thread_text=cleaned_text,
                            provider=st.session_state.provider,
                            temperature=0.2,  # Add default temperature
                        )
                    else:
                        st.error("analyze_email_thread_with_ledger function not found")
                        analysis = {}

                st.success("✅ Analysis complete")

                # Display summary
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Category", analysis.get("category", "Unknown"))
                    st.metric(
                        "Completeness",
                        f"{analysis.get('_metadata', {}).get('completeness_score', 0)}%",
                    )

                with col2:
                    st.metric("Subject", analysis.get("subject", "")[:50])
                    st.metric("Participants", len(analysis.get("participants", [])))

                # Display key findings
                if analysis.get("summary"):
                    st.subheader("📋 Summary")
                    st.write(analysis["summary"])

                # Facts ledger
                if analysis.get("facts_ledger"):
                    st.subheader("📊 Facts Ledger")
                    facts_data = []
                    for fact in analysis["facts_ledger"]:
                        facts_data.append(
                            {
                                "Fact": fact.get("fact", ""),
                                "Category": fact.get("category", ""),
                                "Status": fact.get("status", ""),
                                "Confidence": f"{fact.get('confidence', 0):.2f}",
                                "Source": fact.get("source_email", "")[:30],
                            }
                        )
                    _display_dataframe(
                        facts_data,
                        columns=["Fact", "Category", "Status", "Confidence", "Source"],
                        max_rows=50,
                    )

                # Action items
                if analysis.get("action_items"):
                    st.subheader("✅ Action Items")
                    for item in analysis["action_items"]:
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.write(f"• {item.get('action', '')}")
                        with col2:
                            st.write(f"Owner: {item.get('owner', 'TBD')}")
                        with col3:
                            st.write(f"Due: {item.get('due_date', 'TBD')}")

                # Save outputs
                if write_to_disk:
                    try:
                        # Save JSON
                        json_path = thread_dir / "thread_analysis.json"
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(
                                analysis, f, ensure_ascii=False, indent=2, default=str
                            )

                        # Save Markdown if requested
                        if output_format in ["Markdown", "Both"]:
                            md_path = thread_dir / "thread_analysis.md"
                            md_content = summarize_module.format_analysis_as_markdown(
                                analysis
                            )
                            with open(md_path, "w", encoding="utf-8") as f:
                                f.write(md_content)

                        st.success(f"✅ Saved analysis to {thread_dir}")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")

                # Download options
                col1, col2 = st.columns(2)

                with col1:
                    json_data = _format_json(analysis)
                    st.download_button(
                        "📥 Download JSON",
                        data=json_data,
                        file_name="thread_analysis.json",
                        mime="application/json",
                    )

                with col2:
                    if hasattr(summarize_module, "format_analysis_as_markdown"):
                        md_content = summarize_module.format_analysis_as_markdown(
                            analysis
                        )
                        st.download_button(
                            "📥 Download Markdown",
                            data=md_content,
                            file_name="thread_analysis.md",
                            mime="text/markdown",
                        )

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.exception(e)

# ---------- DOCTOR TAB ----------
with tabs[5]:
    st.header("🩺 System Doctor")

    doctor_module = modules.get("doctor") if modules else None

    if not doctor_module:
        st.info(
            "The System Doctor module isn't available in this build. Install or restore `emailops.doctor` to enable diagnostics."
        )
    else:
        st.markdown("""
        Run comprehensive diagnostics to check:
        - Dependency installation status
        - Environment variables and API keys
        - Index health and compatibility
        - Embedding provider connectivity
        - Index statistics and recommendations
        """)

        col1, col2 = st.columns(2)

        with col1:
            skip_install_check = st.checkbox(
                "Skip Install Check",
                value=False,
                help="Skip checking for missing dependencies",
            )
            install_missing = st.checkbox(
                "Auto-Install Missing",
                value=False,
                help="Automatically install missing dependencies",
            )

        with col2:
            skip_embed_check = st.checkbox(
                "Skip Embedding Check",
                value=False,
                help="Skip the live embedding connectivity probe",
            )
            log_level = st.selectbox(
                "Log Level",
                ["INFO", "DEBUG", "WARNING", "ERROR"],
                index=0,
                help="Set the logging verbosity",
            )

        doctor_provider = st.selectbox(
            "Provider to Check",
            [
                "(Use Index Provider)",
                "vertex",
                "openai",
                "azure",
                "cohere",
                "huggingface",
                "local",
                "qwen",
            ],
            index=0,
            help="Which embedding provider to check (defaults to the one recorded in the index)",
        )

        if st.button("🔍 Run Diagnostics", type="primary", use_container_width=True):
            if not modules:
                st.error("Modules not loaded. Please load modules from the sidebar.")
            else:
                cmd = [
                    sys.executable,
                    "-m",
                    "emailops.doctor",
                    "--root",
                    st.session_state.export_root,
                    "--log-level",
                    log_level,
                ]

                if doctor_provider != "(Use Index Provider)":
                    cmd.extend(["--provider", doctor_provider])

                if skip_install_check:
                    cmd.append("--skip-install-check")
                if install_missing:
                    cmd.append("--install-missing")
                if skip_embed_check:
                    cmd.append("--skip-embed-check")

                _run_command(
                    cmd,
                    workdir=st.session_state.project_root,
                    title="Running System Doctor",
                )

# ---------- LOGS TAB ----------
with tabs[6]:
    st.header("🪵 Debug Logs")

    if LOG_BUFFER_KEY not in st.session_state:
        st.session_state[LOG_BUFFER_KEY] = deque(maxlen=LOG_BUFFER_MAXLEN)

    if "log_level_filter" not in st.session_state:
        st.session_state["log_level_filter"] = ["INFO", "WARNING", "ERROR", "CRITICAL"]

    if "log_search_term" not in st.session_state:
        st.session_state["log_search_term"] = ""

    level_col, search_col = st.columns([2, 1])
    with level_col:
        selected_levels = st.multiselect(
            "Log Levels",
            options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default=st.session_state["log_level_filter"],
            help="Select which log severity levels to display",
            key="log_level_filter",
        )
    with search_col:
        search_term = st.text_input(
            "Search",
            value=st.session_state["log_search_term"],
            placeholder="Filter logs by text...",
            help="Case-insensitive text filter",
            key="log_search_term",
        )

    buffer_snapshot = list(st.session_state[LOG_BUFFER_KEY])

    controls_col1, controls_col2, controls_col3 = st.columns(3)
    with controls_col1:
        st.button(
            "Refresh Logs",
            use_container_width=True,
            help="Click to manually refresh display",
        )
    with controls_col2:
        if st.button(
            "Clear Logs",
            use_container_width=True,
            help="Empty the in-memory log buffer",
        ):
            st.session_state[LOG_BUFFER_KEY].clear()
            buffer_snapshot = []
    with controls_col3:
        download_text = "\n".join(
            f"{entry['timestamp']} | {entry['level']} | {entry['logger']} | {entry['message']}"
            for entry in buffer_snapshot
        )
        st.download_button(
            "Download Logs",
            data=download_text or "",
            file_name="emailops_ui_logs.txt",
            mime="text/plain",
            use_container_width=True,
            disabled=not buffer_snapshot,
        )

    logs_to_show = []
    for entry in buffer_snapshot:
        if entry["level"] not in selected_levels:
            continue
        if search_term:
            needle = search_term.lower()
            haystack = " ".join(
                [entry["message"], entry["logger"], entry["level"]]
            ).lower()
            if needle not in haystack:
                continue
        logs_to_show.append(entry)

    if logs_to_show:
        formatted = "\n".join(
            f"{entry['timestamp']} | {entry['level']:<7} | {entry['logger']} | {entry['message']}"
            for entry in logs_to_show
        )
        st.code(formatted, language="log")
    else:
        st.info(
            "No log entries match the current filters yet. Interact with the app to generate logs."
        )

# ---------- HELP TAB ----------
with tabs[7]:
    st.header("ℹ️ Help & Documentation")

    st.markdown("""
    ## Quick Start Guide

    ### 1. Initial Setup
    - Set the **Project Root** to your EmailOps directory
    - Set the **Export Root** to your conversation exports directory
    - Configure Google Cloud credentials if using Vertex AI
    - Click **Load/Reload Modules** to initialize

    ### 2. Building an Index
    - Go to the **Index** tab
    - Pick the embedding provider and optional model override
    - Adjust batch size or enable the document limit for smoke tests
    - Click **Start Indexing** to build or refresh the FAISS index

    ### 3. Searching & Drafting
    - Use the **Search & Draft** tab to query your emails
    - Adjust Top-K for more/fewer results
    - Provide sender info to generate draft responses
    - Use filters for targeted searches

    ### 4. Thread Summarization
    - Navigate to **Summarize** tab
    - Point to a conversation directory
    - Get facts-ledger analysis with action items

    ### 5. Diagnostics (Optional)
    - If the System Doctor module is installed, run checks from the **Doctor** tab
    - Review recommendations before contacting support

    ## Key Features

    ### 🚀 Performance Optimizations
    - **Parallel Processing**: Multi-core indexing and chunking
    - **Incremental Updates**: Only index new conversations
    - **Resume Capability**: Continue from interruptions
    - **Chunked Files**: Handle large documents efficiently

    ### 🔧 Advanced Capabilities
    - **Multi-Provider Support**: Vertex, OpenAI, Azure, Cohere, etc.
    - **Chat Mode**: Conversational search interface
    - **Attachment Handling**: Intelligent attachment selection
    - **Facts Ledger**: Track facts and their evolution

    ### 📊 Monitoring
    - **Live Status**: Real-time indexing progress
    - **Rate Analysis**: Estimate completion times
    - **Health Checks**: System diagnostics

    ## Troubleshooting

    ### Common Issues

    **Modules won't load**
    - Verify project root contains 'emailops' folder
    - Check Python environment has all dependencies
    - Run System Doctor (when available) for detailed diagnostics

    **Indexing fails**
    - Ensure export root has conversation folders
    - Check file permissions
    - Verify API credentials are set

    **Search returns no results**
    - Confirm index exists (check Status tab)
    - Verify provider matches index provider
    - Try broader search terms

    **Draft generation errors**
    - Ensure Vertex AI is configured
    - Check API quotas and limits
    - Verify sender information format

    ### Environment Variables

    Required for Vertex AI:
    ```
    GCP_PROJECT=your-project-id
    GCP_REGION=global
    GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
    ```

    Optional:
    ```
    EMBED_PROVIDER=vertex|openai|azure|cohere
    INDEX_DIRNAME={config.INDEX_DIRNAME}
    CHUNK_DIRNAME={config.CHUNK_DIRNAME}
    ```

    ## CLI Commands

    The UI executes these underlying commands:

    ```bash
    # Standard indexing
    python -m emailops.email_indexer --root /path/to/exports --provider vertex --batch 64

    # Document chunking (writes to /path/to/exports/{config.CHUNK_DIRNAME})
    python -m processing.processor chunk --input /path/to/exports --output /path/to/exports --workers 6 --pattern "*.txt"

    # Search and draft
    python -m emailops.search_and_draft --root /path --query "..."

    # Thread summarization
    python -m emailops.summarize_email_thread /path/to/thread
    ```

    ## Support

    For issues or questions:
    - Check the System Doctor tab for diagnostics
    - Review logs in the terminal output
    - Ensure all dependencies are installed
    - Verify environment variables are set correctly
    """)

    # Version info
    st.divider()
    st.subheader("📦 Version Information")

    if modules:
        col1, col2 = st.columns(2)

        with col1:
            st.text("Core Modules:")
            for name in ["utils", "llm_client", "email_indexer", "text_chunker"]:
                if modules.get(name):
                    version = getattr(modules[name], "__version__", "N/A")
                    st.text(f"  {name}: {version}")

        with col2:
            st.text("Optional Modules:")
            for name in ["search_and_draft", "summarize_email_thread", "doctor"]:
                if modules.get(name):
                    version = getattr(modules[name], "__version__", "N/A")
                    st.text(f"  {name}: {version}")
                else:
                    st.text(f"  {name}: Not Available")

# ---------- Footer ----------
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
    EmailOps Dashboard v1.0 | Powered by Vertex AI & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
