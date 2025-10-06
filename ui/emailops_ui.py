#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import os
import sys
import json
import time
import subprocess
import importlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
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

# ---------- Custom CSS for Better UI ----------
st.markdown("""
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
""", unsafe_allow_html=True)

# ---------- Utility Functions ----------
def _normpath(p: str | Path) -> str:
    """Normalize and resolve path."""
    try:
        return str(Path(p).expanduser().resolve())
    except Exception:
        return str(p)

def _validate_path(path: Path, must_exist: bool = True, is_dir: bool = True) -> Tuple[bool, str]:
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

def _pkg_root_from(project_root: Path) -> Optional[Path]:
    """Find the package root for importing emailops modules."""
    pr = project_root
    if (pr / "emailops").exists() and (pr / "emailops").is_dir():
        return pr
    if pr.name == "emailops":
        return pr.parent
    return None

@st.cache_resource(show_spinner=False)
def _import_modules(project_root: str) -> Dict[str, Any]:
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
    
    modules = {}
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
                modules[name] = None  # Optional modules
            else:
                raise ImportError(f"Failed to import required module {module_path}: {e}")
    
    return modules

def _format_json(obj: Any) -> str:
    """Format object as JSON string."""
    try:
        if hasattr(obj, 'to_dict'):
            obj = obj.to_dict()
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False, indent=2)

def _run_command(cmd: List[str], workdir: str | None = None, title: str = "Running Command"):
    """Run a command and stream output with better error handling."""
    with st.status(title, expanded=True) as status:
        st.code(" ".join(cmd), language="bash")
        
        try:
            env = os.environ.copy()
            # Ensure EMBED_PROVIDER is set for search operations
            if "EMBED_PROVIDER" not in env:
                env["EMBED_PROVIDER"] = st.session_state.get("provider", "vertex")
            
            proc = subprocess.Popen(
                cmd,
                cwd=workdir or None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                env=env,
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
                    # Show last 500 lines to keep UI responsive
                    display_lines = lines[-500:]
                    log_area.code("\n".join(display_lines), language="log")
        
        rc = proc.poll()
        if rc == 0:
            status.update(label="✅ Completed Successfully", state="complete")
        else:
            status.update(label=f"❌ Failed with exit code {rc}", state="error")
        
        return rc

def _display_dataframe(data: List[Dict[str, Any]], columns: List[str] | None = None, 
                       max_rows: int = 100, title: str | None = None):
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
        height=min(400, 35 * len(df))  # Adaptive height
    )

# ---------- Session State Initialization ----------
if "provider" not in st.session_state:
    st.session_state.provider = "vertex"
if "project_root" not in st.session_state:
    st.session_state.project_root = str(Path.cwd())
if "export_root" not in st.session_state:
    st.session_state.export_root = r"C:\Users\ASUS\Desktop\Outlook"
if "modules" not in st.session_state:
    st.session_state.modules = None

# ---------- Sidebar Configuration ----------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Path Configuration
    st.subheader("📁 Paths")
    project_root = st.text_input(
        "Project Root",
        value=st.session_state.project_root,
        help="Directory containing the 'emailops' package",
        key="project_root_input"
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
        "Export Root",
        value=st.session_state.export_root,
        help="Directory containing conversation folders",
        key="export_root_input"
    )
    st.session_state.export_root = export_root
    
    # Validate export root
    valid, msg = _validate_path(Path(export_root), must_exist=True, is_dir=True)
    if valid:
        st.success("✅ Valid export root")
    else:
        st.error(f"❌ {msg}")
    
    st.divider()
    
    # Provider Configuration
    st.subheader("🔧 Provider Settings")
    provider = st.selectbox(
        "Embedding Provider",
        ["vertex", "openai", "azure", "cohere", "huggingface", "local", "qwen"],
        index=0,
        help="Provider for embedding operations. Generation uses Vertex AI.",
        key="provider_select"
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
            help="Your Google Cloud project ID"
        )
        gcp_region = st.text_input(
            "GCP_REGION",
            value=os.environ.get("GCP_REGION", "global"),
            help="GCP region for Vertex AI"
        )
        credentials_path = st.text_input(
            "GOOGLE_APPLICATION_CREDENTIALS",
            value=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
            help="Path to service account JSON file"
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
        st.info("Please check your project root path in the sidebar and ensure all dependencies are installed.")
        st.stop()
else:
    modules = st.session_state.modules

# ---------- Main Tabs ----------
tabs = st.tabs([
    "📊 Status",
    "🔍 Index",
    "📄 Chunk",
    "🔎 Search & Draft",
    "📝 Summarize",
    "🩺 Doctor",
    "ℹ️ Help"
])

# ---------- STATUS TAB ----------
with tabs[0]:
    st.header("📊 Index Status")
    
    if modules and modules.get("vertex_utils"):
        try:
            monitor_class = modules["vertex_utils"].IndexingMonitor
            monitor = monitor_class(st.session_state.export_root)
            status = monitor.check_status(emit_text=False)
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Documents Indexed",
                    f"{status.documents_indexed:,}",
                    delta=None if status.documents_indexed == 0 else "✓"
                )
            
            with col2:
                st.metric(
                    "Conversations",
                    f"{status.conversations_indexed:,} / {status.conversations_total:,}",
                    delta=f"{status.progress_percent:.1f}%" if status.conversations_total > 0 else None
                )
            
            with col3:
                st.metric(
                    "Progress",
                    f"{status.progress_percent:.1f}%",
                    delta="Active" if status.is_active else "Idle"
                )
            
            with col4:
                st.metric(
                    "Index Status",
                    "✅ Exists" if status.index_exists else "❌ Not Found",
                    delta=status.index_type if status.index_exists else None
                )
            
            # Display metadata
            if status.index_exists:
                st.subheader("Index Metadata")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    metadata = {
                        "Provider": status.provider or "Unknown",
                        "Model": status.model or "Unknown",
                        "Dimensions": status.actual_dimensions or "Unknown",
                        "Index Type": status.index_type or "Unknown"
                    }
                    for key, value in metadata.items():
                        st.text(f"{key}: {value}")
                
                with col2:
                    if status.last_updated:
                        try:
                            last_update = datetime.fromisoformat(status.last_updated)
                            st.text(f"Last Updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
                        except:
                            st.text(f"Last Updated: {status.last_updated}")
                    
                    if status.index_file:
                        st.text(f"Index File: {status.index_file}")
                        if status.index_file_size_mb:
                            st.text(f"Size: {status.index_file_size_mb:.1f} MB")
                
                # Rate analysis
                if st.button("📈 Analyze Indexing Rate"):
                    rate_info = monitor.analyze_rate(emit_text=False)
                    if rate_info:
                        st.subheader("Indexing Rate Analysis")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Rate", f"{rate_info.get('rate_per_hour', 0):.1f} items/hour")
                        with col2:
                            st.metric("Remaining", f"{rate_info.get('remaining_conversations', 0):,}")
                        with col3:
                            st.metric("ETA", f"{rate_info.get('eta_hours', 0):.1f} hours")
        
        except Exception as e:
            st.error(f"Failed to load status: {e}")
    else:
        st.warning("Vertex utils module not available")

# ---------- INDEX TAB ----------
with tabs[1]:
    st.header("🔍 Build/Update Index")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        index_tool = st.selectbox(
            "Indexing Tool",
            ["vertex_indexer (Parallel)", "email_indexer (Standard)"],
            index=0,
            help="Choose between parallel Vertex indexer or standard indexer"
        )
        mode = st.selectbox("Execution Mode", ["parallel", "sequential"], index=0)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=250, value=64,
                                     help="Max 250 for Vertex AI Gemini embeddings")
    
    with col2:
        resume = st.checkbox("Resume from Previous", value=True)
        incremental = st.checkbox("Incremental Update", value=False)
        force_rebuild = st.checkbox("Force Full Rebuild", value=False)
    
    with col3:
        test_mode = st.checkbox("Test Mode", value=False)
        test_chunks = st.number_input(
            "Test Chunks/Limit",
            min_value=1,
            max_value=1000,
            value=100,
            disabled=not test_mode
        )
        chunked_files = st.checkbox("Use Chunked Files", value=False)
    
    if st.button("🚀 Start Indexing", type="primary", use_container_width=True):
        if "vertex_indexer" in index_tool:
            # Use vertex_indexer with corrected parameters
            cmd = [
                sys.executable, "-m", "vertex_indexer",
                "--root", st.session_state.export_root,  # Fixed: was --export-dir
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
            # Use standard email_indexer
            cmd = [
                sys.executable, "-m", "emailops.email_indexer",
                "--root", st.session_state.export_root,  # Fixed: uses --root
            ]
            
            if incremental:
                cmd.append("--incremental")
            if test_mode:
                cmd.extend(["--limit", str(test_chunks)])
        
        _run_command(cmd, workdir=st.session_state.project_root, title="Running Indexer")

# ---------- CHUNK TAB ----------
with tabs[2]:
    st.header("📄 Document Chunking")
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_dir = st.text_input(
            "Input Directory",
            value=st.session_state.export_root,
            help="Directory containing documents to chunk"
        )
    
    with col2:
        output_dir = st.text_input(
            "Output Directory",
            value=str(Path(st.session_state.export_root) / "_chunks"),
            help="Directory for chunked output"
        )
    
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
    
    advanced = st.expander("Advanced Options")
    with advanced:
        col1, col2 = st.columns(2)
        with col1:
            no_resume = st.checkbox("Don't Resume", value=False)
            test_mode_chunk = st.checkbox("Test Mode", value=False, key="chunk_test_mode")
            no_clear = st.checkbox("Don't Clear Screen", value=True)
        
        with col2:
            test_files = st.number_input(
                "Test Files",
                min_value=1,
                value=100,
                disabled=not test_mode_chunk
            )
            max_chars = st.number_input(
                "Max Characters per File",
                min_value=0,
                value=0,
                help="0 = unlimited"
            )
    
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
        "Query",
        placeholder="Enter your search query or draft request...",
        height=100
    )
    
    # Basic settings
    col1, col2 = st.columns(2)
    
    with col1:
        k = st.slider("Top-K Results", min_value=1, max_value=200, value=60, step=5,
                     help="Number of relevant conversations to retrieve")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05,
                              help="Higher = more creative, Lower = more factual")
    
    with col2:
        sender = st.text_input(
            "Sender Name/Email",
            placeholder="John Doe <john@example.com>",
            help="Required for drafting email responses"
        )
        include_attachments = st.checkbox("Include Attachments", value=True,
                                         help="Automatically select relevant attachments")
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            chat_mode = st.checkbox("Chat Mode", value=False,
                                   help="Enable conversational Q&A instead of email drafting")
            session_id = st.text_input(
                "Session ID",
                placeholder="Leave empty for new session",
                disabled=not chat_mode,
                help="Maintain conversation context across queries"
            )
            
            # Add session management buttons
            if chat_mode:
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🔄 Reset Session", disabled=not session_id):
                        if hasattr(st.session_state, 'chat_session'):
                            st.session_state.chat_session.reset()
                            st.session_state.chat_session.save()
                            st.success("Session reset!")
                            del st.session_state.chat_session
                            del st.session_state.current_session_id
                
                with col_b:
                    if st.button("📋 List Sessions"):
                        sessions_dir = Path(st.session_state.export_root) / "_index" / "_chat_sessions"
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
                help="Search only within a specific conversation folder"
            )
            conv_subject = st.text_input(
                "Subject Filter",
                placeholder="e.g., contract, invoice",
                help="Filter by keywords in email subject"
            )
            
            # Add confidence threshold for drafting
            min_confidence = st.slider(
                "Minimum Draft Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Reject drafts below this confidence score"
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
                        d.name for d in export_path.iterdir()
                        if d.is_dir() and not d.name.startswith('_')
                        and (d / "Conversation.txt").exists()
                    ]
                    
                    if conv_folders:
                        selected_folder = st.selectbox(
                            "Select conversation folder:",
                            [""] + sorted(conv_folders)[:100],  # Limit to 100 for performance
                            help="Select a specific conversation to search within"
                        )
                        if selected_folder:
                            conv_id = selected_folder
                            st.success(f"✅ Will search in: {selected_folder}")
                    else:
                        st.warning("No conversation folders found with Conversation.txt")
                else:
                    st.error(f"Export root does not exist: {export_path}")
            except Exception as e:
                st.error(f"Error browsing folders: {e}")
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        search_only = st.button("🔍 Search Only", use_container_width=True)
    
    with col2:
        search_and_draft = st.button("✉️ Search & Draft", use_container_width=True, type="primary")
    
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
                    index_dir = Path(st.session_state.export_root) / os.getenv("INDEX_DIRNAME", "_index")
                    
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
                            hits = search_module._find_conv_ids_by_subject(mapping, conv_subject)
                            conv_id_filter |= hits
                        
                        if not conv_id_filter:
                            st.info("No conversations matched the provided filters.")
                            conv_id_filter = None  # Allow search to continue without filter
                    
                    # Handle chat mode with session
                    if chat_mode:
                        # Initialize or load chat session
                        if not hasattr(st.session_state, 'chat_session') or (session_id and st.session_state.get('current_session_id') != session_id):
                            if session_id:
                                safe_id = search_module._sanitize_session_id(session_id)
                                chat_session = search_module.ChatSession(
                                    base_dir=index_dir,
                                    session_id=safe_id,
                                    max_history=10
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
                                    max_history=10
                                )
                                st.session_state.chat_session = chat_session
                                st.session_state.current_session_id = new_session_id
                        else:
                            chat_session = st.session_state.chat_session
                        
                        # Build effective query with history
                        hist_for_query = chat_session.recent()
                        effective_query = search_module._build_search_query_from_history(
                            hist_for_query, query, max_back=5
                        ) if query else query
                    else:
                        effective_query = query
                        chat_session = None
                    
                    # Perform search with correct parameters
                    results = search_module._search(
                        ix_dir=index_dir,
                        query=effective_query,
                        k=k,
                        provider=st.session_state.provider,
                        conv_id_filter=conv_id_filter
                    )
                
                if not results:
                    st.warning("No results found. Try adjusting your query or increasing K.")
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
                            conv_id = r.get("id", "").split("::")[0] if "::" in r.get("id", "") else "unknown"
                        
                        if conv_id not in conv_groups:
                            conv_groups[conv_id] = []
                        conv_groups[conv_id].append(r)
                    
                    # Display grouped results
                    for conv_id, items in conv_groups.items():
                        # Build conversation folder path
                        conv_path = Path(st.session_state.export_root) / conv_id
                        path_exists = conv_path.exists() and (conv_path / "Conversation.txt").exists()
                        
                        # Create expander for each conversation
                        status_icon = "✅" if path_exists else "⚠️"
                        first_item = items[0]
                        subject = first_item.get("subject", "No subject")[:80]
                        
                        with st.expander(f"{status_icon} **{conv_id}** - {subject} ({len(items)} items)"):
                            # Show conversation info
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**Subject:** {first_item.get('subject', 'N/A')}")
                                st.markdown(f"**Date:** {first_item.get('date', 'N/A')}")
                                st.markdown(f"**From:** {first_item.get('from_name', '')} <{first_item.get('from_email', '')}>")
                                
                                # Show full path
                                if path_exists:
                                    st.markdown(f"**📁 Path:** `{conv_path}`")
                                    
                                    # Add button to open folder (Windows only)
                                    if os.name == 'nt':
                                        if st.button(f"Open Folder", key=f"open_{conv_id}"):
                                            try:
                                                os.startfile(str(conv_path))
                                            except Exception as e:
                                                st.error(f"Could not open folder: {e}")
                                else:
                                    st.warning(f"⚠️ Conversation folder not found at: {conv_path}")
                            
                            with col2:
                                # Show score distribution
                                scores = [float(item.get('score', 0)) for item in items]
                                st.metric("Avg Score", f"{sum(scores)/len(scores):.3f}")
                                st.metric("Items", len(items))
                            
                            # Show individual items in this conversation
                            st.markdown("**Items in this conversation:**")
                            for item in items:
                                doc_type = item.get("doc_type", "")
                                icon = "📎" if doc_type == "attachment" else "📧"
                                score = float(item.get('score', 0))
                                
                                # Format item display
                                item_id = item.get("id", "").split("::")[-1] if "::" in item.get("id", "") else item.get("id", "")
                                snippet = item.get("text", "")[:200] + "..." if len(item.get("text", "")) > 200 else item.get("text", "")
                                
                                st.markdown(f"{icon} **{item_id}** (score: {score:.3f})")
                                st.markdown(f"   {snippet}")
                                
                                # Show attachment info if present
                                if item.get("attachment_name"):
                                    st.markdown(f"   📎 Attachment: {item.get('attachment_name')} ({item.get('attachment_type', 'unknown')})")
                                
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
                        avg_score = sum(float(r.get('score', 0)) for r in results) / len(results) if results else 0
                        st.metric("Average Score", f"{avg_score:.3f}")
                        
                        doc_types = {}
                        for r in results:
                            dtype = r.get("doc_type", "unknown")
                            doc_types[dtype] = doc_types.get(dtype, 0) + 1
                        st.metric("Document Types", ", ".join(f"{k}: {v}" for k, v in doc_types.items()))
                    
                    with col3:
                        valid_paths = sum(1 for conv_id in conv_groups.keys()
                                        if (Path(st.session_state.export_root) / conv_id).exists())
                        st.metric("Valid Folders", f"{valid_paths}/{len(conv_groups)}")
                        
                        date_range = []
                        for r in results:
                            if r.get("date"):
                                date_range.append(r.get("date"))
                        if date_range:
                            st.metric("Date Range", f"{min(date_range)[:10]} to {max(date_range)[:10]}")
                    
                    # Handle chat mode or drafting
                    if chat_mode and not sender:
                        # Chat mode - Q&A without drafting
                        if results:
                            with st.spinner("Generating chat response..."):
                                # Get chat history for context
                                chat_history = chat_session.recent() if chat_session else []
                                
                                # Call chat_with_context function
                                chat_result = search_module.chat_with_context(
                                    query=query,
                                    context_snippets=results,
                                    chat_history=chat_history,
                                    temperature=temperature
                                )
                                
                                # Save to session
                                if chat_session:
                                    # Determine conv_id if filter identifies single conversation
                                    conv_id_for_turn = None
                                    if conv_id_filter and len(conv_id_filter) == 1:
                                        conv_id_for_turn = list(conv_id_filter)[0]
                                    
                                    chat_session.add_message("user", query, conv_id=conv_id_for_turn)
                                    chat_session.add_message("assistant", chat_result.get("answer", ""), conv_id=conv_id_for_turn)
                                    chat_session.save()
                                
                                # Display chat response
                                st.subheader("💬 Chat Response")
                                
                                # Show session info
                                if chat_session:
                                    st.info(f"Session ID: {st.session_state.current_session_id} | Messages: {len(chat_session.messages)}")
                                
                                # Display answer
                                st.markdown("### Answer")
                                st.write(chat_result.get("answer", ""))
                                
                                # Display citations
                                citations = chat_result.get("citations", [])
                                if citations:
                                    with st.expander(f"📚 Citations ({len(citations)})"):
                                        for i, cite in enumerate(citations, 1):
                                            st.markdown(f"**{i}. {cite.get('document_id', '')}**")
                                            st.write(f"   Fact: {cite.get('fact_cited', '')}")
                                            conf = cite.get('confidence', 'low')
                                            conf_color = "🟢" if conf == "high" else "🟡" if conf == "medium" else "🔴"
                                            st.write(f"   Confidence: {conf_color} {conf}")
                                
                                # Display missing information
                                missing_info = chat_result.get("missing_information", [])
                                if missing_info:
                                    with st.expander("❓ Missing Information"):
                                        for item in missing_info:
                                            st.write(f"• {item}")
                                
                                # Show chat history
                                if chat_session and chat_session.messages:
                                    with st.expander("💬 Chat History"):
                                        for msg in chat_session.messages[-10:]:  # Show last 10 messages
                                            role_icon = "👤" if msg.role == "user" else "🤖"
                                            st.markdown(f"**{role_icon} {msg.role.title()}**")
                                            st.write(msg.content[:500])  # Truncate long messages
                                            st.caption(f"Time: {msg.timestamp}")
                                            if msg.conv_id:
                                                st.caption(f"Conversation: {msg.conv_id}")
                                            st.divider()
                        else:
                            st.warning("No results found for chat. Try adjusting your query.")
                    
                    elif search_and_draft and sender:
                        # Email drafting mode
                        with st.spinner("Drafting email with LLM-as-critic..."):
                            # Get chat history if available
                            chat_history = None
                            if chat_session:
                                chat_history = chat_session.recent()
                            
                            draft_result = search_module.draft_email_structured(
                                query=query,
                                sender=sender,
                                context_snippets=results,
                                provider=st.session_state.provider,
                                temperature=temperature,
                                include_attachments=include_attachments,
                                chat_history=chat_history
                            )
                            
                            # Save to chat session if active
                            if chat_session:
                                conv_id_for_turn = None
                                if conv_id_filter and len(conv_id_filter) == 1:
                                    conv_id_for_turn = list(conv_id_filter)[0]
                                
                                chat_session.add_message("user", query, conv_id=conv_id_for_turn)
                                chat_session.add_message("assistant", draft_result.get("final_draft", {}).get("email_draft", ""), conv_id=conv_id_for_turn)
                                chat_session.save()
                        
                        # Display draft with enhanced formatting
                        confidence = draft_result.get("confidence_score", 0.0)
                        
                        # Confidence indicator with explanation
                        if confidence >= 0.7:
                            st.success(f"✅ High Confidence: {confidence:.2f} - Draft is well-supported by context")
                        elif confidence >= 0.4:
                            st.warning(f"⚠️ Medium Confidence: {confidence:.2f} - Review carefully before sending")
                        else:
                            st.error(f"❌ Low Confidence: {confidence:.2f} - Consider refining query or adding context")
                        
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
                                key="draft_display"
                            )
                        
                        with col2:
                            if st.button("📋 Copy to Clipboard",
                                       help="Copy draft to clipboard"):
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
                            st.metric("Quality", draft_result.get("critic_feedback", {}).get("overall_quality", ""))
                        
                        with col3:
                            if draft_result.get("selected_attachments"):
                                st.metric("Attachments", len(draft_result["selected_attachments"]))
                            
                            # Download button
                            json_data = _format_json(draft_result)
                            st.download_button(
                                "📥 Download JSON",
                                data=json_data,
                                file_name="draft_result.json",
                                mime="application/json"
                            )
                        
                        # Additional details in expander with better organization
                        with st.expander("📋 View Details"):
                            tab1, tab2, tab3, tab4 = st.tabs(["Missing Info", "Attachments", "Citations", "Source Conversations"])
                            
                            with tab1:
                                missing_info = draft_result.get("final_draft", {}).get("missing_information", [])
                                if missing_info:
                                    st.subheader("❓ Missing Information")
                                    for item in missing_info:
                                        st.write(f"• {item}")
                                else:
                                    st.success("✅ No missing information - all required details found")
                            
                            with tab2:
                                if draft_result.get("selected_attachments"):
                                    st.subheader("📎 Selected Attachments")
                                    for att in draft_result["selected_attachments"]:
                                        col1, col2, col3 = st.columns([3, 1, 1])
                                        with col1:
                                            st.write(f"📄 **{att['filename']}**")
                                            if 'path' in att:
                                                st.caption(f"Path: `{att['path']}`")
                                        with col2:
                                            st.metric("Size", f"{att['size_mb']} MB")
                                        with col3:
                                            st.metric("Relevance", f"{att['relevance_score']:.2f}")
                                else:
                                    st.info("No attachments selected for this draft")
                            
                            with tab3:
                                citations = draft_result.get("final_draft", {}).get("citations", [])
                                if citations:
                                    st.subheader("📚 Citations")
                                    for i, cite in enumerate(citations, 1):
                                        st.markdown(f"**{i}. {cite.get('document_id', '')}**")
                                        st.write(f"   Fact: {cite.get('fact_cited', '')}")
                                        conf = cite.get('confidence', 'low')
                                        conf_color = "🟢" if conf == "high" else "🟡" if conf == "medium" else "🔴"
                                        st.write(f"   Confidence: {conf_color} {conf}")
                                        st.divider()
                                else:
                                    st.info("No citations in this draft")
                            
                            with tab4:
                                st.subheader("📁 Source Conversations Used")
                                # Extract unique conversation IDs from context
                                conv_ids = set()
                                for snippet in st.session_state.get("search_results", []):
                                    conv_id = snippet.get("conv_id", "")
                                    if not conv_id and "::" in snippet.get("id", ""):
                                        conv_id = snippet["id"].split("::")[0]
                                    if conv_id:
                                        conv_ids.add(conv_id)
                                
                                if conv_ids:
                                    for conv_id in sorted(conv_ids):
                                        conv_path = Path(st.session_state.export_root) / conv_id
                                        if conv_path.exists():
                                            st.success(f"✅ {conv_id}")
                                            st.caption(f"   Path: `{conv_path}`")
                                        else:
                                            st.warning(f"⚠️ {conv_id} (folder not found)")
                                else:
                                    st.info("No conversation folders identified")
                        
                        # Check confidence threshold (outside expander, inside draft block)
                        if draft_result.get("confidence_score", 0) < min_confidence:
                            st.error(f"❌ Draft confidence ({draft_result.get('confidence_score', 0):.2f}) below threshold ({min_confidence:.2f})")
                            st.info("Try refining your query or adjusting search parameters.")
                    
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
        help="Directory containing Conversation.txt"
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
                    
                    # Analyze
                    analysis = summarize_module.analyze_email_thread_with_ledger(
                        thread_text=cleaned_text,
                        provider=st.session_state.provider
                    )
                
                st.success("✅ Analysis complete")
                
                # Display summary
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Category", analysis.get("category", "Unknown"))
                    st.metric("Completeness", f"{analysis.get('_metadata', {}).get('completeness_score', 0)}%")
                
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
                        facts_data.append({
                            "Fact": fact.get("fact", ""),
                            "Category": fact.get("category", ""),
                            "Status": fact.get("status", ""),
                            "Confidence": f"{fact.get('confidence', 0):.2f}",
                            "Source": fact.get("source_email", "")[:30]
                        })
                    _display_dataframe(
                        facts_data,
                        columns=["Fact", "Category", "Status", "Confidence", "Source"],
                        max_rows=50
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
                            json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
                        
                        # Save Markdown if requested
                        if output_format in ["Markdown", "Both"]:
                            md_path = thread_dir / "thread_analysis.md"
                            md_content = summarize_module.format_analysis_as_markdown(analysis)
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
                        mime="application/json"
                    )
                
                with col2:
                    if hasattr(summarize_module, 'format_analysis_as_markdown'):
                        md_content = summarize_module.format_analysis_as_markdown(analysis)
                        st.download_button(
                            "📥 Download Markdown",
                            data=md_content,
                            file_name="thread_analysis.md",
                            mime="text/markdown"
                        )
            
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.exception(e)

# ---------- DOCTOR TAB ----------
with tabs[5]:
    st.header("🩺 System Doctor")
    
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
        skip_install_check = st.checkbox("Skip Install Check", value=False,
                                        help="Skip checking for missing dependencies")
        install_missing = st.checkbox("Auto-Install Missing", value=False,
                                     help="Automatically install missing dependencies")
    
    with col2:
        skip_embed_check = st.checkbox("Skip Embedding Check", value=False,
                                      help="Skip the live embedding connectivity probe")
        log_level = st.selectbox("Log Level",
                                ["INFO", "DEBUG", "WARNING", "ERROR"],
                                index=0,
                                help="Set the logging verbosity")
    
    # Provider selection for doctor
    doctor_provider = st.selectbox(
        "Provider to Check",
        ["(Use Index Provider)", "vertex", "openai", "azure", "cohere", "huggingface", "local", "qwen"],
        index=0,
        help="Which embedding provider to check (defaults to the one recorded in the index)"
    )
    
    if st.button("🔍 Run Diagnostics", type="primary", use_container_width=True):
        if not modules:
            st.error("Modules not loaded. Please load modules from the sidebar.")
        else:
            cmd = [
                sys.executable, "-m", "emailops.doctor",
                "--root", st.session_state.export_root,
                "--log-level", log_level,
            ]
            
            # Add provider if specified
            if doctor_provider != "(Use Index Provider)":
                cmd.extend(["--provider", doctor_provider])
            
            # Add optional flags
            if skip_install_check:
                cmd.append("--skip-install-check")
            if install_missing:
                cmd.append("--install-missing")
            if skip_embed_check:
                cmd.append("--skip-embed-check")
            
            _run_command(cmd, workdir=st.session_state.project_root, title="Running System Doctor")

# ---------- HELP TAB ----------
with tabs[6]:
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
    - Choose between parallel (faster) or standard indexing
    - Select **Incremental Update** for adding new emails
    - Click **Start Indexing** to begin
    
    ### 3. Searching & Drafting
    - Use the **Search & Draft** tab to query your emails
    - Adjust Top-K for more/fewer results
    - Provide sender info to generate draft responses
    - Use filters for targeted searches
    
    ### 4. Thread Summarization
    - Navigate to **Summarize** tab
    - Point to a conversation directory
    - Get facts-ledger analysis with action items
    
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
    - Run System Doctor for detailed diagnostics
    
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
    INDEX_DIRNAME=_index
    CHUNK_DIRNAME=_chunks
    ```
    
    ## CLI Commands
    
    The UI executes these underlying commands:
    
    ```bash
    # Parallel indexing
    python -m vertex_indexer --root /path/to/exports --mode parallel
    
    # Standard indexing  
    python -m emailops.email_indexer --root /path/to/exports
    
    # Document chunking
    python -m parallel_chunker --input-dir /path --output-dir /chunks
    
    # Search and draft
    python -m emailops.search_and_draft --root /path --query "..."
    
    # Thread summarization
    python -m emailops.summarize_email_thread /path/to/thread
    
    # System diagnostics
    python -m emailops.doctor --root /path --check-all
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
            for name in ["utils", "llm_client", "email_indexer"]:
                if modules.get(name):
                    version = getattr(modules[name], "__version__", "N/A")
                    st.text(f"  {name}: {version}")
        
        with col2:
            st.text("Optional Modules:")
            for name in ["vertex_indexer", "vertex_utils"]:
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
    unsafe_allow_html=True
)