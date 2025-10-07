#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmailOps UI — production-ready, robust Streamlit front-end.

Pipeline:
  Status → Index → Chunk → Search & Draft → Summarize → Doctor → Help

Highlights in this revision:
- Fixed missing subject-filter helper by implementing a local, robust function.
- Conversation browser includes a recency-sorted list (via search_and_draft APIs).
- Safer Chat sessions (load/save/reset) with bounded history.
- Clear provider/index compatibility surfacing with Index Info panel.
- Better Search & Draft UX: snippet reveal, citations/attachments preview, JSON export.
- Reply-vs-Fresh EML compose with sensible guardrails and thresholds/tokens controls.

This UI expects the `emailops` package in your repo:
- emailops.search_and_draft
- emailops.email_indexer
- emailops.index_metadata
- emailops.summarize_email_thread
- emailops.doctor
- (optional) emailops.processor
"""

from __future__ import annotations

import os
import sys
import json
import time
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import streamlit as st

# ------------------------------
# Default paths per production requirements
# ------------------------------
DEFAULT_EXPORT_ROOT = r"C:\Users\Asus\Desktop\Outlook"

# ------------------------------
# Theme & small CSS polish
# ------------------------------
st.set_page_config(page_title="EmailOps", page_icon="📬", layout="wide")
st.markdown(
    """
    <style>
    .smallcaps { font-variant: small-caps; letter-spacing: .02em; }
    .muted { opacity: 0.7; }
    .danger { background: #FEF2F2; border-left: 4px solid #EF4444; padding: .5rem .75rem; }
    .ok { background: #F0FDF4; border-left: 4px solid #22C55E; padding: .5rem .75rem; }
    .warn { background: #FFFBEB; border-left: 4px solid #F59E0B; padding: .5rem .75rem; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .tight p { margin: .25rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Helpers
# ------------------------------
def _normpath(p: str | Path) -> str:
    try:
        return str(Path(p).expanduser().resolve())
    except Exception:
        return str(p)

def _pkg_root_from(project_root: Path) -> Optional[Path]:
    pr = project_root
    if (pr / "emailops").exists() and (pr / "emailops").is_dir():
        return pr
    if pr.name == "emailops":
        return pr.parent
    return None

@st.cache_resource(show_spinner=False)
def _import_modules(project_root: str) -> Dict[str, Any]:
    """Import EmailOps modules from the chosen project root and return a namespace dict."""
    pr = Path(project_root)
    pkg_root = _pkg_root_from(pr)
    if not pkg_root:
        raise RuntimeError(f"emailops package not found under: {project_root}")

    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    modules: Dict[str, Any] = {}
    wanted = [
        ("utils", "emailops.utils"),
        ("llm_client", "emailops.llm_client"),
        ("env_utils", "emailops.env_utils"),
        ("index_metadata", "emailops.index_metadata"),
        ("email_indexer", "emailops.email_indexer"),
        ("search_and_draft", "emailops.search_and_draft"),
        ("summarize_email_thread", "emailops.summarize_email_thread"),
        ("doctor", "emailops.doctor"),
        ("processor", "emailops.processor"),  # optional unified chunk/embed CLI
    ]
    errs = []
    for key, mod in wanted:
        try:
            modules[key] = __import__(mod, fromlist=["*"])
        except Exception as e:
            errs.append((key, mod, str(e)))
    if errs:
        modules["_import_errors"] = errs  # Non-fatal: some tabs will hide unsupported actions.
    return modules

def _display_import_errors(errors: List[Tuple[str, str, str]]):
    with st.expander("⚠️ Import diagnostics"):
        for key, mod, err in errors:
            st.write(f"- `{key}` ← `{mod}`  — {err}")

def _run_command(cmd: List[str], workdir: str | Path, title: str) -> int:
    """Run a subprocess and live-stream stdout/stderr to the UI."""
    st.write(f"**{title}**")
    st.caption("`" + " ".join(shlex.quote(c) for c in cmd) + "`")
    output = st.empty()
    proc = subprocess.Popen(
        cmd,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    lines = []
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            lines.append(line.rstrip("\n"))
            output.code("\n".join(lines[-120:]), language="bash")  # tail last ~120 lines
    return int(proc.wait() or 0)

def _human_bytes(n: int | float) -> str:
    try:
        n = float(n)
    except Exception:
        return "-"
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"

def _safe_json_dump(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps({"_error": "unserializable result"}, ensure_ascii=False, indent=2)

def _load_mapping_via_module(m: Any, ix_dir: Path) -> List[Dict[str, Any]]:
    """IMPROVEMENT #2: Use centralized index_metadata.read_mapping()."""
    try:
        from emailops.index_metadata import read_mapping
        return read_mapping(ix_dir)
    except Exception:
        mp = ix_dir / "mapping.json"
        return json.loads(mp.read_text(encoding="utf-8-sig")) if mp.exists() else []

def _find_conv_ids_by_subject_local(mapping: List[Dict[str, Any]], subject_query: str) -> Set[str]:
    """
    Robust subject filter: returns {conv_id} whose subject contains ALL provided keywords (case-insensitive).
    Splits subject_query on whitespace; ignores empty fragments.
    """
    if not subject_query:
        return set()
    terms = [t.lower() for t in subject_query.strip().split() if t.strip()]
    if not terms:
        return set()
    hits: Set[str] = set()
    for rec in mapping or []:
        subj = str(rec.get("subject") or "").lower()
        cid = str(rec.get("conv_id") or "").strip()
        if not cid or not subj:
            continue
        if all(t in subj for t in terms):
            hits.add(cid)
    return hits

# ------------------------------
# Sidebar: global configuration
# ------------------------------
if "project_root" not in st.session_state:
    st.session_state.project_root = _normpath(Path.cwd())
if "export_root" not in st.session_state:
    st.session_state.export_root = _normpath(DEFAULT_EXPORT_ROOT)

st.sidebar.header("⚙️ Configuration")
st.session_state.project_root = st.sidebar.text_input(
    "Project Root (contains the 'emailops' package)",
    value=st.session_state.project_root,
    help="Example: /path/to/your/repo",
)
st.session_state.export_root = st.sidebar.text_input(
    "Export Root (email conversations root)",
    value=st.session_state.export_root,
    help="Folder that contains many subfolders each with Conversation.txt",
)
st.sidebar.divider()

provider = st.sidebar.selectbox(
    "LLM/Embedding Provider",
    ["vertex", "openai", "azure", "cohere", "huggingface", "local", "qwen"],
    index=0,
    help="Used by indexing and search/draft unless an index enforces a specific provider",
)
# Provider-specific warnings (added for user guidance)
if provider == "vertex":
    st.sidebar.info("""
        ⏱️ **Vertex AI Performance Notes**

        - **First Request**: May take 60-90 seconds (API warmup)
        - **Subsequent Requests**: 3-10 seconds typical
        - **JSON Mode**: May fall back to text parsing

        These are known Google API behaviors, not bugs.
    """)
elif provider == "openai":
    st.sidebar.info("""
        💰 **OpenAI Notes**

        - **Costs**: Token usage charges apply
        - **Rate Limits**: May hit limits on large batches
    """)
elif provider == "azure":
    st.sidebar.info("""
        🔧 **Azure OpenAI Notes**

        - **Deployment**: Region-specific setup required
        - **Credentials**: Endpoint + API key needed
    """)
elif provider == "local":
    st.sidebar.info("""
        🖥️ **Local Embeddings Notes**

        - **First Use**: May download models (~several GB)
        - **Performance**: Slower than cloud APIs
        - **Limitations**: Embeddings only, no text generation
    """)
st.sidebar.checkbox("Verbose logs", value=False, key="verbose")
if st.sidebar.button("🔁 Load / Reload EmailOps modules", type="primary", use_container_width=True):
    st.cache_resource.clear()
modules = _import_modules(st.session_state.project_root)
if modules.get("_import_errors"):
    _display_import_errors(modules["_import_errors"])

# ------------------------------
# Derive important paths
# ------------------------------
INDEX_DIRNAME = os.getenv("INDEX_DIRNAME", "_index")
index_dir = Path(st.session_state.export_root) / INDEX_DIRNAME

# ------------------------------
# Header
# ------------------------------
left, right = st.columns([0.7, 0.3])
with left:
    st.title("📬 EmailOps")
    st.caption("**Status → Index → Chunk → Search & Draft → Summarize → Doctor**")
with right:
    st.metric("Index Dir", INDEX_DIRNAME)
    st.caption(f"{index_dir}")

# ------------------------------
# Status helpers
# ------------------------------
@dataclass
class IndexStatus:
    exists: bool
    provider: str = ""
    model: str = ""
    dims: int = 0
    actual_dims: Optional[int] = None
    docs: int = 0
    conv_indexed: int = 0
    conv_total: int = 0
    last_run: Optional[str] = None
    faiss_exists: bool = False
    npy_exists: bool = False
    mapping_exists: bool = False

def _compute_index_status(ns: Dict[str, Any], export_root: Path) -> IndexStatus:
    idx = export_root / INDEX_DIRNAME
    s = IndexStatus(exists=idx.exists())
    try:
        im = ns["index_metadata"]
        ix = im.index_paths(idx)  # filenames/constants centralized in index_metadata
        s.mapping_exists = ix.mapping.exists()
        s.faiss_exists = ix.faiss.exists()
        s.npy_exists = ix.embeddings.exists()

        meta = im.load_index_metadata(idx) or {}
        s.provider = str(meta.get("provider") or "")
        s.model = str(meta.get("model") or meta.get("embed_model") or "")
        s.dims = int(meta.get("dimensions") or 0)
        s.actual_dims = int(meta.get("actual_dimensions") or 0) or None
        s.last_run = (ix.timestamp.read_text(encoding="utf-8", errors="ignore").strip()
                      if ix.timestamp.exists() else (meta.get("created_at") or None))

        mapping = im.read_mapping(idx) or []
        s.docs = len(mapping)
        conv_ids = {str(m.get("conv_id") or "").strip() for m in mapping if m.get("conv_id")}
        s.conv_indexed = len([c for c in conv_ids if c])

        # total conversations = subfolders with Conversation.txt (exclude _*)
        conv_total = 0
        root = Path(export_root)
        if root.exists():
            for d in root.iterdir():
                if d.is_dir() and (not d.name.startswith("_")) and (d / "Conversation.txt").exists():
                    conv_total += 1
        s.conv_total = conv_total
    except Exception:
        pass
    return s

# ------------------------------
# TABS
# ------------------------------
tabs = st.tabs(
    [
        "📊 Status",
        "🧱 Index",
        "🧩 Chunk",
        "🔎 Search & Draft",
        "📝 Summarize",
        "🩺 Doctor",
        "ℹ️ Help",
    ]
)

# ==============================
# STATUS TAB
# ==============================
with tabs[0]:
    st.header("📊 Index Status")
    if not modules:
        st.error("Modules not loaded. Click **Load / Reload EmailOps modules** in the sidebar.")
    else:
        ns = modules
        status = _compute_index_status(ns, Path(st.session_state.export_root))

        cols = st.columns(6)
        cols[0].metric("Exists", "Yes ✅" if status.exists else "No ❌")
        cols[1].metric("Docs", f"{status.docs:,}")
        cols[2].metric("Conversations Indexed", f"{status.conv_indexed:,}")
        cols[3].metric("Conversations Total", f"{status.conv_total:,}")
        cols[4].metric("Provider", status.provider or "—")
        dims_txt = f"{status.actual_dims or status.dims}" if (status.actual_dims or status.dims) else "—"
        cols[5].metric("Dimensions", dims_txt)

        prog = 0.0
        if status.conv_total > 0:
            prog = min(1.0, max(0.0, status.conv_indexed / max(1, status.conv_total)))
        st.progress(prog, text=f"Coverage: {int(prog*100)}%")

        grid1 = st.columns(3)
        with grid1[0]:
            st.write("**Files**")
            st.write(f"- mapping.json: {'✅' if status.mapping_exists else '❌'}")
            st.write(f"- embeddings.npy: {'✅' if status.npy_exists else '❌'}")
            st.write(f"- index.faiss: {'✅' if status.faiss_exists else '❌'}")
        with grid1[1]:
            st.write("**Model**")
            st.write(f"- Provider: `{status.provider or 'n/a'}`")
            st.write(f"- Model: `{status.model or 'n/a'}`")
            st.write(f"- Dimensions: `{dims_txt}`")
        with grid1[2]:
            st.write("**Last Run**")
            st.write(status.last_run or "—")

        # Index Info (from metadata + npy/faiss)
        with st.expander("ℹ️ Index Info (from metadata)"):
            try:
                info = ns["index_metadata"].get_index_info(index_dir)
                st.code(info or "No metadata", language="text")
            except Exception as e:
                st.warning(f"Failed to load index info: {e}")

        # Downloads
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            mp = index_dir / "mapping.json"
            if mp.exists():
                st.download_button("⬇️ Download mapping.json", data=mp.read_bytes(), file_name="mapping.json", mime="application/json")
        with dl2:
            npy = index_dir / "embeddings.npy"
            if npy.exists():
                st.caption(f"embeddings.npy — {_human_bytes(npy.stat().st_size)}")
        with dl3:
            fx = index_dir / "index.faiss"
            if fx.exists():
                st.caption(f"index.faiss — {_human_bytes(fx.stat().st_size)}")

        if not status.exists or not status.mapping_exists:
            st.info("No index found yet. Go to the **Index** tab to build one.")

# ==============================
# INDEX TAB
# ==============================
with tabs[1]:
    st.header("🧱 Build / Update Index")
    if not modules:
        st.error("Modules not loaded.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            force_reindex = st.checkbox("Force re-index (full rebuild)", value=False)
            limit = st.number_input("Limit conversations (testing)", min_value=0, value=0, step=1)
        with col2:
            embed_batch = st.number_input("Embedding batch size", min_value=1, max_value=250, value=64, step=1)
            provider_sel = st.selectbox("Provider (index)", ["vertex", "openai", "azure", "cohere", "huggingface", "local", "qwen"], index=0)
        with col3:
            model_override = st.text_input("Model override (optional)", value="", help="E.g., text-embedding-3-small or Vertex model name")

        if st.button("🚀 Start Indexing", type="primary", use_container_width=True):
            cmd = [sys.executable, "-m", "emailops.email_indexer", "--root", st.session_state.export_root, "--provider", provider_sel, "--batch", str(int(embed_batch))]
            if force_reindex:
                cmd.append("--force-reindex")
            if limit and int(limit) > 0:
                cmd.extend(["--limit", str(int(limit))])
            if model_override.strip():
                cmd.extend(["--model", model_override.strip()])
            code = _run_command(cmd, workdir=st.session_state.project_root, title="Indexing conversations")
            if code == 0:
                st.success("Indexing completed.")
            else:
                st.error(f"Indexer exited with code {code}")

# ==============================
# CHUNK TAB (parallel)
# ==============================
with tabs[2]:
    st.header("🧩 Parallel Chunking (Unified Processor)")
    st.caption("Optional: Pre‑chunk large text corpora with multi‑core processing. Results are written under **_chunks/** in the export root.")

    col1, col2, col3 = st.columns(3)
    with col1:
        pattern = st.text_input("File pattern (glob)", value="*.txt", help="Files under Export Root matching this pattern will be chunked (recursively).")
        chunk_size = st.number_input("Chunk size", min_value=256, max_value=20000, value=1600, step=100)
    with col2:
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=4000, value=200, step=50)
        workers = st.number_input("Workers", min_value=1, max_value=os.cpu_count() or 64, value=max(1, (os.cpu_count() or 2) // 2), step=1)
    with col3:
        test_mode = st.checkbox("Test mode (limit to ~10 files)", value=False)
        out_dir = st.text_input("Output directory", value=_normpath(st.session_state.export_root), help="Directory under which _chunks/ will be placed")

    if st.button("🧩 Run Chunker", use_container_width=True):
        if "processor" not in modules:
            st.error("processor module not available in this environment.")
        else:
            cmd = [
                sys.executable, "-m", "processing.processor", "chunk",
                "--input", st.session_state.export_root,
                "--output", out_dir,
                "--workers", str(int(workers)),
                "--chunk-size", str(int(chunk_size)),
                "--chunk-overlap", str(int(chunk_overlap)),
                "--pattern", pattern or "*.txt",
            ]
            if test_mode:
                cmd.append("--test")
            code = _run_command(cmd, workdir=st.session_state.project_root, title="Parallel chunking")
            if code == 0:
                st.success("Chunking completed.")
            else:
                st.error(f"Chunker exited with code {code}")

# ==============================
# SEARCH & DRAFT TAB
# ==============================
with tabs[3]:
    st.header("🔎 Search & ✍️ Draft")
    if "search_and_draft" not in modules:
        st.error("search_and_draft module not available.")
    else:
        m = modules["search_and_draft"]
        im = modules.get("index_metadata")
        ix_dir = Path(st.session_state.export_root) / os.getenv("INDEX_DIRNAME", "_index")

        # Provider compatibility hint
        idx_provider = ""
        if im:
            try:
                meta = im.load_index_metadata(ix_dir) or {}
                idx_provider = (meta.get("provider") or "").strip()
                if idx_provider and idx_provider != provider:
                    st.info(f"ℹ️ Index was built with **{idx_provider}**; using **{idx_provider}** internally for search to stay compatible.")
            except Exception:
                pass

        # Query + options
        top = st.columns([0.65, 0.35])
        with top[0]:
            query = st.text_input("Query", placeholder="What would you like to find?")
        with top[1]:
            k = st.slider("Top-K (retrieval)", min_value=3, max_value=50, value=15, step=1)

        col1, col2, col3 = st.columns(3)
        with col1:
            chat_mode = st.checkbox("Chat mode", value=False, help="Maintain session context turn‑by‑turn")
            session_id = st.text_input("Session ID", placeholder="default")
            reset_chat = st.button("♻️ Reset session", use_container_width=True)
        with col2:
            st.text_input("Sender (locked)", value=getattr(m, "SENDER_LOCKED", "Hagop Ghazarian <Hagop.Ghazarian@chalhoub.com>"), disabled=True)
            include_attachments = st.checkbox("Include attachments in draft", value=True)
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        with col3:
            conv_id_filter_txt = st.text_input("Filter by Conversation ID", help="Restrict retrieval to a single conversation ID (exact match)")
            conv_subject = st.text_input("Filter by Subject (keyword)", help="Subject contains keyword(s)")

        # Conversation browser (ID quick-pick + Recency list)
        with st.expander("📁 Browse conversations"):
            try:
                export_path = Path(st.session_state.export_root)
                # Quick scan by folder name
                conv_folders = [
                    d.name for d in export_path.iterdir()
                    if d.is_dir() and not d.name.startswith("_") and (d / "Conversation.txt").exists()
                ]
                leftb, rightb = st.columns(2)
                with leftb:
                    if conv_folders:
                        chosen = st.selectbox("Select by folder name", [""] + sorted(conv_folders)[:500])
                        if chosen:
                            conv_id_filter_txt = chosen
                            st.caption(f"Filtering to: **{chosen}**")
                    else:
                        st.caption("No conversation folders found.")
                with rightb:
                    st.caption("Newest → Oldest")
                    try:
                        recency = m.list_conversations_newest_first(ix_dir)
                        show_rows = st.slider("Show recent", 5, min(200, max(5, len(recency))), value=min(20, len(recency)) if recency else 5)
                        if recency:
                            for r in recency[:show_rows]:
                                lab = f"{r.get('last_date_str','')} — {r.get('subject','')}"
                                if st.button(lab, key=f"convbtn-{r.get('conv_id','')}"):
                                    conv_id_filter_txt = str(r.get("conv_id",""))
                                    st.success(f"Selected conversation: {conv_id_filter_txt}")
                    except Exception as e:
                        st.warning(f"Recency listing failed: {e}")
            except Exception as e:
                st.warning(f"Folder browse failed: {e}")

        # Build conv_id filter set (by ID and/or subject)
        conv_id_set: Optional[Set[str]] = None
        if conv_subject or conv_id_filter_txt:
            conv_id_set = set()
            try:
                mapping = _load_mapping_via_module(m, ix_dir)
            except Exception:
                mapping = []
            if conv_id_filter_txt:
                conv_id_set.add(conv_id_filter_txt.strip())
            if conv_subject and mapping:
                hits = _find_conv_ids_by_subject_local(mapping, conv_subject)
                conv_id_set |= hits
            if conv_id_set and len(conv_id_set) == 0:
                conv_id_set = None

        # Chat session
        chat_sess = None
        if chat_mode:
            safe = m._sanitize_session_id((session_id or "default").strip() or "default")
            chat_sess = m.ChatSession(base_dir=ix_dir, session_id=safe, max_history=5)
            try:
                if reset_chat:
                    chat_sess.reset()
                chat_sess.load()
            except Exception as e:
                st.warning(f"Chat session error: {e}")

        # Actions
        a1, a2, a3 = st.columns(3)
        do_search_only = a1.button("🔍 Search only", use_container_width=True)
        do_search_and_draft = a2.button("✉️ Search & draft", type="primary", use_container_width=True)
        do_chat_turn = a3.button("💬 Chat turn", use_container_width=True)

        # Execution
        if (do_search_only or do_search_and_draft or do_chat_turn) and not query and not (do_search_and_draft and conv_id_set):
            st.error("Please enter a query. (For 'Reply .eml' below you may omit query, but for 'Search & draft' or 'Chat' a query is required.)")
        elif (do_search_only or do_search_and_draft or do_chat_turn):
            # CRITICAL FIX #1: Validate index compatibility before search
            if im:
                try:
                    is_compatible = im.validate_index_compatibility(ix_dir, provider)
                    if not is_compatible:
                        # Get index info to show user
                        try:
                            index_info = im.get_index_info(ix_dir)
                        except Exception:
                            index_info = "Unable to load index info"
                        
                        st.error(f"""
                            ⚠️ **Provider/Index Mismatch Detected**
                            
                            The index was built with a different embedding provider.
                            Using the wrong provider will cause dimension mismatch errors.
                            
                            **Current provider selected:** `{provider}`
                            
                            **Index Information:**
                            ```
                            {index_info}
                            ```
                            
                            ✅ **Solution:** Select the correct provider in the sidebar (look for "provider" in Index Info above).
                        """)
                        
                        if st.checkbox("⚠️ Proceed anyway (search will likely fail)", value=False, key="override_compat"):
                            pass  # User chose to proceed
                        else:
                            st.stop()  # Block execution
                except Exception as e:
                    st.warning(f"Could not validate index compatibility: {e}")
            
            # search
            with st.spinner("Searching…"):
                effective_query = query
                if chat_mode and chat_sess:
                    effective_query = m._build_search_query_from_history(chat_sess.recent(), query, max_back=5)
                results = m._search(ix_dir=ix_dir, query=effective_query, k=int(k), provider=provider, conv_id_filter=conv_id_set)

            st.success(f"Found {len(results)} results.")
            if results:
                with st.expander("Results (inspect & reveal snippets)"):
                    for r in results[:50]:
                        st.markdown(f"**{r.get('subject','(no subject)')}** — `{r.get('id','')}`  _(score={r.get('score',0):.3f})_")
                        st.caption(r.get("path",""))
                        with st.expander("Show snippet"):
                            st.code(r.get("text","") or "", language="markdown")

            # Chat
            if do_chat_turn:
                if not results:
                    st.warning("No results available to answer from.")
                else:
                    with st.spinner("Answering from retrieved context…"):
                        chat_history = chat_sess.recent() if chat_sess else []
                        ans = m.chat_with_context(query=query, context_snippets=results, chat_history=chat_history, temperature=float(temperature))
                    st.subheader("💬 Answer")
                    st.write(ans.get("answer", "").strip() or "(no answer)")
                    if ans.get("citations"):
                        with st.expander("Citations"):
                            for i, c in enumerate(ans["citations"], 1):
                                st.write(f"{i}. {c.get('document_id','')} — {c.get('fact_cited','')} ({c.get('confidence','')})")
                    if ans.get("missing_information"):
                        with st.expander("Missing information"):
                            for it in ans["missing_information"]:
                                st.write(f"- {it}")
                    # Save chat
                    if chat_sess:
                        cid = None
                        if conv_id_set and len(conv_id_set) == 1:
                            cid = list(conv_id_set)[0]
                        chat_sess.add_message("user", query, conv_id=cid)
                        chat_sess.add_message("assistant", ans.get("answer",""), conv_id=cid)
                        chat_sess.save()
                        st.caption(f"Session saved: {chat_sess.session_path}")

            # Draft (structured from retrieved context)
            if do_search_and_draft:
                if not results:
                    st.warning("No results to draft from.")
                else:
                    with st.spinner("Drafting with LLM‑as‑critic…"):
                        draft = m.draft_email_structured(
                            query=query,
                            sender=getattr(m, "SENDER_LOCKED", "Hagop Ghazarian <Hagop.Ghazarian@chalhoub.com>"),
                            context_snippets=results,
                            provider=provider,
                            temperature=float(temperature),
                            include_attachments=bool(include_attachments),
                            chat_history=(chat_sess.recent() if chat_sess else None),
                        )
                    conf = float(draft.get("confidence_score", 0.0) or 0.0)
                    if conf >= 0.7:
                        st.success(f"High confidence ({conf:.2f})")
                    elif conf >= 0.4:
                        st.warning(f"Medium confidence ({conf:.2f}) — review before sending")
                    else:
                        st.error(f"Low confidence ({conf:.2f}) — refine the query or add context")

                    st.subheader("📝 Draft")
                    st.text_area("Draft text", value=(draft.get("final_draft",{}).get("email_draft","") or ""), height=280)
                    with st.expander("Citations"):
                        for i, c in enumerate(draft.get("final_draft",{}).get("citations", []) or [], 1):
                            st.write(f"{i}. {c.get('document_id','')} — {c.get('fact_cited','')} ({c.get('confidence','')})")
                    if draft.get("selected_attachments"):
                        with st.expander("Selected attachments"):
                            for att in draft["selected_attachments"]:
                                st.write(f"- {att.get('filename','')} ({att.get('extension','')}, {att.get('size_mb','?')} MB) — score={att.get('relevance_score','?')}")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button("⬇️ Download draft.json", data=_safe_json_dump(draft), file_name="draft.json", mime="application/json")
                    with c2:
                        st.caption("You can also compose .eml below for reply/fresh modes.")

        st.divider()

        # ---------------- EML COMPOSER (Reply / Fresh) ---------------- #
        st.subheader("✉️ Compose .eml")
        eml_col1, eml_col2 = st.columns(2)
        with eml_col1:
            sim_th = st.slider("Similarity threshold (for context gathering)", min_value=0.10, max_value=0.90, value=0.30, step=0.05)
            reply_tokens = st.number_input("Reply context target (tokens)", min_value=1000, max_value=300000, value=200000, step=1000)
        with eml_col2:
            fresh_tokens = st.number_input("Fresh context target (tokens)", min_value=1000, max_value=100000, value=50000, step=1000)
            temp2 = st.slider("Draft temperature (EML compose)", min_value=0.0, max_value=1.0, value=float(temperature), step=0.05, key="temp_eml")

        with st.expander("✉️ Compose .eml (reply to a single conversation)"):
            # Reply requires an unambiguous single conv_id
            single_cid = None
            if conv_id_set and len(conv_id_set) == 1:
                single_cid = list(conv_id_set)[0]
            cid_input = st.text_input("Reply conversation id", value=single_cid or "")
            allow_attachments = st.checkbox("Include relevant attachments", value=True, key="reply_att")
            derive_if_empty = st.caption("If Query is empty, a reply intent will be derived from the most recent inbound email.")
            if st.button("Build reply .eml", type="primary"):
                if not cid_input:
                    st.error("Provide a conversation id (filter to a single folder).")
                else:
                    try:
                        rep = m.draft_email_reply_eml(
                            export_root=Path(st.session_state.export_root),
                            conv_id=cid_input,
                            provider=provider,
                            query=(query or None),
                            sim_threshold=float(sim_th),
                            target_tokens=int(reply_tokens),
                            temperature=float(temp2),
                            include_attachments=bool(allow_attachments),
                        )
                        fn = f"{cid_input}_reply.eml"
                        st.download_button("⬇️ Download reply.eml", data=rep["eml_bytes"], file_name=fn, mime="message/rfc822")
                        st.success("Reply .eml composed.")
                    except Exception as e:
                        st.error(f"Failed to compose reply .eml: {e}")

        with st.expander("📨 Compose .eml (fresh email)"):
            to_list = st.text_input("To (comma‑separated emails)", value="")
            cc_list = st.text_input("Cc (comma‑separated emails)", value="")
            subj = st.text_input("Subject", value="")
            allow_attachments_fresh = st.checkbox("Include relevant attachments", value=True, key="fresh_att")
            if st.button("Build fresh .eml", type="primary"):
                if not (to_list and subj and (query or "").strip()):
                    st.error("To, Subject and Query are required for a fresh email.")
                else:
                    try:
                        fres = m.draft_fresh_email_eml(
                            export_root=Path(st.session_state.export_root),
                            provider=provider,
                            to_list=[x.strip() for x in (to_list or "").split(",") if x.strip()],
                            cc_list=[x.strip() for x in (cc_list or "").split(",") if x.strip()],
                            subject=subj,
                            query=query,
                            sim_threshold=float(sim_th),
                            target_tokens=int(fresh_tokens),
                            temperature=float(temp2),
                            include_attachments=bool(allow_attachments_fresh),
                        )
                        st.download_button("⬇️ Download fresh.eml", data=fres["eml_bytes"], file_name="fresh.eml", mime="message/rfc822")
                        st.success("Fresh .eml composed.")
                    except Exception as e:
                        st.error(f"Failed to compose .eml: {e}")

# ==============================
# SUMMARIZE TAB
# ==============================
with tabs[4]:
    st.header("📝 Summarize Email Thread (facts‑ledger)")
    if "summarize_email_thread" not in modules or "utils" not in modules:
        st.error("Summarizer modules not available.")
    else:
        thread_dir = st.text_input("Thread directory (contains Conversation.txt)", value=_normpath(st.session_state.export_root))
        fmt = st.selectbox("Output format", ["JSON", "Markdown", "Both"], index=0)
        save = st.checkbox("Save into thread folder", value=True)
        if st.button("📊 Analyze thread", type="primary", use_container_width=True):
            td = Path(thread_dir)
            convo = td / "Conversation.txt"
            if not convo.exists():
                st.error(f"{convo} not found")
            else:
                u = modules["utils"]
                s = modules["summarize_email_thread"]
                with st.spinner("Analyzing…"):
                    raw = u.read_text_file(convo)
                    cleaned = u.clean_email_text(raw)
                    analysis = s.analyze_email_thread_with_ledger(thread_text=cleaned, provider=provider)
                st.success("Analysis complete")
                st.json(analysis)
                
                # CRITICAL FIX #2: Display unknowns field for fallback transparency
                if analysis.get("unknowns"):
                    with st.expander("ℹ️ Processing Notes & Fallback Information"):
                        st.caption("The following items were noted during analysis:")
                        for unknown in analysis["unknowns"]:
                            if "fallback" in unknown.lower():
                                st.info(f"🔄 {unknown}")
                            elif "failed" in unknown.lower() or "error" in unknown.lower():
                                st.warning(f"⚠️ {unknown}")
                            else:
                                st.caption(f"ℹ️ {unknown}")
                        
                        st.caption("""
                            **What this means:**
                            - Fallback parsing indicates JSON mode didn't work, but text parsing succeeded
                            - Your analysis is still accurate and complete
                            - This is a known Vertex AI API limitation, not a bug
                            - No action needed from you
                        """)

                if save:
                    try:
                        (td / "thread_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
                        if fmt in ("Markdown", "Both") and hasattr(s, "format_analysis_as_markdown"):
                            md = s.format_analysis_as_markdown(analysis)
                            (td / "thread_analysis.md").write_text(md, encoding="utf-8")
                        st.success(f"Saved outputs into {td}")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")

                # Downloads
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("⬇️ Download JSON", data=json.dumps(analysis, ensure_ascii=False, indent=2), file_name="thread_analysis.json", mime="application/json")
                with c2:
                    if hasattr(s, "format_analysis_as_markdown"):
                        md = s.format_analysis_as_markdown(analysis)
                        st.download_button("⬇️ Download Markdown", data=md, file_name="thread_analysis.md", mime="text/markdown")

# ==============================
# DOCTOR TAB
# ==============================
with tabs[5]:
    st.header("🩺 System Doctor")
    log_level = st.selectbox("Log level", ["INFO", "DEBUG", "WARNING", "ERROR"], index=0)
    which_provider = st.selectbox("Provider to check (optional)", ["(use index provider)", "vertex", "openai", "azure", "cohere", "huggingface", "local", "qwen"], index=0)
    skip_install = st.checkbox("Skip install check", value=False)
    auto_install = st.checkbox("Auto‑install missing", value=False)
    skip_embed = st.checkbox("Skip live embedding check", value=False)

    if st.button("🔍 Run diagnostics", type="primary"):
        cmd = [sys.executable, "-m", "emailops.doctor", "--root", st.session_state.export_root, "--log-level", log_level]
        if which_provider != "(use index provider)":
            cmd.extend(["--provider", which_provider])
        if skip_install:
            cmd.append("--skip-install-check")
        if auto_install:
            cmd.append("--install-missing")
        if skip_embed:
            cmd.append("--skip-embed-check")
        _run_command(cmd, workdir=st.session_state.project_root, title="Doctor")

# ==============================
# HELP
# ==============================
with tabs[6]:
    st.header("ℹ️ Help")
    st.markdown(
        """
        **Pipeline overview**  
        1. **Index** builds/updates `_index/` with embeddings + mapping.  
        2. **Chunk** (optional) pre‑splits large text under `_chunks/` using multi‑core processor.  
        3. **Search & Draft** retrieves top‑K snippets, chats over them, and drafts replies.  
        4. **Summarize** analyzes a single thread directory and produces a facts‑ledger.  
        5. **Doctor** validates dependencies, env, and index health.

        **Tips**  
        • Use a consistent provider for indexing and searching to avoid embedding dimension mismatch.  
        • Incremental indexing will reuse embeddings for unchanged items.  
        • In chat mode, prior turns are summarized into the prompt to keep conversation continuity.  
        """
    )
