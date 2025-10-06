#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmailOps UI — zero‑placeholder, single‑file Streamlit dashboard.

What this gives you:
- Status: live index health (docs, conversations, percent, last update)
- Index: one‑click incremental or full reindex (shows streaming logs)
- Search: query top‑K context with boosted recency (shows scored snippets)
- Draft: structured email draft + critic pass + attachments + confidence
- Summarize: facts‑ledger analysis for a single thread (JSON + save)
- Doctor: dependency + provider + index compatibility checks (logs)

Assumptions:
- Your project has a Python package directory named `emailops` that contains
  the modules referenced below (this matches your repo).
- You already control credentials/environment for your chosen provider.
- No placeholders are required: paths are selected in the UI and env is
  read from the actual OS environment or set via the sidebar.

Run:
    pip install -r requirements_ui.txt
    streamlit run emailops_ui.py
"""
from __future__ import annotations

import os
import sys
import json
import time
import types
import queue
import subprocess
import importlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import streamlit as st

# ---------- Page config ----------
st.set_page_config(page_title="EmailOps UI", page_icon="📬", layout="wide")

# ---------- Small utils ----------
def _normpath(p: str | Path) -> str:
    try:
        return str(Path(p).expanduser().resolve())
    except Exception:
        return str(p)

def _pkg_root_from(project_root: Path) -> Optional[Path]:
    """
    Return the path we must add to sys.path so `import emailops` works.
    If project_root itself contains an 'emailops' folder, use project_root.
    If project_root IS the 'emailops' folder, use its parent.
    Else None.
    """
    pr = project_root
    if (pr / "emailops").exists() and (pr / "emailops").is_dir():
        return pr
    if pr.name == "emailops":
        return pr.parent
    return None

@st.cache_resource(show_spinner=False)
def _import_modules(project_root: str) -> Dict[str, Any]:
    """
    Lazy import of your modules after injecting the correct path.
    Returns a dict of modules.
    """
    pr = Path(project_root)
    pkg_root = _pkg_root_from(pr)
    if not pkg_root:
        raise RuntimeError(
            f"Could not find 'emailops' package under: {project_root}. "
            f"Point 'Project root' to the directory that contains the 'emailops' folder."
        )
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    modules = {}
    # Import in the order that minimizes import churn
    modules["utils"] = importlib.import_module("emailops.utils")
    modules["llm_client"] = importlib.import_module("emailops.llm_client")
    modules["env_utils"] = importlib.import_module("emailops.env_utils")
    modules["email_indexer"] = importlib.import_module("emailops.email_indexer")
    modules["search_and_draft"] = importlib.import_module("emailops.search_and_draft")
    modules["summarize_email_thread"] = importlib.import_module("emailops.summarize_email_thread")
    modules["doctor"] = importlib.import_module("emailops.doctor")
    modules["vertex_utils"] = importlib.import_module("emailops.vertex_utils")
    try:
        modules["vertex_indexer"] = importlib.import_module("emailops.vertex_indexer")
    except Exception:
        modules["vertex_indexer"] = None  # optional

    return modules

def _emit_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False, indent=2)

def _run_cli(cmd: List[str], workdir: str | None = None):
    """
    Run a python -m command and stream output lines.
    """
    with st.status("Running…", expanded=True) as status:
        st.write(" ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=workdir or None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                env=os.environ.copy(),
            )
        except Exception as e:
            st.error(f"Failed to start process: {e}")
            return 1

        log_box = st.empty()
        lines: List[str] = []
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if not line and proc.poll() is not None:
                break
            if line:
                lines.append(line.rstrip("\n"))
                # Show only the last ~400 lines to keep UI responsive
                log_box.code("\n".join(lines[-400:]), language="bash")
        rc = proc.poll()
        if rc == 0:
            status.update(label="Done", state="complete")
        else:
            status.update(label=f"Exited with code {rc}", state="error")
        return rc

def _metric(label: str, value: Any, delta: Optional[str] = None):
    col = st.container()
    with col:
        st.metric(label, value, delta=delta if delta else None)

def _nice_table(rows: List[Dict[str, Any]], cols: List[str] | None = None):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if cols:
        missing = [c for c in cols if c not in df.columns]
        for m in missing:
            df[m] = ""
        df = df[cols]
    st.dataframe(df, use_container_width=True, hide_index=True)

# ---------- Sidebar: environment & paths ----------
with st.sidebar:
    st.header("Environment")
    default_project_root = Path.cwd()
    project_root = Path(
        st.text_input("Project root (the folder that contains 'emailops')", _normpath(default_project_root))
    )
    export_root = Path(
        st.text_input("Export root (folder that contains conversation folders)", _normpath(default_project_root))
    )

    # Provider controls exposed; these are read by your modules
    provider = st.selectbox(
        "Embedding provider (for SEARCH)",
        ["vertex", "openai", "azure", "cohere", "huggingface", "local", "qwen"],
        index=0,
        help="Search uses this provider's embeddings. Draft generation uses your Vertex config by default."
    )

    st.caption("Optional environment overrides (saved to this process only):")
    gcp_project = st.text_input("GCP_PROJECT", os.environ.get("GCP_PROJECT", ""))
    gcp_region = st.text_input("GCP_REGION", os.environ.get("GCP_REGION", "global"))
    creds = st.text_input("GOOGLE_APPLICATION_CREDENTIALS", os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""))
    apply_env = st.button("Apply environment overrides")
    if apply_env:
        if gcp_project:
            os.environ["GCP_PROJECT"] = gcp_project
            os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_project
        os.environ["GCP_REGION"] = gcp_region or "global"
        if creds:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
        st.success("Environment updated for this UI session.")

    # Load modules (injects sys.path once and caches)
    load_btn = st.button("Load/Reload modules")
    if load_btn:
        _import_modules(str(project_root))
        st.success("Modules loaded.")

st.title("📬 EmailOps — Easy UI")

# Auto‑import on first render (safe via cache)
try:
    modules = _import_modules(str(project_root))
except Exception as e:
    st.error(str(e))
    st.stop()

# ---------- Tabs ----------
tab_status, tab_index, tab_search, tab_summarize, tab_doctor = st.tabs(
    ["Status", "Index", "Search & Draft", "Summarize Thread", "Doctor"]
)

# ---------- STATUS ----------
with tab_status:
    st.subheader("Index status")
    try:
        ixmon_cls = modules["vertex_utils"].IndexingMonitor  # class
        mon = ixmon_cls(str(export_root))
        status = mon.check_status()
        # Render key bits
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Documents indexed", f"{status.documents_indexed:,}")
        with col2:
            st.metric("Conversations indexed", f"{status.conversations_indexed:,}")
        with col3:
            pct = f"{status.progress_percent:.1f}%"
            st.metric("Progress", pct)
        with col4:
            st.metric("Index exists", "Yes" if status.index_exists else "No")
        meta = {k: v for k, v in status.to_dict().items() if k in ("provider","model","actual_dimensions","index_type","last_updated","index_file","index_file_size_mb")}
        st.caption("Metadata")
        st.json(meta)
    except Exception as e:
        st.warning(f"Status unavailable: {e}")

# ---------- INDEX ----------
with tab_index:
    st.subheader("Build / Update index")
    st.caption("Runs the indexer module exactly like your CLI, with logs streamed here.")
    colA, colB, colC = st.columns(3)
    with colA:
        force = st.checkbox("Force full reindex (--force-reindex)", value=False)
    with colB:
        limit = st.number_input("Limit conversations (0 = no limit)", min_value=0, step=1, value=0)
    with colC:
        batch = st.number_input("Embed batch size (EMBED_BATCH)", min_value=1, step=1, value=128)

    run_idx = st.button("Run indexer")
    if run_idx:
        env = os.environ.copy()
        env["EMBED_PROVIDER"] = provider
        env["EMBED_BATCH"] = str(int(batch))
        cmd = [
            sys.executable, "-m", "emailops.email_indexer",
            "--root", str(export_root),
            "--provider", provider,
            "--batch", str(int(batch)),
        ]
        if force:
            cmd.append("--force-reindex")
        if int(limit) > 0:
            cmd.extend(["--limit", str(int(limit))])
        # Stream logs
        _run_cli(cmd, workdir=str(project_root))

# ---------- SEARCH & DRAFT ----------
with tab_search:
    st.subheader("Retrieve context and draft a reply")

    q = st.text_input("Query", "")
    k = st.slider("Top‑K snippets", min_value=1, max_value=200, value=60, step=1)
    sender = st.text_input("Sender name/email for the draft signature", "")
    include_attachments = st.checkbox("Try to select relevant attachments", value=True)
    temperature = st.slider("Draft temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    btn_search = st.button("Search only")
    btn_draft = st.button("Search & Draft")

    ctx_results: List[Dict[str, Any]] = []
    if btn_search or btn_draft:
        try:
            sad = modules["search_and_draft"]
            ix_dir = Path(export_root) / os.getenv("INDEX_DIRNAME", "_index")
            ctx_results = sad._search(ix_dir, q, k=k, provider=provider)
            if not ctx_results:
                st.warning("No results. Consider reindexing, refining the query, or increasing K.")
            else:
                st.success(f"Found {len(ctx_results)} snippets")
                # Show a compact table
                rows = [{
                    "id": c.get("id",""),
                    "subject": c.get("subject",""),
                    "date": c.get("date",""),
                    "score": round(float(c.get("score", c.get("original_score", 0.0))), 3),
                    "type": c.get("doc_type",""),
                    "snippet": c.get("text","")
                } for c in ctx_results]
                _nice_table(rows, cols=["id","subject","date","score","type","snippet"])
        except Exception as e:
            st.error(f"Search failed: {e}")

    if btn_draft and ctx_results and sender.strip():
        try:
            with st.spinner("Drafting with LLM‑as‑critic…"):
                sad = modules["search_and_draft"]
                result = sad.draft_email_structured(
                    query=q,
                    sender=sender,
                    context_snippets=ctx_results,
                    provider=provider,
                    temperature=float(temperature),
                    include_attachments=bool(include_attachments)
                )
            # Render draft
            conf = result.get("confidence_score", 0.0)
            meta = result.get("metadata", {})
            st.markdown(f"### Draft (confidence {conf:.2f})")
            st.code(result["final_draft"]["email_draft"] or "", language="markdown")

            cols = st.columns(3)
            with cols[0]:
                st.metric("Citations", int(meta.get("citation_count", 0)))
            with cols[1]:
                st.metric("Word count", int(meta.get("draft_word_count", 0)))
            with cols[2]:
                st.metric("Workflow", str(meta.get("workflow_state","")))

            if result.get("final_draft", {}).get("missing_information"):
                st.markdown("**Missing information**")
                st.write(result["final_draft"]["missing_information"])

            if result.get("selected_attachments"):
                st.markdown("**Selected attachments**")
                _nice_table(result["selected_attachments"], cols=["filename","size_mb","relevance_score","extension"])

            issues = result.get("critic_feedback", {}).get("issues_found", [])
            if issues:
                st.markdown("**Critic issues**")
                _nice_table(issues, cols=["issue_type","severity","description"])

            st.markdown("**Citations**")
            _nice_table(result["final_draft"].get("citations", []), cols=["document_id","fact_cited","confidence"])

            # Save JSON if desired
            if st.button("Download JSON result"):
                st.download_button(
                    label="Download",
                    file_name="draft_result.json",
                    mime="application/json",
                    data=_emit_json(result).encode("utf-8")
                )
        except Exception as e:
            st.error(f"Drafting failed: {e}")

# ---------- SUMMARIZE THREAD ----------
with tab_summarize:
    st.subheader("Analyze a single thread (facts ledger)")
    thread_dir = st.text_input(
        "Thread directory (must contain Conversation.txt)",
        _normpath(export_root)
    )
    write_md = st.checkbox("Also write summary.md beside summary.json", value=False)
    run_sum = st.button("Analyze thread")

    if run_sum:
        try:
            tdir = Path(thread_dir)
            convo = tdir / "Conversation.txt"
            if not convo.exists():
                st.error(f"Conversation.txt not found in {tdir}")
            else:
                smod = modules["summarize_email_thread"]
                # Returns the dict without writing; we handle optional writes below
                # We re‑use the analyzer function directly to avoid argparse
                from emailops.utils import read_text_file, clean_email_text  # safe import after cache
                raw = read_text_file(convo)
                cleaned = clean_email_text(raw)
                data = smod.analyze_email_thread_with_ledger(thread_text=cleaned)
                st.success("Analysis complete")
                st.json(data)
                # Persist files if asked
                (tdir / "summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                if write_md:
                    # Simple, human readable MD
                    md_lines = []
                    md_lines.append(f"# Email Thread Analysis")
                    md_lines.append("")
                    md_lines.append(f"**Category**: {data.get('category','unknown')}  ")
                    md_lines.append(f"**Subject**: {data.get('subject','')}  ")
                    md_lines.append("")
                    md_lines.append("## Summary")
                    for s in data.get("summary", []):
                        md_lines.append(f"- {s}")
                    md_lines.append("")
                    md_lines.append("## Next Actions")
                    for a in data.get("next_actions", []):
                        md_lines.append(f"- **{a.get('owner','')}**: {a.get('action','')} ({a.get('priority','')}, {a.get('status','')})")
                    (tdir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")
                st.info(f"Wrote files to: {tdir}")
        except Exception as e:
            st.error(f"Summarization failed: {e}")

# ---------- DOCTOR ----------
with tab_doctor:
    st.subheader("Doctor diagnostics")
    col1, col2, col3 = st.columns(3)
    with col1:
        skip_install = st.checkbox("Skip dependency checks (--skip-install-check)", value=False)
    with col2:
        auto_install = st.checkbox("Auto‑install missing (--install-missing)", value=False)
    with col3:
        skip_embed = st.checkbox("Skip embed connectivity (--skip-embed-check)", value=False)

    log_level = st.selectbox("Log level", ["DEBUG","INFO","WARNING","ERROR"], index=1)
    run_doctor = st.button("Run Doctor")

    if run_doctor:
        cmd = [
            sys.executable, "-m", "emailops.doctor",
            "--root", str(export_root),
            "--log-level", log_level
        ]
        if skip_install:
            cmd.append("--skip-install-check")
        if auto_install:
            cmd.append("--install-missing")
        if skip_embed:
            cmd.append("--skip-embed-check")
        _run_cli(cmd, workdir=str(project_root))

st.caption("© EmailOps UI — Streamlit frontend")
