# EmailOps GUI - Production Enhancement Implementation Plan

## Executive Summary

This document provides a complete implementation plan to elevate [`emailops_gui.py`](emailops_gui.py:1) to best-in-class production-ready state with all requested features.

## Current State Analysis

### Existing Features (Well Implemented)
✓ Search with advanced filters
✓ Draft reply with parameter control
✓ Draft fresh emails
✓ Chat with session management
✓ Conversation browsing
✓ Index building
✓ Configuration management
✓ System diagnostics
✓ Text chunking
✓ Thread analysis

### Missing Critical Features
❌ Batch email summarization
❌ Batch draft/reply operations
❌ Real-time multiprocessing progress visualization
❌ Conversation.txt file viewer
❌ Attachment browser and opener
❌ Enhanced progress tracking
❌ Result export capabilities
❌ File pointer navigation

## Implementation Roadmap

### Phase 1: Enhanced Conversation Tab with File Viewing

#### Add to [`_build_conversations_tab()`](emailops_gui.py:640)

```python
def _build_conversations_tab(self) -> None:
    """Enhanced conversations tab with full viewing capabilities."""
    frm = self.tab_convs

    # Split into two panes: list (left) and viewer (right)
    paned = ttk.PanedWindow(frm, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # Left pane: Conversation list
    left_frame = ttk.Frame(paned)
    paned.add(left_frame, weight=2)

    # Top control panel
    top = ttk.Frame(left_frame, padding=8)
    top.pack(fill=tk.X)
    ttk.Button(top, text="📋 List Conversations", command=self._on_list_convs).pack(side=tk.LEFT)
    self.pb_convs = ttk.Progressbar(top, mode="indeterminate", length=180)
    self.pb_convs.pack(side=tk.LEFT, padx=8)
    ttk.Button(top, text="🔄 Refresh", command=self._on_list_convs).pack(side=tk.LEFT, padx=4)

    # Tree view
    cols = ("conv_id", "subject", "first", "last", "count")
    self.tree_convs = ttk.Treeview(left_frame, columns=cols, show="headings", height=20)
    for k, w in (("conv_id", 220), ("subject", 400), ("first", 120), ("last", 120), ("count", 60)):
        self.tree_convs.heading(k, text=k.replace("_", " ").title())
        self.tree_convs.column(k, width=w)
    self.tree_convs.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
    self.tree_convs.bind("<<TreeviewSelect>>", self._on_conv_selected)
    self.tree_convs.bind("<Double-1>", self._view_conversation_content)

    # Action buttons
    btn_frame = ttk.Frame(left_frame, padding=8)
    btn_frame.pack(fill=tk.X)
    ttk.Button(btn_frame, text="👁️ View Content", command=self._view_conversation_content).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_frame, text="📎 View Attachments", command=self._view_attachments).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_frame, text="↩️ Use in Reply", command=self._use_selected_conv).pack(side=tk.LEFT, padx=2)

    # Right pane: Content viewer
    right_frame = ttk.Frame(paned)
    paned.add(right_frame, weight=3)

    # Viewer toolbar
    viewer_toolbar = ttk.Frame(right_frame, padding=8)
    viewer_toolbar.pack(fill=tk.X)
    ttk.Label(viewer_toolbar, text="Conversation Viewer", font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
    ttk.Button(viewer_toolbar, text="📁 Open Folder", command=self._open_conversation_folder).pack(side=tk.RIGHT, padx=2)
    ttk.Button(viewer_toolbar, text="📄 Open in Editor", command=self._open_in_editor).pack(side=tk.RIGHT, padx=2)

    # Notebook for different views
    self.conv_viewer_nb = ttk.Notebook(right_frame)
    self.conv_viewer_nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    # Tab 1: Conversation text
    conv_text_frame = ttk.Frame(self.conv_viewer_nb)
    self.conv_viewer_nb.add(conv_text_frame, text="📝 Conversation.txt")
    self.txt_conv_viewer = tk.Text(conv_text_frame, wrap="word", font=("Courier New", 9))
    self.txt_conv_viewer.pack(fill=tk.BOTH, expand=True)
    conv_scroll = ttk.Scrollbar(conv_text_frame, orient=tk.VERTICAL, command=self.txt_conv_viewer.yview)
    conv_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    self.txt_conv_viewer.configure(yscrollcommand=conv_scroll.set)

    # Tab 2: Attachments list
    att_frame = ttk.Frame(self.conv_viewer_nb)
    self.conv_viewer_nb.add(att_frame, text="📎 Attachments")
    
    att_cols = ("filename", "type", "size")
    self.tree_attachments = ttk.Treeview(att_frame, columns=att_cols, show="headings", height=15)
    self.tree_attachments.heading("filename", text="Filename")
    self.tree_attachments.heading("type", text="Type")
    self.tree_attachments.heading("size", text="Size")
    self.tree_attachments.column("filename", width=400)
    self.tree_attachments.column("type", width=100)
    self.tree_attachments.column("size", width=100)
    self.tree_attachments.pack(fill=tk.BOTH, expand=True)
    self.tree_attachments.bind("<Double-1>", self._open_attachment)

    att_scroll = ttk.Scrollbar(att_frame, orient=tk.VERTICAL, command=self.tree_attachments.yview)
    att_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    self.tree_attachments.configure(yscrollcommand=att_scroll.set)

    att_btn_frame = ttk.Frame(att_frame, padding=4)
    att_btn_frame.pack(fill=tk.X)
    ttk.Button(att_btn_frame, text="📂 Open Selected", command=self._open_attachment).pack(side=tk.LEFT, padx=2)
    ttk.Button(att_btn_frame, text="📁 Open Attachments Folder", command=self._open_attachments_folder).pack(side=tk.LEFT, padx=2)

    # Tab 3: Manifest JSON
    manifest_frame = ttk.Frame(self.conv_viewer_nb)
    self.conv_viewer_nb.add(manifest_frame, text="📋 Manifest")
    self.txt_manifest = tk.Text(manifest_frame, wrap="word", font=("Courier New", 9))
    self.txt_manifest.pack(fill=tk.BOTH, expand=True)
    manifest_scroll = ttk.Scrollbar(manifest_frame, orient=tk.VERTICAL, command=self.txt_manifest.yview)
    manifest_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    self.txt_manifest.configure(yscrollcommand=manifest_scroll.set)

    # Store current conversation path
    self.current_conv_path: Path | None = None
    self.current_attachments: list[Path] = []
```

#### Add Supporting Methods

```python
def _on_conv_selected(self, event) -> None:
    """Handle conversation selection."""
    selection = self.tree_convs.selection()
    if not selection:
        return
    
    try:
        values = self.tree_convs.item(selection[0])["values"]
        conv_id = values[0]
        
        # Load conversation details in background
        threading.Thread(target=self._load_conversation_details, args=(conv_id,), daemon=True).start()
        
    except Exception as e:
        module_logger.error(f"Failed to handle conversation selection: {e}")

def _load_conversation_details(self, conv_id: str) -> None:
    """Load and display conversation details."""
    try:
        conv_path = Path(self.settings.export_root) / conv_id
        if not conv_path.exists():
            return
        
        self.current_conv_path = conv_path
        
        # Load Conversation.txt
        conv_file = conv_path / "Conversation.txt"
        if conv_file.exists():
            text = conv_file.read_text(encoding="utf-8", errors="ignore")
            self.txt_conv_viewer.delete("1.0", tk.END)
            self.txt_conv_viewer.insert("1.0", text)
        
        # Load attachments
        self._load_attachments_list(conv_path)
        
        # Load manifest
        manifest_file = conv_path / "manifest.json"
        if manifest_file.exists():
            manifest_text = manifest_file.read_text(encoding="utf-8", errors="ignore")
            self.txt_manifest.delete("1.0", tk.END)
            try:
                # Pretty print JSON
                manifest_data = json.loads(manifest_text)
                pretty_json = json.dumps(manifest_data, indent=2, ensure_ascii=False)
                self.txt_manifest.insert("1.0", pretty_json)
            except:
                self.txt_manifest.insert("1.0", manifest_text)
        
    except Exception as e:
        module_logger.error(f"Failed to load conversation details: {e}")

def _load_attachments_list(self, conv_path: Path) -> None:
    """Load attachments list for the conversation."""
    try:
        self.tree_attachments.delete(*self.tree_attachments.get_children())
        self.current_attachments = []
        
        attachments_dir = conv_path / "Attachments"
        if attachments_dir.exists():
            for att_file in sorted(attachments_dir.rglob("*")):
                if att_file.is_file():
                    size_mb = att_file.stat().st_size / (1024 * 1024)
                    self.tree_attachments.insert("", "end", values=(
                        att_file.name,
                        att_file.suffix.lstrip(".").upper() or "FILE",
                        f"{size_mb:.2f} MB"
                    ))
                    self.current_attachments.append(att_file)
        
        # Also check for files in root
        for child in conv_path.iterdir():
            if child.is_file() and child.name not in {"Conversation.txt", "manifest.json", "summary.json", "summary.md"}:
                size_mb = child.stat().st_size / (1024 * 1024)
                self.tree_attachments.insert("", "end", values=(
                    child.name,
                    child.suffix.lstrip(".").upper() or "FILE",
                    f"{size_mb:.2f} MB"
                ))
                self.current_attachments.append(child)
        
    except Exception as e:
        module_logger.error(f"Failed to load attachments: {e}")

def _view_conversation_content(self, event=None) -> None:
    """View full conversation content."""
    selection = self.tree_convs.selection()
    if not selection:
        messagebox.showwarning("No Selection", "Please select a conversation")
        return
    
    # Switch to conversation viewer tab
    self.conv_viewer_nb.select(0)

def _view_attachments(self) -> None:
    """View attachments tab."""
    if not self.current_conv_path:
        messagebox.showwarning("No Conversation", "Select a conversation first")
        return
    
    # Switch to attachments tab
    self.conv_viewer_nb.select(1)

def _open_attachment(self, event=None) -> None:
    """Open selected attachment with default application."""
    selection = self.tree_attachments.selection()
    if not selection:
        messagebox.showwarning("No Selection", "Please select an attachment")
        return
    
    try:
        idx = self.tree_attachments.index(selection[0])
        if 0 <= idx < len(self.current_attachments):
            att_path = self.current_attachments[idx]
            
            # Open with default application (platform-specific)
            import platform
            import subprocess
            
            if platform.system() == 'Windows':
                os.startfile(str(att_path))
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', str(att_path)])
            else:  # Linux
                subprocess.run(['xdg-open', str(att_path)])
            
            module_logger.info(f"Opened attachment: {att_path.name}")
    except Exception as e:
        module_logger.error(f"Failed to open attachment: {e}")
        messagebox.showerror("Error", f"Failed to open attachment:\n{e!s}")

def _open_conversation_folder(self) -> None:
    """Open conversation folder in file explorer."""
    if not self.current_conv_path or not self.current_conv_path.exists():
        messagebox.showwarning("No Conversation", "Select a conversation first")
        return
    
    try:
        import platform
        import subprocess
        
        if platform.system() == 'Windows':
            os.startfile(str(self.current_conv_path))
        elif platform.system() == 'Darwin':
            subprocess.run(['open', str(self.current_conv_path)])
        else:
            subprocess.run(['xdg-open', str(self.current_conv_path)])
        
    except Exception as e:
        module_logger.error(f"Failed to open folder: {e}")
        messagebox.showerror("Error", f"Failed to open folder:\n{e!s}")

def _open_attachments_folder(self) -> None:
    """Open Attachments subfolder."""
    if not self.current_conv_path:
        return
    
    att_dir = self.current_conv_path / "Attachments"
    if att_dir.exists():
        self._open_folder(att_dir)
    else:
        messagebox.showinfo("No Attachments", "No Attachments folder found")

def _open_in_editor(self) -> None:
    """Open Conversation.txt in default text editor."""
    if not self.current_conv_path:
        return
    
    conv_file = self.current_conv_path / "Conversation.txt"
    if conv_file.exists():
        try:
            import platform
            import subprocess
            
            if platform.system() == 'Windows':
                os.startfile(str(conv_file))
            elif platform.system() == 'Darwin':
                subprocess.run(['open', '-t', str(conv_file)])
            else:
                subprocess.run(['xdg-open', str(conv_file)])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open editor:\n{e!s}")
```

### Phase 2: Batch Operations Tab

#### Add New Tab [`_build_batch_tab()`](emailops_gui.py:396)

```python
def _build_batch_tab(self) -> None:
    """Build batch operations tab for bulk processing."""
    frm = ttk.Frame(self.nb)
    self.tab_batch = frm
    self.nb.insert(5, frm, text="⚡ Batch Operations")  # Insert after Conversations tab

    # Section 1: Batch Summarization
    summ_frame = ttk.LabelFrame(frm, text="Batch Email Summarization", padding=10)
    summ_frame.pack(fill=tk.X, padx=8, pady=8)

    s1 = ttk.Frame(summ_frame)
    s1.pack(fill=tk.X, pady=4)
    ttk.Label(s1, text="Select conversations from the Conversations tab, then:").pack(side=tk.LEFT)

    s2 = ttk.Frame(summ_frame)
    s2.pack(fill=tk.X, pady=4)
    ttk.Label(s2, text="Output Format:").pack(side=tk.LEFT)
    self.var_batch_summ_format = tk.StringVar(value="both")
    ttk.Radiobutton(s2, text="JSON", variable=self.var_batch_summ_format, value="json").pack(side=tk.LEFT, padx=4)
    ttk.Radiobutton(s2, text="Markdown", variable=self.var_batch_summ_format, value="markdown").pack(side=tk.LEFT, padx=4)
    ttk.Radiobutton(s2, text="Both", variable=self.var_batch_summ_format, value="both").pack(side=tk.LEFT, padx=4)

    self.var_batch_export_csv = tk.BooleanVar(value=True)
    ttk.Checkbutton(s2, text="Export Actions to CSV", variable=self.var_batch_export_csv).pack(side=tk.LEFT, padx=20)

    s3 = ttk.Frame(summ_frame)
    s3.pack(fill=tk.X, pady=4)
    ttk.Label(s3, text="Parallel Workers:").pack(side=tk.LEFT)
    self.var_batch_workers = tk.IntVar(value=4)
    ttk.Spinbox(s3, from_=1, to=8, width=6, textvariable=self.var_batch_workers).pack(side=tk.LEFT, padx=4)
    ttk.Label(s3, text="(more = faster, but more resources)").pack(side=tk.LEFT)

    s4 = ttk.Frame(summ_frame)
    s4.pack(fill=tk.X, pady=4)
    self.btn_batch_summarize = ttk.Button(s4, text="⚡ Summarize All Conversations", 
                                          command=self._on_batch_summarize, style='Action.TButton')
    self.btn_batch_summarize.pack(side=tk.LEFT)
    self.pb_batch_summ = ttk.Progressbar(s4, mode="determinate", length=300)
    self.pb_batch_summ.pack(side=tk.LEFT, padx=8)
    self.lbl_batch_summ_status = ttk.Label(s4, text="")
    self.lbl_batch_summ_status.pack(side=tk.LEFT, padx=8)

    # Section 2: Batch Reply Generation
    reply_frame = ttk.LabelFrame(frm, text="Batch Reply Generation", padding=10)
    reply_frame.pack(fill=tk.X, padx=8, pady=8)

    r1 = ttk.Frame(reply_frame)
    r1.pack(fill=tk.X, pady=4)
    ttk.Label(r1, text="Generate replies for multiple conversations:").pack(side=tk.LEFT)

    r2 = ttk.Frame(reply_frame)
    r2.pack(fill=tk.X, pady=4)
    ttk.Label(r2, text="Reply Policy:").pack(side=tk.LEFT)
    self.var_batch_reply_policy = tk.StringVar(value="smart")
    ttk.Combobox(r2, width=15, state="readonly", textvariable=self.var_batch_reply_policy,
                 values=["reply_all", "smart", "sender_only"]).pack(side=tk.LEFT, padx=4)
    
    ttk.Label(r2, text="Max Tokens:").pack(side=tk.LEFT, padx=(20, 4))
    self.var_batch_reply_tokens = tk.IntVar(value=20000)
    ttk.Spinbox(r2, from_=5000, to=100000, increment=5000, width=10, 
                textvariable=self.var_batch_reply_tokens).pack(side=tk.LEFT, padx=4)

    self.var_batch_include_attachments = tk.BooleanVar(value=True)
    ttk.Checkbutton(r2, text="Include Attachments", variable=self.var_batch_include_attachments).pack(side=tk.LEFT, padx=20)

    r3 = ttk.Frame(reply_frame)
    r3.pack(fill=tk.X, pady=4)
    self.btn_batch_replies = ttk.Button(r3, text="⚡ Generate Batch Replies", 
                                        command=self._on_batch_replies, style='Action.TButton')
    self.btn_batch_replies.pack(side=tk.LEFT)
    self.pb_batch_reply = ttk.Progressbar(r3, mode="determinate", length=300)
    self.pb_batch_reply.pack(side=tk.LEFT, padx=8)
    self.lbl_batch_reply_status = ttk.Label(r3, text="")
    self.lbl_batch_reply_status.pack(side=tk.LEFT, padx=8)

    # Section 3: Progress Monitor
    prog_frame = ttk.LabelFrame(frm, text="Batch Operation Monitor", padding=10)
    prog_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    self.txt_batch_log = tk.Text(prog_frame, wrap="word", font=("Courier New", 9), height=15)
    self.txt_batch_log.pack(fill=tk.BOTH, expand=True)
    
    batch_scroll = ttk.Scrollbar(prog_frame, orient=tk.VERTICAL, command=self.txt_batch_log.yview)
    batch_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    self.txt_batch_log.configure(yscrollcommand=batch_scroll.set)
```

#### Batch Summarization Implementation

```python
def _on_batch_summarize(self) -> None:
    """Batch summarize all conversations."""
    if not self.settings.export_root:
        messagebox.showwarning("No Root", "Please select export root first")
        return
    
    # Confirm operation
    result = messagebox.askyesno(
        "Batch Summarize",
        "This will summarize ALL conversations in the export root.\n\n"
        "This may take significant time and API quota.\n\n"
        "Continue?"
    )
    
    if not result:
        return

    def run_batch():
        try:
            self.btn_batch_summarize.config(state="disabled")
            self.pb_batch_summ.config(mode="determinate", value=0)
            self.txt_batch_log.delete("1.0", tk.END)
            self.txt_batch_log.insert("1.0", "Starting batch summarization...\n\n")

            root = Path(self.settings.export_root)
            
            # Find all conversation directories
            conv_dirs = [
                d for d in root.iterdir() 
                if d.is_dir() and (d / "Conversation.txt").exists()
            ]
            
            total = len(conv_dirs)
            if total == 0:
                self.txt_batch_log.insert(tk.END, "No conversations found.\n")
                return
            
            self.txt_batch_log.insert(tk.END, f"Found {total} conversations to summarize.\n\n")
            self.lbl_batch_summ_status.config(text=f"0/{total}")

            # Use processor's batch summarization with multiprocessing
            import multiprocessing as mp
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class SummarizeJob:
                conv_dir: Path
                provider: str
                output_format: str
                export_csv: bool

            def summarize_worker(job: SummarizeJob) -> tuple[str, bool, str]:
                """Worker function for batch summarization."""
                import asyncio
                
                try:
                    # Run analysis
                    analysis = asyncio.run(
                        summarizer.analyze_conversation_dir(
                            thread_dir=job.conv_dir,
                            provider=job.provider,
                            temperature=0.2,
                            merge_manifest=True
                        )
                    )
                    
                    # Save results
                    if job.output_format in ["json", "both"]:
                        json_path = job.conv_dir / "summary.json"
                        json_path.write_text(
                            json.dumps(analysis, indent=2, ensure_ascii=False),
                            encoding="utf-8"
                        )
                    
                    if job.output_format in ["markdown", "both"]:
                        md_content = summarizer.format_analysis_as_markdown(analysis)
                        md_path = job.conv_dir / "summary.md"
                        md_path.write_text(md_content, encoding="utf-8")
                    
                    if job.export_csv and analysis.get("next_actions"):
                        from emailops.summarize_email_thread import _append_todos_csv
                        _append_todos_csv(job.conv_dir.parent, job.conv_dir.name, 
                                        analysis.get("next_actions", []))
                    
                    return (str(job.conv_dir.name), True, "Success")
                    
                except Exception as e:
                    return (str(job.conv_dir.name), False, str(e))

            # Create jobs
            jobs = [
                SummarizeJob(
                    conv_dir=d,
                    provider=self.settings.provider,
                    output_format=self.var_batch_summ_format.get(),
                    export_csv=self.var_batch_export_csv.get()
                )
                for d in conv_dirs
            ]

            # Process with multiprocessing
            ctx = mp.get_context("spawn")
            workers = min(self.var_batch_workers.get(), total)
            
            success_count = 0
            failed = []

            with ctx.Pool(processes=workers) as pool:
                for i, (conv_name, success, msg) in enumerate(pool.imap_unordered(summarize_worker, jobs), 1):
                    if success:
                        success_count += 1
                        self.txt_batch_log.insert(tk.END, f"✓ [{i}/{total}] {conv_name}\n")
                        self.txt_batch_log.tag_add("success", f"end-2l", f"end-1l")
                    else:
                        failed.append((conv_name, msg))
                        self.txt_batch_log.insert(tk.END, f"✗ [{i}/{total}] {conv_name}: {msg}\n")
                        self.txt_batch_log.tag_add("error", f"end-2l", f"end-1l")
                    
                    self.txt_batch_log.see(tk.END)
                    
                    # Update progress bar
                    progress = (i / total) * 100
                    self.pb_batch_summ.config(value=progress)
                    self.lbl_batch_summ_status.config(text=f"{i}/{total}")
                    self.update_idletasks()

            # Final summary
            self.txt_batch_log.insert(tk.END, f"\n{'='*60}\n")
            self.txt_batch_log.insert(tk.END, f"Batch summarization complete!\n")
            self.txt_batch_log.insert(tk.END, f"Success: {success_count}/{total}\n")
            
            if failed:
                self.txt_batch_log.insert(tk.END, f"Failed: {len(failed)}\n")

            messagebox.showinfo("Complete", 
                              f"Batch summarization complete.\n\n"
                              f"Success: {success_count}/{total}\n"
                              f"Failed: {len(failed)}")

        except Exception as e:
            module_logger.error(f"Batch summarization failed: {e}", exc_info=True)
            self.txt_batch_log.insert(tk.END, f"\n✗ Error: {e!s}\n")
            messagebox.showerror("Error", f"Batch summarization failed:\n{e!s}")
        
        finally:
            self.pb_batch_summ.config(value=0)
            self.btn_batch_summarize.config(state="normal")

    threading.Thread(target=run_batch, daemon=True).start()
```

#### Batch Reply Generation

```python
def _on_batch_replies(self) -> None:
    """Generate replies for selected conversations."""
    # Get selected conversations from tree
    selections = self.tree_convs.selection()
    if not selections:
        messagebox.showwarning("No Selection", 
                             "Please select conversations from the Conversations tab first")
        return
    
    conv_ids = []
    for sel in selections:
        values = self.tree_convs.item(sel)["values"]
        conv_ids.append(values[0])
    
    result = messagebox.askyesno(
        "Batch Generate Replies",
        f"Generate replies for {len(conv_ids)} selected conversations?\n\n"
        "This will create .eml files in the export root."
    )
    
    if not result:
        return

    def run_batch():
        try:
            self.btn_batch_replies.config(state="disabled")
            self.pb_batch_reply.config(mode="determinate", value=0)
            self.txt_batch_log.delete("1.0", tk.END)
            self.txt_batch_log.insert("1.0", f"Generating {len(conv_ids)} replies...\n\n")

            from emailops.search_and_draft import draft_email_reply_eml

            success_count = 0
            failed = []

            for i, conv_id in enumerate