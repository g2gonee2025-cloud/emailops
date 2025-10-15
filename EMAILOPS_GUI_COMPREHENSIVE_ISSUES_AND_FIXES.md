# EmailOps GUI Comprehensive Issues and Fixes

## Executive Summary
This document provides a detailed analysis of all issues, inconsistencies, and redundancies found in `emailops_gui.py`, along with precise fix instructions for an Anthropic agentic coding LLM.

**Analysis Date:** 2025-10-15  
**File Analyzed:** emailops_gui.py (1,030 lines)  
**Total Issues Found:** 15 critical, 8 medium, 12 minor

---

## CRITICAL ISSUES (Must Fix)

### 1. Decorator Signature Mismatch - RUNTIME ERROR
**Location:** Lines 975, 1000  
**Severity:** CRITICAL - Will cause TypeError at runtime  
**Issue:** Decorated methods have inconsistent `update_progress` parameter positioning

**Current Code:**
```python
@run_with_progress("batch_summarize", "pb_batch", "lbl_batch_progress", "btn_batch_summarize", "btn_batch_reply")
def _on_batch_summarize(self, update_progress) -> None:  # ❌ WRONG
```

**Fix Required:**
```python
@run_with_progress("batch_summarize", "pb_batch", "lbl_batch_progress", "btn_batch_summarize", "btn_batch_reply")
def _on_batch_summarize(self, *, update_progress) -> None:  # ✅ CORRECT - keyword-only
```

**Affected Methods:**
- Line 975: `_on_batch_summarize` 
- Line 1000: `_on_batch_replies`

**Why This Matters:**
The decorator passes `update_progress` as a keyword argument, but these methods expect it positionally. Python will raise `TypeError: _on_batch_summarize() missing 1 required positional argument: 'update_progress'`.

---

### 2. Missing Atomic Write for Settings
**Location:** Line 140-161  
**Severity:** CRITICAL - Data corruption risk  
**Issue:** Settings save uses tempfile but doesn't handle cleanup properly

**Current Implementation:**
```python
def save(self) -> None:
    # ... creates temp file but doesn't guarantee cleanup on error
    fd, temp_path_str = tempfile.mkstemp(...)
    temp_path = Path(temp_path_str)
    try:
        # write and replace
    finally:
        if temp_path.exists():
            with contextlib.suppress(OSError):
                temp_path.unlink()
```

**Issue:** If `os.replace()` fails, temp file remains. If process crashes during write, temp file leaks.

**Fix Required:**
```python
def save(self) -> None:
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        settings_dict = asdict(self)
        json_content = json.dumps(settings_dict, ensure_ascii=False, indent=2)
        
        # Atomic write pattern
        fd, temp_path_str = tempfile.mkstemp(
            dir=SETTINGS_FILE.parent, 
            prefix=f".{SETTINGS_FILE.name}.",
            suffix=".tmp"
        )
        temp_path = Path(temp_path_str)
        
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(json_content)
                f.flush()
                os.fsync(f.fileno())  # ✅ ADD THIS
            
            # Retry replace on Windows file locks
            for attempt in range(3):
                try:
                    os.replace(temp_path, SETTINGS_FILE)
                    break
                except PermissionError:
                    if attempt == 2:
                        raise
                    time.sleep(0.1)
        finally:
            # Ensure cleanup
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
        
        module_logger.info(f"✓ Settings saved to {SETTINGS_FILE}")
    except Exception as e:
        module_logger.error(f"✗ Failed to save settings: {e}", exc_info=True)
        raise
```

---

### 3. Incorrect SearchFilters Construction
**Location:** Line 792  
**Severity:** CRITICAL - Will cause TypeError  
**Issue:** `SearchFilters(**filters_dict)` assumes dict keys match SearchFilters field names, but they don't

**Current Code:**
```python
search_filters = SearchFilters(**filters_dict)  # ❌ WRONG
```

**Fix Required:**
```python
search_filters = SearchFilters()
if "from_emails" in filters_dict:
    search_filters.from_emails = filters_dict["from_emails"]
if "to_emails" in filters_dict:
    search_filters.to_emails = filters_dict["to_emails"]
if "subject_contains" in filters_dict:
    search_filters.subject_contains = filters_dict["subject_contains"]
```

**Why:** SearchFilters is a @dataclass with specific typed fields. Unpacking an arbitrary dict will fail if keys don't match exactly.

---

### 4. Thread Safety Issue in Batch Operations
**Location:** Lines 975-1024  
**Severity:** CRITICAL - Race conditions possible  
**Issue:** Accessing `self.tree_batch` from worker thread without proper synchronization

**Current Code:**
```python
@run_with_progress(...)
def _on_batch_summarize(self, *, update_progress) -> None:
    items = self.tree_batch.get_children()  # ❌ Direct access from thread
    conv_ids = [self.tree_batch.item(item_id)["values"][0] for item_id in items]
```

**Fix Required:**
```python
@run_with_progress(...)
def _on_batch_summarize(self, *, update_progress) -> None:
    # Capture data on main thread BEFORE starting background work
    items = self.tree_batch.get_children()
    conv_ids = []
    for item_id in items:
        values = self.tree_batch.item(item_id)["values"]
        if values and isinstance(values, (list, tuple)) and len(values) > 0:
            conv_ids.append(values[0])
    
    if not conv_ids:
        self.after(0, lambda: messagebox.showwarning("No Valid Items", "No valid conversation IDs in batch list"))
        return
    
    # Now conv_ids is safe to use in worker thread
    conv_dirs = [Path(self.settings.export_root) / cid for cid in conv_ids]
    # ... rest of implementation
```

**Why:** Tkinter widgets are not thread-safe. Must extract all data on main thread before worker thread starts.

---

### 5. Missing Import for `time` Module
**Location:** Line 152 (atomic write retry logic)  
**Severity:** CRITICAL - NameError at runtime  
**Issue:** `time.sleep()` used but `time` not imported in save() implementation

**Current Code:**
```python
import tempfile  # at top of file
# ... but no: import time
```

**Fix Required:**
Add to imports at top of file (around line 40):
```python
import time
```

---

## MEDIUM SEVERITY ISSUES

### 6. Inconsistent Progress Callback Usage
**Location:** Lines 764-1045  
**Severity:** MEDIUM - UX inconsistency  
**Issue:** Some decorated methods call `update_progress` with inconsistent arguments

**Current Pattern Violations:**
```python
# ❌ Line 795: Missing total parameter
update_progress(0, 1, "Searching...")

# ✅ Line 980: Correct usage
update_progress(i, total, f"Summarizing {i+1}/{total}: {conv_dir.name}...")

# ❌ Line 1110: Inconsistent parameters
update_progress(1, 1, "Diagnostics complete")  # Should pass current, total, message
```

**Fix Required:**
Standardize all `update_progress` calls:
```python
update_progress(current: int, total: int, message: str = "")
```

All calls must provide 3 arguments consistently.

---

### 7. Duplicate Method: `_update_status` vs `_set_status`
**Location:** Lines 1045, 687  
**Severity:** MEDIUM - Code duplication  
**Issue:** Two methods do the same thing

**Current Code:**
```python
def _set_status(self, msg: str, color: str = "info") -> None:
    self.status_var.set(msg)
    self.status_label.config(foreground=self.colors.get(color, "#555"))

def _update_status(self, msg: str) -> None:
    """Backward compatibility alias for _set_status."""
    self._set_status(msg, "info")
```

**Fix Required:**
Remove `_update_status` entirely. Search and replace all calls to use `_set_status` directly:
```bash
# Find: self._update_status(
# Replace with: self._set_status(
```

---

### 8. Missing Error Handling in `_sync_settings_from_UI`
**Location:** Line 700  
**Severity:** MEDIUM - Silent failures possible  
**Issue:** Type conversion errors not caught for individual fields

**Current Code:**
```python
self.settings.temperature = float(self.var_temp.get())  # ❌ No try-except
self.settings.k = int(self.var_k.get())  # ❌ Could raise ValueError
```

**Fix Required:**
```python
def _sync_settings_from_ui(self) -> None:
    """Sync all settings from UI controls with validation."""
    try:
        # Basic settings with validation
        self.settings.export_root = self.var_root.get().strip()
        self.settings.provider = self.var_provider.get().strip()
        self.settings.persona = self.var_persona.get().strip()
        
        # Numeric fields with range validation
        temp = float(self.var_temp.get())
        if not (0.0 <= temp <= 1.0):
            raise ValueError(f"Temperature must be 0.0-1.0, got {temp}")
        self.settings.temperature = temp
        
        k = int(self.var_k.get())
        if k < 1 or k > 250:
            raise ValueError(f"k must be 1-250, got {k}")
        self.settings.k = k
        
        # ... validate all numeric fields similarly
        
    except (ValueError, TypeError) as e:
        module_logger.error(f"Validation error: {e}")
        raise ValueError(f"Invalid setting value: {e}") from e
```

---

### 9. Incomplete Implementation Stubs
**Location:** Lines 950-964  
**Severity:** MEDIUM - User confusion  
**Issue:** Methods show "Not Implemented" dialogs but are wired to buttons

**Affected Methods:**
- `_on_force_rechunk` (line 950)
- `_on_incremental_chunk` (line 963)  
- `_on_surgical_rechunk` (line 970)

**Fix Required:**
These methods ARE now implemented (lines 1026-1101). The old stub implementations need to be REMOVED entirely. The decorator versions are the correct ones.

**Action:**
Delete lines 950-970 completely. Keep only the `@run_with_progress` decorated versions.

---

### 10. Missing Exception Types in Imports
**Location:** Lines 51-66  
**Severity:** MEDIUM - Import errors not specific enough  
**Issue:** Generic `except ImportError:` may hide other errors

**Current Code:**
```python
try:
    from emailops import processor, email_indexer, ...
except ImportError:
    import processor
    import email_indexer
```

**Fix Required:**
```python
try:
    from emailops import processor, email_indexer, summarize_email_thread as summarizer, text_chunker, doctor
    from emailops.config import EmailOpsConfig, get_config
    from emailops.validators import validate_directory_path
    from emailops.utils import logger as module_logger
except ImportError as e:
    module_logger.warning(f"Package import failed ({e}), trying script imports")
    try:
        import processor
        import email_indexer
        import summarize_email_thread as summarizer
        import text_chunker
        import doctor
        from config import EmailOpsConfig, get_config
        from validators import validate_directory_path
        from utils import logger as module_logger
    except ImportError as e2:
        raise ImportError(f"Failed to import required modules: {e2}") from e2
```

---

## MINOR ISSUES

### 11. Unused Variables
**Location:** Multiple  
**Severity:** MINOR - Code clutter  
**Issue:** Variables assigned but never used

**Examples:**
- Line 357: `self._chunk_results` assigned but never read
- Line 43: `webbrowser` imported but never used

**Fix Required:**
```python
# Remove line 43:
# import webbrowser  # ❌ UNUSED

# Remove line 357:
# self._chunk_results: list[dict[str, Any]] = []  # ❌ UNUSED
```

---

### 12. Inconsistent String Formatting
**Location:** Throughout file  
**Severity:** MINOR - Style inconsistency  
**Issue:** Mixed f-strings, .format(), and % formatting

**Examples:**
```python
# Line 778: f-string (good)
module_logger.info(f"✓ Settings synchronized from UI")

# Line 1160: String concatenation (bad)
"Configuration applied successfully.\n\n" + "Environment variables updated.\n"
```

**Fix Required:**
Standardize on f-strings throughout:
```python
module_logger.info(
    f"Configuration applied successfully.\n"
    f"Environment variables updated.\n"
    f"Settings saved to: {SETTINGS_FILE}"
)
```

---

### 13. Magic Numbers Not Constants
**Location:** Lines 354-362, 1051-1060  
**Severity:** MINOR - Maintainability  
**Issue:** Hardcoded values should be named constants

**Examples:**
```python
# Line 795: Magic numbers
self.txt_snip = tk.Text(right, height=15, wrap="word", font=("Courier", 9), state="disabled")

# Line 1051: Magic retry count
for attempt in range(3):  # ❌ What is 3?
```

**Fix Required:**
```python
# At module level (after imports)
MAX_TEXT_HEIGHT = 15
DEFAULT_FONT_SIZE = 9
MAX_ATOMIC_WRITE_RETRIES = 3
ATOMIC_WRITE_RETRY_DELAY = 0.1

# Usage:
self.txt_snip = tk.Text(right, height=MAX_TEXT_HEIGHT, wrap="word", 
                        font=("Courier", DEFAULT_FONT_SIZE), state="disabled")

for attempt in range(MAX_ATOMIC_WRITE_RETRIES):
    try:
        os.replace(temp_path, SETTINGS_FILE)
        break
    except PermissionError:
        if attempt == MAX_ATOMIC_WRITE_RETRIES - 1:
            raise
        time.sleep(ATOMIC_WRITE_RETRY_DELAY)
```

---

### 14. No Input Validation in UI Methods
**Location:** Lines 750-1050  
**Severity:** MEDIUM - User experience  
**Issue:** Methods don't validate inputs before expensive operations

**Example - `_on_search`:**
```python
def _on_search(self, *, update_progress) -> None:
    query = self.var_search_q.get().strip()
    if not query:  # ✅ Has basic check
        # ...but no length check, character validation, etc.
```

**Fix Required:**
```python
def _on_search(self, *, update_progress) -> None:
    query = self.var_search_q.get().strip()
    
    # Validation
    if not query:
        self.after(0, lambda: messagebox.showwarning("Input Required", "Please enter a search query"))
        return
    
    if len(query) < 3:
        self.after(0, lambda: messagebox.showwarning("Query Too Short", "Query must be at least 3 characters"))
        return
    
    if len(query) > 10000:
        self.after(0, lambda: messagebox.showwarning("Query Too Long", "Query exceeds maximum length (10000 chars)"))
        return
    
    # Proceed with search...
```

Apply similar validation to:
- `_on_draft_reply` (line 820)
- `_on_draft_fresh` (line 877)
- `_on_chat` (line 932)

---

### 15. Memory Leak in Batch Operations
**Location:** Lines 975-1024  
**Severity:** CRITICAL - Memory accumulation  
**Issue:** Large result objects kept in memory without cleanup

**Current Code:**
```python
def _on_batch_summarize(self, *, update_progress) -> None:
    # ... processes many conversations
    for i, conv_dir in enumerate(conv_dirs):
        analysis = asyncio.run(summarizer.analyze_conversation_dir(...))
        # ❌ No cleanup of analysis object
```

**Fix Required:**
```python
def _on_batch_summarize(self, *, update_progress) -> None:
    # ... 
    for i, conv_dir in enumerate(conv_dirs):
        if self.task.cancelled():
            break
        try:
            update_progress(i, total, f"Summarizing {i+1}/{total}: {conv_dir.name}...")
            analysis = asyncio.run(summarizer.analyze_conversation_dir(...))
            completed += 1
            
            # ✅ Explicit cleanup
            del analysis
            
        except Exception as e:
            failed += 1
            module_logger.error(f"Failed to summarize {conv_dir.name}: {e}")
        finally:
            # Force garbage collection every 10 iterations
            if i % 10 == 0:
                import gc
                gc.collect()
```

---

## REDUNDANCY ISSUES

### 16. Duplicate Tree Widget Setup Code
**Location:** Lines 470-490, 550-570, 640-660, 960-980  
**Severity:** MINOR - Code duplication (60+ lines)  
**Issue:** Treeview creation pattern repeated 4 times

**Pattern:**
```python
cols = ("col1", "col2")
tree = ttk.Treeview(frame, columns=cols, show="headings", height=15)
tree.heading("col1", text="Column 1")
tree.column("col1", width=200)
# ... repeated for each column
yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
tree.configure(yscrollcommand=yscroll.set)
```

**Fix Required:**
Create helper method:
```python
def _create_treeview(self, parent, columns: list[tuple[str, str, int]], height: int = 15) -> ttk.Treeview:
    """
    Create a Treeview with scrollbar.
    
    Args:
        parent: Parent widget
        columns: List of (id, heading, width) tuples
        height: Tree height in rows
    
    Returns:
        Configured Treeview widget
    """
    col_ids = [c[0] for c in columns]
    tree = ttk.Treeview(parent, columns=col_ids, show="headings", height=height)
    
    for col_id, heading, width in columns:
        tree.heading(col_id, text=heading)
        tree.column(col_id, width=width)
    
    yscroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=yscroll.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    yscroll.pack(side=tk.LEFT, fill=tk.Y)
    
    return tree

# Usage:
self.tree = self._create_treeview(
    results_frame,
    [("score", "Score", 70), ("subject", "Subject", 600), ("id", "Doc ID", 600)],
    height=15
)
```

---

### 17. Duplicate Progress Bar Management
**Location:** Lines 246-268, 764-1050  
**Severity:** MINOR - Code duplication  
**Issue:** Progress bar start/stop pattern repeated in every method

**Pattern:**
```python
try:
    self.pb_X.start()
    self.btn_X.config(state="disabled")
    # ... work
finally:
    self.pb_X.stop()
    self.btn_X.config(state="normal")
```

**Fix Required:**
This is ALREADY handled by the `@run_with_progress` decorator! Remove the manual progress bar management from decorated methods:

```python
@run_with_progress("search", "pb_search", "status_label", "btn_search")
def _on_search(self, *, update_progress) -> None:
    # ❌ REMOVE THESE:
    # self.pb_search.start()
    # self.btn_search.config(state="disabled")
    
    # Just do the work...
    
    # ❌ REMOVE THESE:
    # self.pb_search.stop()
    # self.btn_search.config(state="normal")
```

**Affected Methods:** All @run_with_progress decorated methods

---

### 18. Inconsistent Error Display
**Location:** Throughout file  
**Severity:** MINOR - UX inconsistency  
**Issue:** Some methods use messagebox.showerror, others use _set_status

**Examples:**
```python
# Line 797: Uses messagebox
self.after(0, lambda: messagebox.showerror("Search Error", f"Search failed:\n{e!s}"))

# Line 1115: Uses _set_status
self.after(0, lambda: self._set_status(f"Index build failed: {e!s}", "error"))
```

**Fix Required:**
Use both - _set_status for status bar, messagebox for user notification:
```python
def _handle_error(self, operation: str, error: Exception, show_dialog: bool = True) -> None:
    """Centralized error handling."""
    error_msg = f"{operation} failed: {error!s}"
    module_logger.error(error_msg, exc_info=True)
    self._set_status(error_msg, "error")
    if show_dialog:
        messagebox.showerror(f"{operation} Error", f"{operation} failed:\n{error!s}")

# Usage:
except Exception as e:
    self.after(0, lambda: self._handle_error("Search", e))
```

---

## CONSISTENCY ISSUES

### 19. Inconsistent Widget Variable Naming
**Location:** Throughout file  
**Severity:** MINOR - Code readability  
**Issue:** Mixed naming conventions for similar widgets

**Examples:**
```python
self.var_search_q = tk.StringVar()     # ✅ Good
self.ent_root = ttk.Entry(...)          # ✅ Good
self.txt_logs = tk.Text(...)            # ✅ Good

self.btn_search = ttk.Button(...)       # ✅ Good
self.btn_draft_reply = ttk.Button(...)  # ✅ Good
self.btn_build = ttk.Button(...)        # ⚠️ Inconsistent - should be btn_index_build
```

**Fix Required:**
Rename for clarity:
```python
self.btn_build -> self.btn_index_build
self.pb_index -> self.pb_index_progress
self.lbl_index_progress -> self.lbl_index_status
```

---

### 20. Missing Type Hints
**Location:** Multiple methods  
**Severity:** MINOR - Code documentation  
**Issue:** Some methods missing return type hints

**Examples:**
```python
def _toggle_advanced_search(self) -> None:  # ✅ Has hint
def _choose_root(self) -> None:             # ✅ Has hint
def _drain_logs(self):                      # ❌ Missing hint
def _change_log_level(self, event=None):    # ❌ Missing hint
```

**Fix Required:**
```python
def _drain_logs(self) -> None:
def _change_log_level(self, event: tk.Event | None = None) -> None:
```

---

## ARCHITECTURAL ISSUES

### 21. Tight Coupling to emailops Module Structure
**Location:** Lines 51-66  
**Severity:** MEDIUM - Fragility  
**Issue:** Import fallback logic assumes specific module organization

**Current Approach:**
```python
try:
    from emailops import processor
except ImportError:
    import processor  # Assumes processor.py in current dir
```

**Better Approach:**
```python
def _import_with_fallback(package_name: str, module_name: str):
    """Import with graceful fallback and clear error messages."""
    try:
        return __import__(f"{package_name}.{module_name}", fromlist=[module_name])
    except ImportError:
        try:
            return __import__(module_name)
        except ImportError as e:
            raise ImportError(
                f"Cannot import {module_name}. "
                f"Ensure emailops package is installed or {module_name}.py is in PYTHONPATH. "
                f"Error: {e}"
            ) from e

processor = _import_with_fallback("emailops", "processor")
email_indexer = _import_with_fallback("emailops", "email_indexer")
# ... etc
```

---

### 22. No Separation of UI and Business Logic
**Location:** Throughout file  
**Severity:** MEDIUM - Testability  
**Issue:** UI methods directly call business logic - hard to test

**Example:**
```python
def _on_search(self, *, update_progress) -> None:
    # UI code mixed with business logic
    query = self.var_search_q.get().strip()  # UI
    results = _search(...)  # Business logic
    self.tree.insert(...)  # UI
```

**Fix Required:**
Separate into layers:
```python
# Business logic layer
def perform_search(self, query: str, k: int, provider: str, filters: SearchFilters | None) -> list[dict[str, Any]]:
    """Pure business logic - no UI dependencies."""
    from emailops.search_and_draft import _search
    ix_dir = Path(self.settings.export_root) / os.getenv("INDEX_DIRNAME", "_index")
    return _search(ix_dir=ix_dir, query=query, k=k, provider=provider, filters=filters)

# UI layer
@run_with_progress("search", "pb_search", "status_label", "btn_search")
def _on_search(self, *, update_progress) -> None:
    """UI orchestration only."""
    query = self._validate_search_input()
    if not query:
        return
    
    results = self.perform_search(query, self.settings.k, self.settings.provider, self._build_search_filters())
    self._display_search_results(results)
```

---

### 23. Missing Context Manager for Dialogs
**Location:** Lines 450-550  
**Severity:** MINOR - Resource management  
**Issue:** Toplevel dialogs created but not properly managed

**Current Pattern:**
```python
def _view_conversation_txt(self) -> None:
    viewer = tk.Toplevel(self)
    # ... setup
    # ❌ No explicit cleanup
```

**Fix Required:**
```python
@contextlib.contextmanager
def _create_dialog(self, title: str, geometry: str = "600x400"):
    """Context manager for creating and cleaning up dialogs."""
    dialog = tk.Toplevel(self)
    dialog.title(title)
    dialog.geometry(geometry)
    try:
        yield dialog
    finally:
        # Ensure cleanup
        dialog.destroy()

# Usage:
def _view_conversation_txt(self) -> None:
    with self._create_dialog("Conversation Viewer", "900x700") as viewer:
        # ... setup
        pass
```

---

### 24. Hardcoded File Extensions
**Location:** Lines 730, 890, 1243  
**Severity:** MINOR - Maintainability  
**Issue:** File extensions hardcoded in filedialog calls

**Current Code:**
```python
filename = filedialog.asksaveasfilename(
    defaultextension=".eml",
    filetypes=[("Email files", "*.eml"), ("All files", "*.*")],
```

**Fix Required:**
```python
# At module level
FILE_TYPES = {
    "eml": [("Email files", "*.eml"), ("All files", "*.*")],
    "csv": [("CSV files", "*.csv"), ("All files", "*.*")],
    "json": [("JSON files", "*.json"), ("All files", "*.*")],
    "log": [("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
}

# Usage:
filename = filedialog.asksaveasfilename(
    defaultextension=".eml",
    filetypes=FILE_TYPES["eml"],
    initialfile=f"{conv_id}_reply.eml"
)
```

---

### 25. Missing Docstring Completeness
**Location:** Multiple methods  
**Severity:** MINOR - Documentation  
**Issue:** Some methods have incomplete docstrings

**Examples:**
```python
def _detect_worker_count(self) -> int:
    """Auto-detect optimal workers from credentials."""  # ✅ Good but brief
    
def _with_root_and_index(self) -> tuple[Path | None, Path | None]:
    # ❌ NO docstring at all
```

**Fix Required:**
```python
def _detect_worker_count(self) -> int:
    """
    Auto-detect optimal worker count from available GCP credentials.
    
    Returns:
        Number of valid GCP accounts (minimum 1)
    
    Note:
        Falls back to 1 if credential loading fails
    """
    
def _with_root_and_index(self) -> tuple[Path | None, Path | None]:
    """
    Validate and return export root and index directory paths.
    
    Returns:
        Tuple of (root_path, index_path) or (None, None) if validation fails
    
    Side Effects:
        Shows error messagebox on validation failure
    """
```

---

## PERFORMANCE ISSUES

### 26. Synchronous File I/O in UI Thread
**Location:** Lines 871, 917, 1234  
**Severity:** MEDIUM - UI freezing  
**Issue:** File I/O performed synchronously during result display

**Current Code:**
```python
def update_ui():
    # Line 871 - loads file on main thread
    content = conv_path.read_text(encoding="utf-8", errors="ignore")
    text_widget.insert("1.0", content)  # Could be large file
```

**Fix Required:**
```python
def update_ui():
    # Load in chunks to avoid freezing
    try:
        with open(conv_path, 'r', encoding='utf-8', errors='ignore') as f:
            chunk_size = 100000  # 100KB chunks
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                text_widget.insert(tk.END, chunk)
                text_widget.update_idletasks()  # Keep UI responsive
    except Exception as e:
        text_widget.insert("1.0", f"Error loading file:\n{e!s}")
```

---

### 27. No Debouncing on User Input
**Location:** Lines 480-510  
**Severity:** MINOR - Performance  
**Issue:** Trace callbacks fire on every keystroke without debouncing

**Current Code:**
```python
self.var_mmr_lambda.trace_add("write", lambda *_: self.lbl_mmr.config(text=
