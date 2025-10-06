# Implementation Plan

## [Overview]

Consolidate and refactor the `analysis/` folder to reduce clutter by combining 6 separate analysis scripts into 3 focused, well-organized modules with shared utilities.

The current `analysis/` folder contains overlapping functionality across multiple scripts with inconsistent patterns, hard-coded paths, and redundant diagnostic capabilities. This refactoring will create a cleaner architecture with:
- **diagnostics.py** - Unified diagnostic functionality for account testing and index verification
- **statistics.py** - Consolidated analysis and monitoring for file statistics, chunk counting, and indexing progress
- **utils.py** - Shared utility functions for common operations like path handling and logging setup

This consolidation reduces file count from 6 scripts to 3 modules (plus `__init__.py`), eliminates code duplication, standardizes logging patterns, and improves maintainability while preserving all existing CLI functionality.

## [Types]

No new types or data structures are required for this refactoring.

The consolidation uses existing type hints and data structures from the current scripts:
- Function signatures remain compatible with existing CLI calls
- Return types (bool, int, dict, None) are preserved from original implementations
- Path objects continue to use `pathlib.Path`
- Configuration parameters maintain their current types (str, int, bool, Optional)

## [Files]

Files to be created, modified, and deleted during the consolidation.

**New Files to Create:**
- `analysis/diagnostics.py` - Consolidates `diagnose_accounts.py` and `verify_index_alignment.py`
  - Purpose: Unified diagnostic module for testing Vertex AI accounts and verifying index integrity
  - Contains: `test_account()`, `diagnose_all_accounts()`, `verify_index_alignment()`, `check_index_consistency()`
  
- `analysis/statistics.py` - Consolidates `file_processing_analysis.py`, `file_stats.py`, `count_chunks.py`, and `monitor_indexing.py`
  - Purpose: Centralized analysis and monitoring functionality
  - Contains: `analyze_file_processing()`, `get_file_statistics()`, `count_total_chunks()`, `monitor_indexing_progress()`
  
- `analysis/utils.py` - New shared utilities module
  - Purpose: Common functions used across analysis modules
  - Contains: `setup_logging()`, `get_index_path()`, `get_export_root()`, `format_timestamp()`, `save_json_report()`

**Files to Modify:**
- `analysis/__init__.py` - Update to export new consolidated modules and maintain backward compatibility
- `cli.py` - Update import statements to use new module structure (lines 150-180 approximately)

**Files to Delete:**
- `analysis/count_chunks.py` - Functionality moved to `statistics.py`
- `analysis/diagnose_accounts.py` - Functionality moved to `diagnostics.py`
- `analysis/file_processing_analysis.py` - Functionality moved to `statistics.py`
- `analysis/file_stats.py` - Functionality moved to `statistics.py`
- `analysis/monitor_indexing.py` - Functionality moved to `statistics.py`
- `analysis/verify_index_alignment.py` - Functionality moved to `diagnostics.py`

**Configuration Files:**
- No changes to `.env`, `requirements.txt`, or other configuration files

## [Functions]

Detailed breakdown of function consolidation and modifications.

**New Functions in `analysis/diagnostics.py`:**
- `test_account(account: VertexAccount) -> Tuple[bool, str]`
  - Location: analysis/diagnostics.py
  - Purpose: Test a single Vertex AI account (from diagnose_accounts.py)
  - Parameters: account object with project_id, credentials_path, account_group
  - Returns: (success: bool, message: str)
  
- `diagnose_all_accounts() -> int`
  - Location: analysis/diagnostics.py
  - Purpose: Main entry point for account diagnostics (from diagnose_accounts.py)
  - Returns: Exit code (0 for success, 1 for failures)
  - Creates diagnostic report as JSON
  
- `verify_index_alignment(root: str) -> None`
  - Location: analysis/diagnostics.py
  - Purpose: Verify index integrity and alignment (from verify_index_alignment.py)
  - Parameters: root directory path
  - Validates mapping.json, embeddings.npy, and meta.json consistency
  
- `check_index_consistency(root: Path) -> Dict[str, Any]`
  - Location: analysis/diagnostics.py
  - Purpose: New function to perform detailed consistency checks
  - Returns: Dictionary with check results and recommendations

**New Functions in `analysis/statistics.py`:**
- `analyze_file_processing() -> None`
  - Location: analysis/statistics.py
  - Purpose: Display file processing analysis (from file_processing_analysis.py)
  - Prints detailed breakdown of which files get chunked vs ignored
  
- `get_file_statistics(root: Path) -> Dict[str, Any]`
  - Location: analysis/statistics.py
  - Purpose: Generate file statistics for directory (from file_stats.py)
  - Parameters: root path to Outlook export
  - Returns: Dictionary with file counts by extension
  
- `count_total_chunks(export_dir: str) -> int`
  - Location: analysis/statistics.py
  - Purpose: Count total chunks in embeddings directory (from count_chunks.py)
  - Parameters: export directory path
  - Returns: Total chunk count
  
- `monitor_indexing_progress(log_file: Optional[Path] = None) -> Dict[str, Any]`
  - Location: analysis/statistics.py
  - Purpose: Monitor real-time indexing progress (from monitor_indexing.py)
  - Parameters: Optional log file path (auto-detects if not provided)
  - Returns: Progress statistics dictionary

**New Functions in `analysis/utils.py`:**
- `setup_logging(level: str = "INFO") -> logging.Logger`
  - Location: analysis/utils.py
  - Purpose: Configure logging with consistent format
  - Parameters: log level string
  - Returns: Configured logger instance
  
- `get_index_path(root: Optional[str] = None) -> Path`
  - Location: analysis/utils.py
  - Purpose: Resolve index directory path from root or environment
  - Parameters: Optional root directory
  - Returns: Path to _index directory
  
- `get_export_root() -> Path`
  - Location: analysis/utils.py
  - Purpose: Determine export root from environment or current directory
  - Returns: Path to export root
  
- `format_timestamp(dt: datetime) -> str`
  - Location: analysis/utils.py
  - Purpose: Consistent timestamp formatting
  - Returns: Formatted timestamp string
  
- `save_json_report(data: Dict[str, Any], filename: str) -> Path`
  - Location: analysis/utils.py
  - Purpose: Save analysis results as JSON with proper formatting
  - Returns: Path to saved file

**Modified Functions in `cli.py`:**
- `main()` - Update import paths in the analyze and diagnose command handlers
  - Lines ~150-180: Change from `from analysis.file_stats import main` to `from analysis.statistics import get_file_statistics`
  - Lines ~160-165: Change from `from analysis.diagnose_accounts import main` to `from analysis.diagnostics import diagnose_all_accounts`
  - Maintain same command-line interface and behavior

**Functions Being Removed:**
- All functions in deleted files are moved, not removed:
  - `count_chunks.py::count_total_chunks()` → `statistics.py::count_total_chunks()`
  - `diagnose_accounts.py::test_account()` → `diagnostics.py::test_account()`
  - `diagnose_accounts.py::main()` → `diagnostics.py::diagnose_all_accounts()`
  - `file_processing_analysis.py::<print statements>` → `statistics.py::analyze_file_processing()`
  - `file_stats.py::<main code>` → `statistics.py::get_file_statistics()`
  - `monitor_indexing.py::monitor_worker2()` → `statistics.py::monitor_indexing_progress()`
  - `verify_index_alignment.py::main()` → `diagnostics.py::verify_index_alignment()`

## [Classes]

No new classes or class modifications are required for this refactoring.

All functionality uses functions and existing data structures:
- `VertexAccount` class (from emailops.env_utils) is used but not modified
- No new classes are introduced
- No existing classes are modified or removed

The refactoring maintains a functional programming approach consistent with the existing codebase structure.

## [Dependencies]

No new dependencies are required for this refactoring.

**Existing Dependencies Used:**
- Standard library: `os`, `sys`, `json`, `logging`, `pathlib`, `datetime`, `pickle`, `subprocess`
- Project modules: `emailops.env_utils`, `emailops.llm_client`
- Third-party (already in requirements.txt): `numpy` (for embeddings verification)

**No Changes Required:**
- `requirements.txt` - No additions or modifications
- `environment.yml` - No changes
- All required packages are already installed and listed

**Import Consolidation:**
- Duplicate imports across old scripts are consolidated in new modules
- Shared utilities reduce redundant import statements
- No new external dependencies introduced

## [Testing]

Testing strategy to ensure refactoring maintains functionality.

**Validation Approach:**
- Manual testing of all CLI commands that use analysis functions
- Verify output consistency between old and new implementations
- Test error handling and edge cases

**Test Cases:**
1. **Diagnostics Testing:**
   - Run `python cli.py diagnose --accounts` and verify account testing works
   - Run `python cli.py diagnose --index` with valid index directory
   - Test with missing credentials files
   - Test with invalid index directory
   - Verify JSON report generation

2. **Statistics Testing:**
   - Run `python cli.py analyze --stats` and compare output format
   - Run `python cli.py analyze --chunks` and verify chunk counting
   - Run `python cli.py analyze --files` and check file processing analysis
   - Run `python cli.py monitor` and verify progress monitoring
   - Test with missing directories and incomplete indexes

3. **Integration Testing:**
   - Verify all CLI commands still function correctly
   - Test with various directory structures
   - Confirm backward compatibility with existing workflows
   - Validate error messages and logging output

**Test Files:**
- No new test files needed initially
- Existing manual testing workflow is sufficient for validation
- Future enhancement: Add unit tests in `tests/test_analysis.py`

**Validation Checklist:**
- [ ] All CLI commands execute without errors
- [ ] Output format matches previous implementations
- [ ] Error handling is consistent and informative
- [ ] Log files are created with proper formatting
- [ ] JSON reports are valid and complete
- [ ] No regression in functionality

## [Implementation Order]

Numbered steps showing the logical sequence of implementation.

**Step 1: Create Shared Utilities Module**
- Create `analysis/utils.py` with common functions
- Implement logging setup, path resolution, and formatting utilities
- Test utility functions independently
- Rationale: Other modules depend on these utilities

**Step 2: Create Diagnostics Module**
- Create `analysis/diagnostics.py`
- Migrate `test_account()` from `diagnose_accounts.py`
- Migrate `verify_index_alignment()` from `verify_index_alignment.py`
- Add new `check_index_consistency()` function
- Import shared utilities from Step 1
- Test diagnostic functions with sample data

**Step 3: Create Statistics Module**
- Create `analysis/statistics.py`
- Migrate `count_total_chunks()` from `count_chunks.py`
- Migrate file statistics logic from `file_stats.py`
- Migrate analysis logic from `file_processing_analysis.py`
- Migrate monitoring logic from `monitor_indexing.py`
- Import shared utilities from Step 1
- Test each function individually

**Step 4: Update Package Initialization**
- Modify `analysis/__init__.py`
- Export new module functions
- Add backward compatibility imports if needed
- Update module docstring

**Step 5: Update CLI Integration**
- Modify `cli.py` to import from new modules
- Update analyze command handlers (--files, --stats, --chunks)
- Update diagnose command handlers (--accounts, --index)
- Update monitor command handler
- Test each CLI command thoroughly

**Step 6: Verification and Cleanup**
- Run complete test suite (manual testing of all commands)
- Verify output consistency with original implementations
- Confirm no regressions in functionality
- Document any changes in behavior

**Step 7: Remove Old Files**
- Delete `analysis/count_chunks.py`
- Delete `analysis/diagnose_accounts.py`
- Delete `analysis/file_processing_analysis.py`
- Delete `analysis/file_stats.py`
- Delete `analysis/monitor_indexing.py`
- Delete `analysis/verify_index_alignment.py`
- Only delete after confirming new implementation works

**Step 8: Final Documentation**
- Update README.md if analysis commands are documented
- Add inline documentation to new modules
- Create summary document of changes
- Commit changes with descriptive message
