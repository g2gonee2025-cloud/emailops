# EmailOps Vertex AI - Refactoring Plan

## Current Issues Identified

1. **Flat Structure**: 20+ Python scripts in root directory making it hard to navigate
2. **Duplicate Code**: Multiple diagnostic scripts with similar functionality (diagnose_accounts.py vs diagnose_accounts_fixed.py)
3. **Mixed Concerns**: Testing, processing, utilities, and setup scripts all mixed together
4. **Inconsistent Error Handling**: Some scripts have try-catch blocks, others don't
5. **Missing Documentation**: No central README or proper docstrings
6. **No Clear Entry Point**: Multiple scripts that could be main entry points

## Proposed New Directory Structure

```
emailops_vertex_ai/
│
├── emailops/               # Core library (keep as-is)
│   ├── __init__.py
│   ├── doctor.py
│   ├── email_indexer.py
│   ├── env_utils.py
│   ├── index_metadata.py
│   ├── llm_client.py
│   ├── search_and_draft.py
│   ├── summarize_email_thread.py
│   ├── text_chunker.py
│   └── utils.py
│
├── diagnostics/            # Diagnostic and debugging tools
│   ├── __init__.py
│   ├── diagnose_accounts.py (consolidated version)
│   ├── debug_parallel_indexer.py
│   ├── check_failed_batches.py
│   ├── verify_index_alignment.py
│   └── check_all_files.py
│
├── processing/             # Data processing scripts
│   ├── __init__.py
│   ├── vertex_indexer.py
│   ├── parallel_chunker.py
│   ├── parallel_summarizer.py
│   ├── fix_failed_embeddings.py
│   ├── repair_vertex_parallel_index.py
│   └── run_vertex_finalize.py
│
├── analysis/               # Analysis and statistics tools
│   ├── __init__.py
│   ├── file_processing_analysis.py
│   ├── file_stats.py
│   ├── count_chunks.py
│   └── monitor_indexing.py
│
├── tests/                  # Test scripts
│   ├── __init__.py
│   └── test_all_accounts_live.py
│
├── setup/                  # Setup and configuration
│   ├── __init__.py
│   ├── enable_vertex_apis.py
│   ├── setup_vertex_env.bat
│   ├── activate_env.bat
│   └── activate_env.ps1
│
├── utils/                  # Utility modules
│   ├── __init__.py
│   └── vertex_utils.py
│
├── data/                   # Data files
│   ├── validated_accounts.json
│   ├── account_diagnostics.json
│   ├── account_diagnostics_fixed.json
│   └── live_api_test_results.json
│
├── docs/                   # Documentation
│   └── WORKER_ISSUE_REPORT.md
│
├── ui/                     # User interface
│   └── emailops_ui.py
│
├── cli.py                  # Main CLI entry point
├── requirements.txt        # Dependencies
├── Dockerfile             # Container configuration
├── environment.yml        # Conda environment
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
├── README.md             # Project documentation
└── .streamlit/           # Streamlit configuration
```

## Key Refactoring Actions

### 1. Directory Organization
- Group scripts by functionality into logical folders
- Maintain clear separation of concerns
- Keep core library (emailops/) intact

### 2. Code Consolidation
- Merge `diagnose_accounts.py` and `diagnose_accounts_fixed.py` into single improved version
- Combine similar functionality from various scripts
- Remove duplicate code

### 3. Code Quality Improvements
- Add type hints to all functions
- Add comprehensive docstrings
- Implement consistent error handling
- Add logging instead of print statements

### 4. Create Unified CLI
- Single entry point (`cli.py`) with subcommands
- Use argparse or click for better CLI experience
- Example: `python cli.py diagnose --account all`

### 5. Documentation
- Create comprehensive README.md
- Document each module's purpose
- Add usage examples
- Include setup instructions

## Implementation Priority

1. **Phase 1 - Structure** (High Priority)
   - Create directory structure
   - Move files to appropriate folders
   - Update imports

2. **Phase 2 - Consolidation** (High Priority)
   - Merge duplicate scripts
   - Remove unused code
   - Consolidate utilities

3. **Phase 3 - Quality** (Medium Priority)
   - Add error handling
   - Add type hints
   - Add docstrings

4. **Phase 4 - Polish** (Low Priority)
   - Create CLI entry point
   - Write documentation
   - Add tests

## Benefits After Refactoring

1. **Better Organization**: Clear structure makes navigation easier
2. **Reduced Duplication**: Less code to maintain
3. **Improved Reliability**: Better error handling and testing
4. **Easier Onboarding**: Clear documentation and structure
5. **Maintainability**: Modular design allows easier updates
6. **Professional Quality**: Production-ready codebase

## Next Steps

1. Review this plan and provide feedback
2. Switch to Code mode to implement the refactoring
3. Test each component after moving
4. Update documentation