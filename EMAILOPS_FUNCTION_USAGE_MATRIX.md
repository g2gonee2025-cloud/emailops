# EmailOps Function Usage Matrix

## Executive Summary

This matrix maps function dependencies across the `emailops/` modules, showing how components interact and identifying critical integration points.

## Core Module Architecture

```
┌─────────────────┐
│     config.py   │ ←── Central configuration (used by ALL modules)
└─────────────────┘

┌─────────────────┐
│   llm_runtime.py│ ←── Core LLM functionality (used by client, search, summarizer)
└─────────────────┘

┌─────────────────┐
│     utils.py    │ ←── Text processing utilities (used by indexer, search, summarizer)
└─────────────────┘
```

## Function Usage Matrix

### 🔧 **config.py** - Configuration Management
| Function | Used By | Usage Pattern |
|----------|---------|---------------|
| [`get_config()`](emailops/config.py:188) | `email_indexer`, `llm_runtime`, `processor`, `search_and_draft`, `utils` | **HIGH** - Singleton pattern |
| [`EmailOpsConfig.load()`](emailops/config.py:67) | `email_indexer`, `search_and_draft`, `utils` | **HIGH** - Configuration loading |
| [`EmailOpsConfig.update_environment()`](emailops/config.py:132) | `email_indexer`, `processor` | **MEDIUM** - Env setup |
| [`EmailOpsConfig.get_credential_file()`](emailops/config.py:99) | `config` | **LOW** - Internal credential discovery |

### 🔍 **doctor.py** - System Diagnostics
| Function | Used By | Usage Pattern |
|----------|---------|---------------|
| [`check_and_install_dependencies()`](emailops/doctor.py:166) | `doctor` CLI | **LOW** - Diagnostic tool |
| [`_normalize_provider()`](emailops/doctor.py:33) | `doctor` | **LOW** - Internal helper |
| [`_probe_embeddings()`](emailops/doctor.py:249) | `doctor` | **LOW** - Health check |

### 📚 **email_indexer.py** - Vector Index Builder
| Function | Used By | Usage Pattern |
|----------|---------|---------------|
| [`build_corpus()`](emailops/email_indexer.py:560) | `email_indexer` | **HIGH** - Core indexing |
| [`build_incremental_corpus()`](emailops/email_indexer.py:726) | `email_indexer` | **HIGH** - Incremental updates |
| [`save_index()`](emailops/email_indexer.py:950) | `email_indexer` | **HIGH** - Index persistence |
| [`load_existing_index()`](emailops/email_indexer.py:891) | `email_indexer` | **HIGH** - Index loading |
| [`_build_doc_entries()`](emailops/email_indexer.py:421) | `email_indexer` | **MEDIUM** - Document processing |
| [`_extract_manifest_metadata()`](emailops/email_indexer.py:278) | `email_indexer` | **MEDIUM** - Metadata extraction |
| [`_initialize_gcp_credentials()`](emailops/email_indexer.py:211) | `email_indexer` | **MEDIUM** - GCP setup |

### 📊 **index_metadata.py** - Index Metadata Management
| Function | Used By | Usage Pattern |
|----------|---------|---------------|
| [`read_mapping()`](emailops/index_metadata.py:783) | `email_indexer`, `search_and_draft`, `doctor` | **HIGH** - Index data access |
| [`write_mapping()`](emailops/index_metadata.py:806) | `email_indexer` | **HIGH** - Index data persistence |
| [`load_index_metadata()`](emailops/index_metadata.py:466) | `search_and_draft`, `doctor`, `index_metadata` | **HIGH** - Metadata access |
| [`validate_index_compatibility()`](emailops/index_metadata.py:631) | `search_and_draft` | **HIGH** - Provider validation |
| [`index_paths()`](emailops/index_metadata.py:107) | `email_indexer`, `index_metadata` | **MEDIUM** - Path resolution |
| [`create_index_metadata()`](emailops/index_metadata.py:375) | `email_indexer` | **MEDIUM** - Metadata creation |
| [`check_index_consistency()`](emailops/index_metadata.py:482) | `email_indexer` | **MEDIUM** - Data integrity |

### 🤖 **llm_runtime.py** - LLM & Embedding Services
| Function | Used By | Usage Pattern |
|----------|---------|---------------|
| [`embed_texts()`](emailops/llm_runtime.py:674) | `email_indexer`, `search_and_draft`, `doctor` | **CRITICAL** - Core embedding |
| [`complete_text()`](emailops/llm_runtime.py:544) | `search_and_draft`, `summarize_email_thread` | **HIGH** - Text generation |
| [`complete_json()`](emailops/llm_runtime.py:598) | `search_and_draft`, `summarize_email_thread` | **HIGH** - Structured generation |
| [`_init_vertex()`](emailops/llm_runtime.py:271) | `llm_runtime`, `env_utils` | **HIGH** - GCP initialization |
| [`load_validated_accounts()`](emailops/llm_runtime.py:137) | `llm_runtime` | **MEDIUM** - Account management |
| [`_check_rate_limit()`](emailops/llm_runtime.py:65) | `llm_runtime` | **MEDIUM** - API throttling |

### 🎛️ **processor.py** - CLI Orchestrator
| Function | Used By | Usage Pattern |
|----------|---------|---------------|
| [`_ensure_env()`](emailops/processor.py:119) | `processor` | **HIGH** - Environment setup |
| [`_run_email_indexer()`](emailops/processor.py:145) | `processor` | **HIGH** - Subprocess execution |
| [`cmd_index()`, `cmd_reply()`, etc.](emailops/processor.py:205) | `processor` CLI | **HIGH** - Command handlers |
| [`_summarize_worker()`](emailops/processor.py:340) | `processor` | **MEDIUM** - Multiprocessing worker |

### 🔍 **search_and_draft.py** - Search & Email Generation
| Function | Used By | Usage Pattern |
|----------|---------|---------------|
| [`draft_email_reply_eml()`](emailops/search_and_draft.py:1616) | `processor` | **HIGH** - Email reply generation |
| [`draft_fresh_email_eml()`](emailops/search_and_draft.py:1706) | `processor` | **HIGH** - Fresh email generation |
| [`chat_with_context()`](emailops/search_and_draft.py:1859) | `processor` | **HIGH** - Contextual chat |
| [`_search()`](emailops/search_and_draft.py:1948) | `search_and_draft`, `processor` | **HIGH** - Core search functionality |
| [`draft_email_structured()`](emailops/search_and_draft.py:1271) | `search_and_draft` | **HIGH** - Structured drafting |
| [`list_conversations_newest_first()`](emailops/search_and_draft.py:496) | External CLI | **MEDIUM** - Conversation listing |
| [`parse_filter_grammar()`](emailops/search_and_draft.py:623) | `search_and_draft`, `processor` | **MEDIUM** - Query parsing |

### 📝 **summarize_email_thread.py** - Email Thread Analysis
| Function | Used By | Usage Pattern |
|----------|---------|---------------|
| [`analyze_conversation_dir()`](emailops/summarize_email_thread.py:1740) | `processor` | **HIGH** - Thread analysis |
| [`format_analysis_as_markdown()`](emailops/summarize_email_thread.py:1776) | `processor` | **HIGH** - Output formatting |
| [`analyze_email_thread_with_ledger()`](emailops/summarize_email_thread.py:1163) | `summarize_email_thread` | **HIGH** - Core analysis |
| [`_normalize_analysis()`](emailops/summarize_email_thread.py:365) | `summarize_email_thread` | **MEDIUM** - Data normalization |
| [`_try_load_json()`](emailops/summarize_email_thread.py:140) | `summarize_email_thread` | **MEDIUM** - JSON parsing |

### ✂️ **text_chunker.py** - Text Chunking
| Function | Used By | Usage Pattern |
|----------|---------|---------------|
| [`prepare_index_units()`](emailops/text_chunker.py:236) | `email_indexer` | **CRITICAL** - Text chunking for indexing |
| [`TextChunker.chunk_text()`](emailops/text_chunker.py:191) | `text_chunker` | **MEDIUM** - Programmatic chunking |
| [`_ranges_with_overlap()`](emailops/text_chunker.py:95) | `text_chunker` | **MEDIUM** - Internal chunking logic |

### 🛠️ **utils.py** - Core Utilities
| Function | Used By | Usage Pattern |
|----------|---------|---------------|
| [`read_text_file()`](emailops/utils.py:160) | `email_indexer`, `summarize_email_thread`, `search_and_draft` | **CRITICAL** - File reading |
| [`clean_email_text()`](emailops/utils.py:675) | `email_indexer`, `search_and_draft`, `summarize_email_thread`, `llm_runtime` | **CRITICAL** - Text cleaning |
| [`extract_text()`](emailops/utils.py:353) | `utils` | **HIGH** - Multi-format text extraction |
| [`find_conversation_dirs()`](emailops/utils.py:824) | `email_indexer`, `utils` | **HIGH** - Directory discovery |
| [`load_conversation()`](emailops/utils.py:831) | `email_indexer`, `utils` | **HIGH** - Conversation loading |
| [`ensure_dir()`](emailops/utils.py:993) | `email_indexer`, `summarize_email_thread` | **HIGH** - Directory creation |
| [`extract_email_metadata()`](emailops/utils.py:724) | `summarize_email_thread` | **MEDIUM** - Header parsing |
| [`get_text_preprocessor()`](emailops/utils.py:400) | `search_and_draft` | **MEDIUM** - Text optimization |
| [`should_skip_retrieval_cleaning()`](emailops/utils.py:373) | `search_and_draft` | **MEDIUM** - Performance optimization |
| [`monitor_performance()`](emailops/utils.py:1039) | `llm_runtime` | **MEDIUM** - Performance monitoring |

### ✅ **validators.py** - Input Validation
| Function | Used By | Usage Pattern |
|----------|---------|---------------|
| [`validate_file_path()`](emailops/validators.py:119) | `search_and_draft` | **HIGH** - Security validation |
| [`validate_command_args()`](emailops/validators.py:207) | `processor` | **MEDIUM** - Command security |
| [`validate_directory_path()`](emailops/validators.py:63) | External usage | **MEDIUM** - Path security |
| [`validate_email_format()`](emailops/validators.py:315) | External usage | **LOW** - Email validation |

## Critical Integration Points

### 🎯 **Most Used Functions (Cross-Module)**
1. **[`get_config()`](emailops/config.py:188)** - Used by 6+ modules for configuration
2. **[`embed_texts()`](emailops/llm_runtime.py:674)** - Used by indexer, search, diagnostics
3. **[`read_text_file()`](emailops/utils.py:160)** - Used by indexer, summarizer, search
4. **[`clean_email_text()`](emailops/utils.py:675)** - Used by indexer, search, summarizer, runtime
5. **[`read_mapping()`](emailops/index_metadata.py:783)** - Used by indexer, search, diagnostics

### 🔄 **Module Dependency Flow**
```
config.py → (provides config to) → ALL modules
    ↓
llm_runtime.py → (provides LLM services to) → search_and_draft, summarize_email_thread, email_indexer
    ↓
utils.py → (provides utilities to) → email_indexer, search_and_draft, summarize_email_thread
    ↓
index_metadata.py → (provides index mgmt to) → email_indexer, search_and_draft
    ↓
text_chunker.py → (provides chunking to) → email_indexer
    ↓
validators.py → (provides validation to) → search_and_draft, processor
```

### 🚨 **High-Risk Dependencies**
- **[`embed_texts()`](emailops/llm_runtime.py:674)** failure breaks indexing and search
- **[`get_config()`](emailops/config.py:188)** failure breaks entire system  
- **[`read_text_file()`](emailops/utils.py:160)** failure breaks text processing
- **[`prepare_index_units()`](emailops/text_chunker.py:236)** failure breaks indexing

## API Surface Analysis

### 📤 **Public APIs (Entry Points)**
| Module | Primary APIs | Used By |
|--------|-------------|---------|
| `email_indexer.py` | [`main()`](emailops/email_indexer.py:1004) | CLI users |
| `processor.py` | [`main()`](emailops/processor.py:505), [`build_cli()`](emailops/processor.py:430) | CLI users |
| `search_and_draft.py` | [`draft_email_reply_eml()`](emailops/search_and_draft.py:1616), [`draft_fresh_email_eml()`](emailops/search_and_draft.py:1706), [`chat_with_context()`](emailops/search_and_draft.py:1859) | `processor`, external |
| `summarize_email_thread.py` | [`analyze_conversation_dir()`](emailops/summarize_email_thread.py:1740), [`format_analysis_as_markdown()`](emailops/summarize_email_thread.py:1776) | `processor`, external |

### 🔗 **Internal Dependencies**
| Function | Dependency Chain | Risk Level |
|----------|-----------------|------------|
| [`draft_email_reply_eml()`](emailops/search_and_draft.py:1616) | → [`_gather_context_for_conv()`](emailops/search_and_draft.py:919) → [`embed_texts()`](emailops/llm_runtime.py:674) → [`_init_vertex()`](emailops/llm_runtime.py:271) | 🔴 **CRITICAL** |
| [`build_corpus()`](emailops/email_indexer.py:560) | → [`prepare_index_units()`](emailops/text_chunker.py:236) → [`clean_email_text()`](emailops/utils.py:675) | 🔴 **CRITICAL** |
| [`analyze_conversation_dir()`](emailops/summarize_email_thread.py:1740) | → [`complete_json()`](emailops/llm_runtime.py:598) → [`_init_vertex()`](emailops/llm_runtime.py:271) | 🔴 **CRITICAL** |

## Usage Patterns

### 🎯 **Singleton Pattern**
- **[`get_config()`](emailops/config.py:188)** - Global configuration instance
- **[`get_text_preprocessor()`](emailops/utils.py:400)** - Global text processor

### 🔄 **Strategy Pattern**
- **[`embed_texts()`](emailops/llm_runtime.py:674)** - Provider-specific embedding strategies
- **[`extract_text()`](emailops/utils.py:353)** - File-type-specific extraction strategies

### 🏭 **Factory Pattern**
- **[`index_paths()`](emailops/index_metadata.py:107)** - Creates [`IndexPaths`](emailops/index_metadata.py:96) objects
- **[`create_index_metadata()`](emailops/index_metadata.py:375)** - Creates metadata dictionaries

### 🛡️ **Decorator Pattern**
- **[`monitor_performance()`](emailops/utils.py:1039)** - Performance monitoring decorator
- **[`log_timing()`](emailops/search_and_draft.py:198)** - Timing context manager

## Function Complexity Analysis

### 🏗️ **High Complexity (100+ lines)**
- **[`build_corpus()`](emailops/email_indexer.py:560)** - 160 lines - Complex indexing logic
- **[`draft_email_structured()`](emailops/search_and_draft.py:1271)** - 280 lines - Multi-stage email generation
- **[`analyze_email_thread_with_ledger()`](emailops/summarize_email_thread.py:1163)** - 570 lines - Complex LLM analysis
- **[`_gather_context_for_conv()`](emailops/search_and_draft.py:919)** - 100 lines - Context retrieval

### ⚡ **Performance Critical Functions**
- **[`embed_texts()`](emailops/llm_runtime.py:674)** - Batched API calls with retry logic
- **[`_ensure_embeddings_ready()`](emailops/search_and_draft.py:425)** - Memory-mapped file access
- **[`prepare_index_units()`](emailops/text_chunker.py:236)** - Text chunking with boundary detection
- **[`clean_email_text()`](emailops/utils.py:675)** - Regex-heavy text processing

## Security Analysis

### 🔐 **Security-Critical Functions**
- **[`validate_file_path()`](emailops/validators.py:119)** - Path traversal prevention
- **[`_hard_strip_injection()`](emailops/search_and_draft.py:251)** - Prompt injection prevention
- **[`_atomic_write_*`](emailops/email_indexer.py:114)** functions - Atomic file operations
- **[`validate_command_args()`](emailops/validators.py:207)** - Command injection prevention

### 🛡️ **Input Validation Chain**
```
User Input → validators.py → utils.py → core processing
           ↑                ↑         ↑
    Path validation    Text sanitization    Business logic
```

## Recommendations

### 🚀 **Performance Optimizations**
1. **Cache [`get_config()`](emailops/config.py:188)** results to reduce environment variable access
2. **Pool [`embed_texts()`](emailops/llm_runtime.py:674)** calls to reduce API overhead
3. **Optimize [`clean_email_text()`](emailops/utils.py:675)** regex compilation

### 🔧 **Maintainability Improvements**  
1. **Consolidate validation** - Multiple path validation functions could be unified
2. **Reduce function complexity** - Break down 200+ line functions
3. **Standardize error handling** - Inconsistent exception patterns across modules

### 🛡️ **Security Hardening**
1. **Input sanitization** at module boundaries
2. **Credential validation** before file operations
3. **Rate limiting** for all external API calls

---

**Generated:** 2025-01-14 02:03 UTC  
**Status:** Complete - All 13 modules analyzed  
**Critical Import Bug:** ✅ Fixed in [`summarize_email_thread.py:14-19`](emailops/summarize_email_thread.py:14-19)