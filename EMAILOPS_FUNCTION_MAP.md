# EmailOps Function Dependency Map & Flow Visualization
**Generated:** 2025-10-15  
**Purpose:** Visual reference for code navigation and understanding system architecture

---

## 🗺️ Complete Module Dependency Graph

```
                           ┌─────────────────┐
                           │   config.py     │
                           │  (Foundation)   │
                           │                 │
                           │ • EmailOpsConfig│
                           │ • get_config()  │
                           └────────┬────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  ↓                 ↓                 ↓
         ┌────────────────┐ ┌──────────────┐ ┌─────────────┐
         │ exceptions.py  │ │ file_utils.py│ │validators.py│
         │  (Error Types) │ │  (File I/O)  │ │ (Security)  │
         └────────┬───────┘ └──────┬───────┘ └──────┬──────┘
                  │                │                 │
                  └────────────────┼─────────────────┘
                                   ↓
                      ┌────────────────────────┐
                      │    llm_runtime.py      │
                      │   (LLM Operations)     │
                      │                        │
                      │ • complete_text()      │
                      │ • complete_json()      │
                      │ • embed_texts()        │
                      │ • _embed_vertex()      │
                      │ • Project rotation     │
                      │ • Rate limiting        │
                      └───────────┬────────────┘
                                  │
                      ┌───────────┴────────────┐
                      ↓                        ↓
            ┌─────────────────┐      ┌─────────────────┐
            │ llm_client.py   │      │ processing_     │
            │ (Compat Shim)   │      │ utils.py        │
            └────────┬────────┘      └────────┬────────┘
                     │                        │
        ┌────────────┼────────────────────────┼────────────┐
        ↓            ↓            ↓           ↓            ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│email_        │ │text_         │ │conversation_ │ │email_        │
│processing.py │ │extraction.py │ │loader.py     │ │indexer.py    │
│              │ │              │ │              │ │              │
│• clean_email │ │• extract_text│ │• load_       │ │• build_corpus│
│  _text()     │ │• _extract_pdf│ │  conversation│ │• save_index()│
│• extract_    │ │• _extract_   │ └──────────────┘ └──────────────┘
│  email_      │ │  docx        │
│  metadata()  │ │• _extract_   │
│• split_email │ │  excel       │
│  _thread()   │ │• _extract_msg│
└──────────────┘ └──────────────┘
        │                │
        └────────┬───────┘
                 ↓
        ┌──────────────────┐
        │ text_chunker.py  │
        │                  │
        │• prepare_index_  │
        │  units()         │
        │• TextChunker     │
        │• ChunkConfig     │
        └────────┬─────────┘
                 │
    ┌────────────┼────────────┐
    ↓            ↓            ↓
┌──────────┐ ┌──────────┐ ┌────────────────┐
│index_    │ │search_   │ │summarize_email_│
│metadata  │ │and_draft │ │thread.py       │
│.py       │ │.py       │ │                │
│          │ │          │ │• analyze_      │
│• read_   │ │• _search │ │  conversation_ │
│  mapping │ │• draft_  │ │  dir()         │
│• write_  │ │  email_  │ │• format_       │
│  mapping │ │  reply_  │ │  analysis_as_  │
│• validate│ │  eml()   │ │  markdown()    │
│  _index_ │ │• draft_  │ └────────────────┘
│  compat  │ │  fresh_  │
└──────────┘ │  email_  │
             │  eml()   │
             │• chat_   │
             │  with_   │
             │  context │
             └────┬─────┘
                  │
    ┌─────────────┼─────────────┐
    ↓             ↓             ↓
┌──────────┐ ┌──────────┐ ┌──────────────┐
│processor │ │emailops_ │ │parallel_     │
│.py       │ │gui.py    │ │indexer.py    │
│(CLI)     │ │(GUI)     │ │(Multi-worker)│
└──────────┘ └──────────┘ └──────────────┘
```

---

## 🔄 Function Call Flow: Email Indexing

```
USER: Builds Index
    ↓
┌─────────────────────────────────────────┐
│ email_indexer.main()                    │
│   ├─ _initialize_gcp_credentials()      │
│   ├─ build_corpus() OR                  │
│   │   build_incremental_corpus()        │
│   │   ├─ find_conversation_dirs()       │
│   │   │   └─ file_utils.py              │
│   │   ├─ load_conversation()            │
│   │   │   └─ conversation_loader.py     │
│   │   ├─ _extract_manifest_metadata()   │
│   │   └─ _build_doc_entries()           │
│   │       ├─ clean_email_text()         │
│   │       │   └─ email_processing.py    │
│   │       └─ prepare_index_units()      │
│   │           └─ text_chunker.py        │
│   ├─ embed_texts() [BATCH LOOP]         │
│   │   └─ llm_runtime._embed_vertex()    │
│   │       ├─ _check_rate_limit()        │
│   │       ├─ google.genai.Client        │
│   │       └─ _rotate_to_next_project()  │
│   └─ save_index()                       │
│       ├─ write_mapping()                │
│       │   └─ index_metadata.py          │
│       ├─ np.save(embeddings.npy)        │
│       └─ faiss.write_index()            │
└─────────────────────────────────────────┘
    ↓
OUTPUT: FAISS index + mapping.json + embeddings.npy
```

---

## 🔍 Function Call Flow: Email Search

```
USER: Enters search query
    ↓
┌─────────────────────────────────────────┐
│ search_and_draft._search()              │
│   ├─ validate_index_compatibility()     │
│   │   └─ index_metadata.py              │
│   ├─ read_mapping()                     │
│   │   └─ index_metadata.py              │
│   ├─ parse_filter_grammar()             │
│   │   └─ Build SearchFilters object     │
│   ├─ apply_filters()                    │
│   │   └─ Filter by metadata             │
│   ├─ embed_texts([query])               │
│   │   └─ _get_cached_query_embedding()  │
│   │       OR _embed_vertex()            │
│   ├─ Cosine similarity: embs @ query.T  │
│   ├─ _boost_scores_for_indices()        │
│   │   └─ Recency boost calculation      │
│   ├─ Summary-aware reranking:           │
│   │   ├─ _candidate_summary_text()      │
│   │   ├─ embed_texts([summaries])       │
│   │   └─ _blend_scores()                │
│   ├─ _mmr_select()                      │
│   │   └─ Diversity optimization         │
│   ├─ _deduplicate_chunks()              │
│   │   └─ By content_hash                │
│   └─ Read and window text               │
│       └─ _window_text_around_query()    │
└─────────────────────────────────────────┘
    ↓
OUTPUT: Ranked, deduplicated search results
```

---

## ✉️ Function Call Flow: Email Drafting (Reply)

```
USER: Drafts reply to conversation
    ↓
┌─────────────────────────────────────────┐
│ search_and_draft.draft_email_reply_eml()│
│   ├─ _load_conv_data()                  │
│   │   ├─ Read Conversation.txt          │
│   │   ├─ Read manifest.json             │
│   │   └─ _extract_messages_from_        │
│   │       manifest()                    │
│   ├─ _derive_query_from_last_inbound()  │
│   │   └─ Extract last email content     │
│   ├─ _gather_context_for_conv()         │
│   │   └─ Same as search flow above      │
│   ├─ draft_email_structured()           │
│   │   ├─ PASS 1: Initial Draft          │
│   │   │   └─ complete_json()            │
│   │   │       └─ llm_runtime.py         │
│   │   ├─ PASS 2: Critic Review          │
│   │   │   └─ complete_json()            │
│   │   └─ PASS 3: Audit Loop (max 5x)    │
│   │       ├─ _audit_json()              │
│   │       │   └─ complete_json()        │
│   │       └─ complete_text()            │
│   │           [if improvements needed]  │
│   ├─ _select_attachments_from_citations│
│   │   OR _select_attachments_from_      │
│   │      mentions()                     │
│   ├─ calculate_draft_confidence()       │
│   ├─ _derive_recipients_for_reply()     │
│   ├─ _derive_subject_for_reply()        │
│   └─ _build_eml()                       │
│       └─ Create RFC-822 .eml file       │
└─────────────────────────────────────────┘
    ↓
OUTPUT: .eml file with headers, body, attachments
```

---

## 💬 Function Call Flow: Chat

```
USER: Asks question
    ↓
┌─────────────────────────────────────────┐
│ search_and_draft.chat_with_context()    │
│   ├─ _search()                          │
│   │   └─ Retrieve relevant context      │
│   ├─ _format_chat_history_for_prompt()  │
│   │   └─ Format previous messages       │
│   ├─ complete_json()                    │
│   │   └─ Generate answer with schema    │
│   └─ ChatSession.save()                 │
│       └─ Persist history to JSON        │
└─────────────────────────────────────────┘
    ↓
OUTPUT: Answer with citations & missing info
```

---

## 📊 Function Call Flow: Thread Analysis

```
USER: Analyzes conversation thread
    ↓
┌──────────────────────────────────────────────┐
│ summarize_email_thread.                      │
│   analyze_conversation_dir()                 │
│   ├─ Read Conversation.txt                   │
│   │   └─ read_text_file()                    │
│   ├─ clean_email_text()                      │
│   ├─ analyze_email_thread_with_ledger()      │
│   │   ├─ PASS 1: Initial Analysis            │
│   │   │   └─ complete_json()                 │
│   │   │       [structured output]            │
│   │   ├─ _normalize_analysis()               │
│   │   │   ├─ _coerce_enum()                  │
│   │   │   ├─ _normalize_subject_line()       │
│   │   │   └─ Schema enforcement              │
│   │   ├─ PASS 2: Critic Review               │
│   │   │   └─ complete_json()                 │
│   │   │       [check completeness]           │
│   │   └─ PASS 3: Improvement (if needed)     │
│   │       └─ complete_json()                 │
│   │           [enhance analysis]             │
│   ├─ _merge_manifest_into_analysis()         │
│   │   ├─ _read_manifest()                    │
│   │   ├─ _participants_from_manifest()       │
│   │   └─ Union merge of data                 │
│   └─ _normalize_analysis() [final pass]      │
└──────────────────────────────────────────────┘
    ↓
OUTPUT: summary.json + summary.md
```

---

## 🏗️ Detailed Module Breakdown

### 1. Configuration Layer

#### `config.py` - Central Configuration
```python
EmailOpsConfig
├─ load() → classmethod, loads from env
├─ get_secrets_dir() → resolve secrets path
├─ _is_valid_service_account_json() → validate GCP keys
├─ get_credential_file() → find valid credentials
├─ update_environment() → sync env vars
└─ to_dict() → export as dict

get_config() → singleton accessor
reset_config() → testing helper
```

**Dependencies:** None (foundation module)  
**Dependent Modules:** ALL (15+)

---

### 2. LLM Runtime Layer

#### `llm_runtime.py` - LLM Operations Hub
```python
# Account Management
VertexAccount → dataclass for GCP accounts
load_validated_accounts() → load & validate accounts
save_validated_accounts() → persist account list
validate_account() → quick validation
_init_vertex() → initialize Vertex AI SDK
reset_vertex_init() → reset init state

# Project Rotation
_ensure_projects_loaded() → lazy load projects
_rotate_to_next_project() → rotate on quota exhaustion

# Rate Limiting
_check_rate_limit() → enforce API limits

# Text Generation
complete_text() → @monitor_performance
    ├─ _init_vertex()
    ├─ _vertex_model()
    ├─ _check_rate_limit()
    ├─ model.generate_content()
    └─ Retry with rotation on errors

complete_json() → @monitor_performance
    ├─ Same as complete_text()
    ├─ response_mime_type: "application/json"
    └─ Fallback to text mode + _extract_json_from_text()

# Embeddings
embed_texts() → @monitor_performance
    ├─ Provider routing
    ├─ _embed_vertex() → primary
    │   ├─ google.genai.Client (preferred)
    │   └─ TextEmbeddingModel (legacy)
    ├─ _embed_openai()
    ├─ _embed_azure_openai()
    ├─ _embed_cohere()
    ├─ _embed_huggingface()
    ├─ _embed_qwen()
    └─ _embed_local()

# Utilities
_normalize() → unit normalize vectors
_is_retryable_error() → classify errors
_should_rotate_on() → rotation heuristics
_sleep_with_backoff() → exponential backoff
_extract_json_from_text() → JSON extraction
_find_complete_json_structure() → bracket counting
_is_balanced_json() → validation
_validate_json_syntax() → syntax check
```

**Key Features:**
- Thread-safe rate limiting
- Automatic project rotation on quota exhaustion
- Multi-provider support (7 providers)
- Robust JSON extraction with fallbacks
- Performance monitoring via decorators

---

### 3. Indexing Layer

#### `email_indexer.py` - Vector Index Builder
```python
# Main Entry
main() → CLI entry point
    ├─ _initialize_gcp_credentials()
    ├─ Parallel vs Serial decision
    ├─ build_corpus() OR build_incremental_corpus()
    └─ save_index()

# Corpus Building
build_corpus(root, index_dir, last_run_time?, limit?) 
    → (new_docs, unchanged_docs)
    ├─ find_conversation_dirs()
    ├─ load_conversation()
    ├─ _extract_manifest_metadata()
    ├─ _build_doc_entries()
    │   ├─ clean_email_text()
    │   ├─ prepare_index_units() → chunking
    │   ├─ _iter_attachment_files()
    │   └─ _att_id() → stable attachment IDs
    └─ Timestamp-based change detection

build_incremental_corpus(root, file_times, mapping, limit?)
    → (new_docs, deleted_ids)
    ├─ Precise file-level change tracking
    ├─ Handles deletions correctly
    └─ Per-conversation limit enforcement

# Index Persistence
save_index(index_dir, embeddings, mapping, provider, num_folders)
    ├─ _atomic_write_bytes(embeddings.npy)
    ├─ write_mapping(mapping.json)
    ├─ faiss.write_index(index.faiss) [optional]
    ├─ save_index_metadata(meta.json)
    └─ check_index_consistency() [post-save]

load_existing_index(index_dir)
    → (faiss_index, mapping, file_times, embeddings)

# Utilities
_atomic_write_bytes() → safe binary write
_atomic_write_text() → safe text write
_prefix_from_id() → normalize doc IDs
_att_id() → generate stable attachment ID
_clean_index_text() → light cleaning for embeddings
_materialize_text_for_docs() → ensure text field
_get_last_run_time() → read timestamp
_save_run_time() → write timestamp
_local_check_index_consistency() → fallback checker
```

**Key Features:**
- Incremental indexing with file-level change tracking
- Atomic writes prevent corruption
- Parallel indexing support (via `parallel_indexer.py`)
- FAISS + NumPy dual storage
- Per-conversation doc limits

---

#### `parallel_indexer.py` - Multi-Worker Indexing
```python
WorkerBatch → dataclass for worker config

parallel_index_conversations(root, index_dir, num_workers, ...)
    → (merged_embeddings, merged_mapping)
    ├─ Split conversations across workers
    ├─ Assign GCP accounts round-robin
    ├─ _index_worker() [in parallel]
    │   ├─ Set GCP credentials for worker
    │   ├─ Chunk all assigned conversations
    │   ├─ Embed all chunks
    │   └─ Save partial results
    ├─ Merge results (deterministic order)
    └─ Cleanup temp files

_index_worker(batch: WorkerBatch) → worker_result
    [Runs in separate process]
```

**Key Features:**
- Process pool with 'spawn' start method (Windows-safe)
- GCP account per worker (parallel quota)
- Deterministic result merging
- Comprehensive cleanup

---

### 4. Search & Retrieval Layer

#### `search_and_draft.py` - Search, Draft, Chat
```python
# Search Core
_search(ix_dir, query, k, provider, filters?, mmr_lambda?, rerank_alpha?)
    → ranked results
    ├─ validate_index_compatibility()
    ├─ parse_filter_grammar() → extract fielded filters
    ├─ apply_filters() → pre-embedding filter
    ├─ embed_texts([query]) → with caching
    ├─ scores = embs @ query.T → cosine similarity
    ├─ _boost_scores_for_indices() → recency boost
    ├─ Early deduplication by content_hash
    ├─ Summary-aware reranking:
    │   ├─ _candidate_summary_text()
    │   ├─ embed_texts([summaries])
    │   └─ _blend_scores()
    ├─ _mmr_select() → diversity via MMR
    └─ _deduplicate_chunks() → final dedup

# Context Gathering
_gather_context_for_conv(ix_dir, conv_id, query, ...)
    → context snippets
    [Same pipeline as _search but filtered to conv_id]

_gather_context_fresh(ix_dir, query, ...)
    → context snippets
    [Same pipeline as _search but no conv_id filter]

# Email Drafting
draft_email_structured(query, sender, context, ...)
    → draft_result
    ├─ validate_context_quality()
    ├─ PASS 1: Initial Draft
    │   └─ complete_json() with schema
    │       [citations, missing_info, assumptions]
    ├─ PASS 2: Critic Review
    │   └─ complete_json() with critic schema
    │       [issues_found, improvements_needed]
    ├─ PASS 3: Audit Loop (up to 5x)
    │   ├─ _audit_json() → score on rubric
    │   └─ complete_text() → improve if needed
    ├─ Attachment Selection:
    │   ├─ _select_attachments_from_mentions()
    │   ├─ _select_attachments_from_citations()
    │   └─ select_relevant_attachments() [fallback]
    └─ calculate_draft_confidence()

draft_email_reply_eml(export_root, conv_id, ...)
    → {eml_bytes, draft_json, ...}
    ├─ _load_conv_data()
    ├─ _derive_query_from_last_inbound()
    ├─ _gather_context_for_conv()
    ├─ draft_email_structured()
    ├─ _derive_recipients_for_reply()
    ├─ _derive_subject_for_reply()
    └─ _build_eml()

draft_fresh_email_eml(export_root, to_list, subject, query, ...)
    → {eml_bytes, draft_json, ...}
    ├─ parse_filter_grammar()
    ├─ _gather_context_fresh()
    ├─ draft_email_structured()
    └─ _build_eml()

# EML Construction
_build_eml(from, to, cc, subject, body, attachments?, ...)
    → bytes
    ├─ Create EmailMessage
    ├─ Set headers (From, To, Cc, Subject, Date, Message-ID)
    ├─ Set threading headers (In-Reply-To, References)
    ├─ Set text/plain body
    ├─ Add text/html alternative
    └─ Attach files (with size validation)

# Chat
chat_with_context(query, context, chat_history?, temp?)
    → {answer, citations, missing_info}
    ├─ _format_chat_history_for_prompt()
    ├─ complete_json() with chat schema
    └─ Fallback to complete_text() + parse

ChatSession → persistent chat
├─ load() → from JSON
├─ save() → to JSON
├─ reset() → clear history
├─ add_message(role, content)
└─ recent() → get history

# Filters & Utilities
SearchFilters → dataclass for filters
parse_filter_grammar(query) → (filters, cleaned_query)
    Supports: subject:, from:, to:, cc:, after:, before:, 
              has:attachment, type:pdf, -exclusion

apply_filters(mapping, filters) → filtered indices
validate_context_quality() → check context adequacy
select_relevant_attachments() → heuristic selection
list_conversations_newest_first() → conversation list

# Helper Functions (30+)
_embed_query_compatible() → dimension-safe embedding
_sim_scores_for_indices() → cosine similarity
_boost_scores_for_indices() → recency boost
_mmr_select() → MMR diversity
_blend_scores() → rerank blending
_deduplicate_chunks() → by content_hash
_window_text_around_query() → smart windowing
_bidirectional_expand_text() → expand from center
_sanitize_header_value() → header safety
_clean_addr() → address cleaning
_dedupe_keep_order() → unique preserving order
... and 20 more
```

**Key Features:**
- Three-pass drafting (draft → critic → audit)
- MMR for diversity
- Summary-aware reranking
- Early deduplication before expensive operations
- Query caching (5min TTL)
- Mapping cache with mtime invalidation
- Grammar-based filter parsing
- Attachment selection strategies

---

### 5. Analysis Layer

#### `summarize_email_thread.py` - Thread Analysis
```python
# Main API
analyze_conversation_dir(thread_dir, catalog?, provider?, temp?, merge_manifest?)
    → analysis_dict (async)
    ├─ read_text_file()
    ├─ clean_email_text()
    ├─ analyze_email_thread_with_ledger()
    └─ _merge_manifest_into_analysis()

analyze_email_thread_with_ledger(thread_text, catalog, provider, temp)
    → analysis_dict (async)
    ├─ PASS 1: Initial Analysis
    │   ├─ complete_json() with full schema
    │   ├─ _try_load_json() → robust parsing
    │   └─ _normalize_analysis()
    ├─ PASS 2: Critic Review
    │   ├─ complete_json() with critic schema
    │   └─ Check completeness_score
    └─ PASS 3: Improvement Loop (if score < 85)
        ├─ complete_json() with improvement prompt
        ├─ _normalize_analysis()
        └─ _union_analyses() → merge without data loss

format_analysis_as_markdown(analysis) → markdown_str
    [Formats all sections: summary, participants, facts ledger, etc.]

# JSON Parsing (Robust)
_try_load_json(data) → dict
    ├─ Strategy 1: Direct json.loads()
    ├─ Strategy 2: Extract from ```json fence
    └─ Strategy 3: _extract_first_balanced_json_object()

_extract_first_balanced_json_object(s) → json_str?
    [Bracket counting with string literal handling]

# Normalization & Validation
_normalize_analysis(data, catalog) → dict
    ├─ Schema enforcement
    ├─ _coerce_enum() → standardize enums
    ├─ _normalize_subject_line() → clean subject
    ├─ _normalize_name() → clean names
    ├─ Apply size caps (MAX_PARTICIPANTS, etc.)
    └─ De-duplication

_coerce_enum(val, allowed, default, synonyms?) → str
    [Map variants to canonical values]

# Manifest Integration
_read_manifest(convo_dir) → manifest_dict
    [BOM-tolerant, control char stripping]

_participants_from_manifest(manifest) → participant_list
    [Extract from first message]

_merge_manifest_into_analysis(analysis, convo_dir, raw_text)
    → enriched_analysis
    ├─ Union merge participants
    ├─ Add start/end dates
    └─ Preserve existing data

_union_analyses(improved, initial, catalog) → merged_dict
    [Union merge to prevent data loss]

# File Operations
_atomic_write_text(path, content) → None
    [Temp file + os.replace with retries]

_append_todos_csv(root, thread_name, todos) → None
    ├─ De-duplication by (owner, action, thread)
    └─ DictWriter for safety

# Utilities
_safe_str(v, max_len) → str
_md_escape(v) → str [markdown escaping]
_normalize_name(n) → str
_normalize_subject_line(s) → str
_safe_csv_cell(x) → str [injection prevention]
_calc_max_output_tokens() → int [dynamic budget]
_llm_routing_kwargs(provider) → dict
_retry(callable, retries?, delay?) → result (async helper)
```

**Schema:** 8-field facts ledger (known_facts, key_dates, commitments, etc.)  
**Workflow:** 3-pass analysis with union merging

---

### 6. Utility Modules

#### `text_chunker.py` - Text Splitting
```python
ChunkConfig → dataclass (chunk_size, overlap, etc.)

TextChunker(config)
└─ chunk_text(text, metadata?) → chunk_list

prepare_index_units(text, doc_id, doc_path, ...)
    → chunk_list [for indexing]
    ├─ _apply_progressive_scaling() → adaptive sizing
    ├─ _ranges_with_overlap()
    │   ├─ _compute_breakpoints() → sentence/para boundaries
    │   └─ Forward progress guarantee
    └─ Generate chunk IDs: doc_id, doc_id::chunk1, ...

# Internal
_apply_progressive_scaling() → (size, overlap)
    [Scale up for large docs]

_compute_breakpoints(text, respect_sentences?, respect_paragraphs?)
    → breakpoint_list
    [PARA_RE, SENT_RE patterns]

_ranges_with_overlap(text, size, overlap, ...)
    → [(start, end), ...]
    [Boundary-aware splitting]
```

**Key Features:**
- Boundary-aware (sentence, paragraph)
- Progressive scaling for large docs
- Guaranteed forward progress
- Tiny tail merging

---

#### `text_extraction.py` - File Format Handling
```python
extract_text(path, max_chars?, use_cache?) → str
    ├─ Cache check (1-hour TTL)
    ├─ Format routing:
    │   ├─ TEXT: read_text_file()
    │   ├─ PDF: _extract_pdf()
    │   ├─ DOCX: _extract_word_document()
    │   ├─ DOC: _extract_text_from_doc_win32() [Windows]
    │   ├─ XLSX: _extract_excel()
    │