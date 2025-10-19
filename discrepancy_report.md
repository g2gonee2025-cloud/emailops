# Discrepancy Report: emailops Front-end and Back-end Alignment

This document tracks discrepancies found between the `emailops` back-end functionality and the `emailops_gui.py` front-end implementation.

## Analysis Checklist
- [x] `emailops/llm_client.py`
- [x] `emailops/config.py`
- [x] `emailops/index_metadata.py`
- [x] `emailops/validators.py`
- [x] `emailops/env_utils.py`
- [x] `emailops/email_indexer.py`
- [x] `emailops/conversation_loader.py`
- [x] `emailops/parallel_indexer.py`
- [x] `emailops/processing_utils.py`
- [x] `emailops/file_utils.py`
- [x] `emailops/summarize_email_thread.py`
- [x] `emailops/exceptions.py`
- [x] `emailops/text_chunker.py`
- [x] `emailops/utils.py`
- [x] `emailops/search_and_draft.py`
- [x] `emailops/processor.py`
- [x] `emailops/llm_runtime.py`
- [x] `emailops/email_processing.py`
- [x] `emailops/text_extraction.py`
- [x] `emailops/doctor.py`
- [x] `emailops/emailops_gui.py`

## Discrepancies Found

### `emailops/llm_client.py`

1.  **Configuration Mismatch:** The `ClientConfig` in `llm_client.py` provides extensive configuration options (timeouts, retries, rate limiting, circuit breaker, caching) that are not exposed in the GUI's "Configuration" tab. The user has no control over these important runtime parameters from the front-end.
2.  **Inconsistent LLM Interaction:** The module offers two distinct ways to interact with LLMs: the generic `LLMClient` class and the hardcoded Vertex AI convenience functions (`complete_text`, `complete_json`, `embed_texts`). The application primarily uses the convenience functions, which limits flexibility and makes it harder to switch providers. This should be unified to use the `LLMClient` exclusively.
3.  **`LLMClient` Not Directly Used:** The GUI does not directly instantiate or use the `LLMClient` class. While this is acceptable as an abstraction, it's important to ensure that the underlying client is being configured and used correctly by the intermediary modules.

### `emailops/config.py`

1.  **Missing GUI Configuration:** The `EmailOpsConfig` class in the backend defines `ALLOWED_FILE_PATTERNS`, which controls the types of attachments that are processed. This important setting is not exposed in the GUI, preventing users from customizing it.
2.  **Dual Configuration Systems:** The application uses two separate configuration classes: `AppSettings` for the GUI (persisted to JSON) and `EmailOpsConfig` for the backend (loaded from environment variables). While the GUI correctly sets environment variables to align them, this dual system is redundant and could lead to inconsistencies. A single, unified configuration model should be adopted.

### `emailops/index_metadata.py`

1.  **Missing Time Decay Configuration:** The `TimeDecayConfig` class allows for configuring the `half_life_days` for search result weighting. This is a powerful tuning parameter that is not exposed in the GUI.
2.  **Incomplete Consistency Checks in GUI:** The `IndexMetadata.validate_consistency()` method is a crucial tool for verifying the integrity of the search index. The GUI's diagnostic tools should explicitly call this method to provide users with the most reliable index health information.

### `emailops/validators.py`

1.  **Underutilized Validation Functions:** The GUI does not make full use of the validation functions available in `validators.py`. For example, it could use `is_email` for email fields and `is_url` for URL fields to provide immediate feedback to the user.
2.  **Inconsistent Path Validation:** The GUI's path validation could be improved by consistently using the `validate_path_is_dir` and `validate_path_is_file` functions to ensure robustness and provide clearer error messages.
3.  **Missing Command-Line Validation:** The `_on_build_index` method in the GUI constructs and executes a command-line process without using the `validate_command_args` function. This is a security risk that should be addressed by validating all command-line arguments before execution.

### `emailops/env_utils.py`

No significant discrepancies found. The GUI appears to correctly use the account management and validation functions provided by this module.

### `emailops/email_indexer.py`

1.  **Missing File Size Configuration:** The `email_indexer.py` module defines `MAX_FILE_SIZE_MB` and `MAX_TEXT_CHARS` to control the size of files that can be indexed. These settings are not exposed in the GUI, which could be problematic for users with very large files.
2.  **Model Override Not Exposed:** The indexer's `_apply_model_override` function allows for using different embedding models, but this flexibility is not available in the GUI, which is hardcoded to use the default Vertex AI model.

### `emailops/conversation_loader.py`

1.  **Attachment Text Strategy Not Exposed:** The `load_conversation` function's `attachment_text_strategy` parameter, which allows for concatenating attachment text to the main content, is not configurable from the GUI.
2.  **Allowed Extensions Not Exposed:** The `allowed_extensions` parameter in `load_conversation` is not exposed in the GUI, preventing users from filtering which attachments are loaded.

### `emailops/parallel_indexer.py`

No significant discrepancies found. The GUI correctly uses the parallel indexing functionality, allowing the user to configure the number of workers.

### `emailops/processing_utils.py`

1.  **Missing Configuration:** The `ProcessingConfig` class defines several important parameters (`max_attachment_chars`, `excel_max_cells`, `skip_attachment_over_mb`) that are not exposed in the GUI.
2.  **Text Preprocessor Not Exposed:** The `TextPreprocessor` class is a key component for cleaning text, but the GUI does not provide any way to configure or interact with it.
3.  **Inconsistent Batch Processing:** The GUI has batch operations, but it's unclear if they use the `BatchProcessor` class. A direct integration would improve consistency and performance.

### `emailops/file_utils.py`

1.  **Underutilized Atomic I/O:** The GUI performs file-saving operations without using the atomic I/O functions (`write_bytes`, `write_text`) from `file_utils.py`. This could lead to corrupted files if the application is interrupted during a write operation.
2.  **Missing File Locking:** The GUI does not use the `FileLock` class to prevent race conditions when accessing files. This could be an issue if the user performs multiple operations on the same file simultaneously.

### `emailops/summarize_email_thread.py`

1.  **Missing Configuration:** The summarization module has several configuration options (`MAX_THREAD_CHARS`, `CRITIC_THREAD_CHARS`, `IMPROVE_THREAD_CHARS`) that are not exposed in the GUI.
2.  **Catalog Not Exposed:** The `analyze_email_thread_with_ledger` function's `catalog` parameter is not configurable from the GUI, preventing users from customizing the classification categories.

### `emailops/exceptions.py`

No significant discrepancies found. The GUI has a basic error handling mechanism, and the custom exceptions in this module are primarily intended for use by the backend modules.

### `emailops/text_chunker.py`

1.  **Missing Configuration:** The `ChunkConfig` class provides several important parameters for controlling the chunking process, such as `target_chunk_chars`, `min_chunk_chars`, `max_chunk_chars`, and `overlap_chars`. While the GUI displays the chunk size and overlap, it does not allow the user to modify them. The other parameters are not exposed at all.
2.  **Progressive Scaling Not Exposed:** The `_apply_progressive_scaling` function is a useful feature for adjusting the chunk size based on the document length, but the user cannot disable it from the GUI.
3.  **Break On Not Exposed:** The `break_on` parameter in `ChunkConfig` is a powerful feature for fine-tuning the chunking process, but it is not exposed in the GUI.

### `emailops/utils.py`

No significant discrepancies found. This module has been refactored to delegate functionality to other modules, and the GUI correctly uses the functions from those specialized modules.

### `emailops/search_and_draft.py`

1.  **Missing Configuration:** The `search_and_draft.py` module has several configuration options (`RECENCY_BOOST_STRENGTH`, `CANDIDATES_MULTIPLIER`, `FORCE_RENORM`) that are not exposed in the GUI.

### `emailops/processor.py`

No significant discrepancies found. The `processor.py` module is a command-line orchestrator and is not directly used by the GUI, which is an acceptable design choice.

### `emailops/llm_runtime.py`

1.  **Missing Configuration:** The `LLMRuntime` class has a rich set of configuration options for timeouts, retries, circuit breakers, and rate limiting. None of these are exposed in the GUI.

### `emailops/email_processing.py`

1.  **Missing Cleaning Options:** The `clean_email_text` function has options for redacting emails and URLs, and for stripping HTML. These options are not exposed in the GUI.
2.  **Thread Splitting Not Exposed:** The `split_email_thread` function is not configurable from the GUI.

### `emailops/text_extraction.py`

No significant discrepancies found. The GUI indirectly uses the `extract_text` function through the `conversation_loader.py` module, which is an acceptable design choice.

### `emailops/doctor.py`

No significant discrepancies found. The GUI's "Diagnostics" tab correctly uses the functions from this module to provide the user with detailed information about the system's health.

## Proposed Plan of Action

1.  **Unify Configuration:**
    *   Create a single, unified configuration model that will be used by both the backend and the GUI. This will eliminate the redundancy of the `AppSettings` and `EmailOpsConfig` classes and provide a single source of truth for all configuration settings.
    *   The new configuration model will be loaded from a file (e.g., `config.json` or `config.toml`) and can be overridden by environment variables.
    *   The GUI will be updated to use this new configuration model, and the "Configuration" tab will be updated to expose all the relevant settings to the user.

2.  **Refactor LLM Interaction:**
    *   Refactor the backend to use the `LLMClient` class exclusively for all LLM interactions. This will involve removing the hardcoded Vertex AI convenience functions and updating all the modules that use them.
    *   This will make the application more flexible and easier to maintain, as it will be possible to switch LLM providers by simply changing the configuration.

3.  **Expose Backend Features in GUI:**
    *   Update the GUI to expose all the backend features that are currently not available to the user. This includes:
        *   `LLMClient` configuration (timeouts, retries, etc.)
        *   `EmailOpsConfig` settings (`ALLOWED_FILE_PATTERNS`)
        *   `TimeDecayConfig` settings (`half_life_days`)
        *   `ProcessingConfig` settings (`max_attachment_chars`, etc.)
        *   `ChunkConfig` settings (`target_chunk_chars`, etc.)
        *   `Search and Draft` settings (`RECENCY_BOOST_STRENGTH`, etc.)
        *   `Summarization` settings (`MAX_THREAD_CHARS`, etc.)
        *   `Email Processing` settings (redaction, etc.)

4.  **Improve GUI Robustness and Security:**
    *   Update the GUI to use the validation functions from `validators.py` to provide immediate feedback to the user.
    *   Update the GUI to use the atomic I/O functions from `file_utils.py` to prevent file corruption.
    *   Update the GUI to use the `FileLock` class from `file_utils.py` to prevent race conditions.
    *   Update the GUI to use the `validate_command_args` function from `validators.py` to prevent command injection vulnerabilities.

5.  **Enhance GUI Diagnostics:**
    *   Update the GUI's "Diagnostics" tab to use the `IndexMetadata.validate_consistency()` method to provide more detailed and reliable index health information.
