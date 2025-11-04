# `emailops.util_processing`

**Primary Goal:** To provide a collection of shared, high-level utilities that support and optimize the application's data processing workflows. This module contains tools for batch processing, performance monitoring, and centralized text cleaning.

## Directory Mapping

```
.
└── emailops/
    └── util_processing.py
```

---

## Core Components & Connections

### Configuration Management

- **`ProcessingConfig` & `get_processing_config()`:**
    - **Purpose:** This dataclass centralizes configuration settings specifically related to document processing, such as character limits for attachments and cell limits for Excel files. The `get_processing_config()` function provides a singleton instance of this configuration.
    - **Connections:** This config is used by `core_conversation_loader` to determine whether to skip large attachments or truncate extracted text, ensuring that processing limits are applied consistently at the point of data loading.

### Performance Monitoring

- **`@monitor_performance` Decorator:**
    - **Purpose:** A simple decorator to measure and log the execution time of a function.
    - **Functionality:** It wraps a function call, records the start and end times, and logs a warning if the execution time exceeds a certain threshold (e.g., 1 second). It also logs any exceptions that occur.
    - **Connections:** This decorator is used on key functions, such as `TextPreprocessor.prepare_for_indexing`, to help developers identify performance bottlenecks in the data processing pipeline.

### Batch Processing

- **`BatchProcessor` Class:**
    - **Purpose:** Provides a generic and reusable mechanism for processing a large list of items in smaller, parallel batches.
    - **Functionality:** The `process_items` method takes a list of items and a `processor` function. It uses a `ThreadPoolExecutor` to apply the `processor` function to the items in parallel, respecting the configured `batch_size` and `max_workers`. It includes error handling to catch exceptions from individual items without halting the entire batch.
    - **Connections:** While not explicitly used in the other core modules we've reviewed, this is a foundational utility that would be invaluable for tasks like bulk re-indexing, running batch analysis jobs, or any operation that involves applying the same function to thousands of documents or records.

### PII Redaction

- **`redact_pii(text)`:**
    - **Purpose:** A utility function that redacts common PII like email addresses and phone numbers from a given string.
    - **Functionality:** It uses regular expressions to find and replace PII with placeholders like `[email redacted]` and `[phone redacted]`.
    - **Connections:** This function is called by the `TextPreprocessor` to sanitize text before indexing.

### Centralized Text Preprocessing

- **`TextPreprocessor` Class & `get_text_preprocessor()`:**
    - **Purpose:** This is arguably the most critical component in this module. It provides a **single, canonical place** for all text cleaning and preparation before indexing.
    - **Functionality:**
        1.  **`prepare_for_indexing(text, ...)`:** This is the main method. It takes raw text and a `text_type` (e.g., 'email', 'attachment').
        2.  **Type-Specific Cleaning:** Based on the `text_type`, it applies different cleaning logic. It uses the aggressive `clean_email_text` for emails but applies a lighter touch (`_clean_attachment_text`) for attachments, which might contain structured text like code or logs that should be preserved.
        3.  **Metadata Generation:** It returns not just the cleaned text but also a dictionary of metadata, including a `cleaning_version` and a `pre_cleaned: True` flag. This metadata is stored in the index.
        4.  **Caching:** It maintains an internal cache to avoid re-processing identical inputs, providing a significant performance boost.
- **`should_skip_retrieval_cleaning(chunk_or_doc)`:**
    - **Purpose:** This is the counterpart to the preprocessor. It's a helper function used during the *retrieval* phase.
    - **Functionality:** It checks a document's metadata for the `pre_cleaned: True` flag and a compatible `cleaning_version`.
    - **Connections:** This function is a key optimization. It is called by `feature_search_draft._gather_context_for_conv` before assembling the context for the LLM. By checking this flag, the system knows that the text from the index is already clean and can **skip** running `clean_email_text` on it again, which provides a major performance improvement (the docstring claims 40-60%).

---

## Key Design Patterns

- **Singleton:** The `get_processing_config()` and `get_text_preprocessor()` functions use the singleton pattern to provide a single, shared instance of their respective classes throughout the application.
- **Centralized Preprocessing:** The `TextPreprocessor` embodies the principle of "Don't Repeat Yourself" (DRY). By cleaning text *once* before indexing and storing a flag to indicate this, it eliminates redundant and expensive cleaning operations that would otherwise have to happen every time a document is retrieved for a search. This is a powerful architectural pattern for optimizing RAG pipelines.
- **Strategy Pattern:** The `prepare_for_indexing` method uses a simple form of the Strategy pattern. Based on the `text_type`, it chooses a different cleaning strategy (`clean_email_text`, `_clean_attachment_text`, etc.), applying the most appropriate logic for the given content.