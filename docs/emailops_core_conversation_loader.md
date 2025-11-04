# `emailops.core_conversation_loader`

**Primary Goal:** To load, consolidate, and preprocess all data associated with a single email conversation from the file system. This module is a critical first step in the data ingestion pipeline, preparing a complete "conversation object" for further processing, indexing, and analysis.

## Directory Mapping

```apache
.
└── emailops/
    └── core_conversation_loader.py
```

---

## Core Functions & Connections

This module is responsible for finding, reading, and aggregating multiple files that constitute a single conversation.

### `load_conversation(...)`

- **Purpose:** This is the main public function of the module. It takes a path to a conversation directory and orchestrates the loading of all its constituent parts: the main conversation text, attachments, and metadata.
- **Parameters:**
  - `convo_dir`: The path to the directory representing a single conversation.
  - `include_attachment_text`: A boolean flag to determine if the extracted text from attachments should be appended to the main conversation text.
  - Various `max_*` and `skip_*` parameters to control resource usage (e.g., skip large attachments, limit text extraction).
- **Return Value:** A dictionary containing the complete conversation data (`path`, `conversation_txt`, `attachments`, `summary`, `manifest`) or `None` if the directory is invalid or contains no usable content.
- **Connections:**
  - **Calls `_load_conversation_text()`:** To read the primary `Conversation.txt` file.
  - **Calls `_process_attachments()`:** To handle the discovery and text extraction of all associated attachments.
  - **Calls `core_manifest.load_manifest()`:** To load structured metadata from `manifest.json`.
  - **Calls `_load_summary()`:** To load any pre-computed analysis from `summary.json`.
  - **Uses `util_processing.get_processing_config()`:** To fetch default values for processing limits if they are not provided explicitly.

---

## Internal Helper Functions

These functions are internal to the module and support the main `load_conversation` function.

### `_load_conversation_text(convo_dir)`

- **Purpose:** Safely reads the `Conversation.txt` file from a given directory.
- **Connections:**
  - Uses `utils.read_text_file()`, which likely contains robust logic for handling different file encodings and sanitizing content (e.g., stripping control characters).

### `_process_attachments(...)`

- **Purpose:** Orchestrates the processing of all attachments within a conversation directory.
- **Functionality:**
  1. Calls `_collect_attachment_files()` to get a deterministic list of all attachment files.
  2. Iterates through each file and calls `_process_single_attachment()` on it.
  3. If `include_attachment_text` is true, it aggregates the extracted text from attachments and appends it to the main conversation body, respecting the `max_total_attachment_text` limit.
- **Connections:** This function is a key part of the data aggregation process, ensuring that attachment content is available for downstream search and analysis.

### `_process_single_attachment(...)`

- **Purpose:** Handles the logic for a single attachment file.
- **Functionality:**
  1. Checks if the file size exceeds the `skip_if_attachment_over_mb` threshold and skips it if necessary.
  2. Calls `core_text_extraction.extract_text()` to perform the actual text extraction, respecting the `max_attachment_text_chars` limit.
- **Connections:**
  - **Delegates to `core_text_extraction`:** This is a crucial connection. `core_conversation_loader` does not know *how* to extract text; it only knows that it *needs* text. It delegates the complex task of handling different file types (PDFs, Word documents, images with OCR) to the specialized `core_text_extraction` module.

### `_collect_attachment_files(convo_dir)`

- **Purpose:** Efficiently and deterministically finds all files that should be treated as attachments.
- **Functionality:**
  1. It scans the root of the `convo_dir` and the `Attachments` subdirectory.
  2. It uses a `set` to automatically handle duplicates.
  3. It explicitly excludes metadata files like `manifest.json` and `Conversation.txt`.
  4. Finally, it sorts the collected files to ensure that the processing order is always the same, which is critical for reproducible results.

---

## Key Design Patterns

- **Facade Pattern:** The `load_conversation` function acts as a simple facade for a complex process that involves reading multiple files, extracting text from various formats, and handling potential errors. A caller only needs to provide a path and gets a complete, ready-to-use data structure in return.
- **Separation of Concerns:** This module is only concerned with *loading* data. The actual parsing of email formats or extraction of text from a PDF is handled by other modules (`core_email_processing`, `core_text_extraction`). This makes the system easier to maintain and extend.
- **Configuration-Driven:** The function's behavior (e.g., attachment size limits) is controlled by parameters that default to values from the central configuration (`get_processing_config()`), making the system flexible without requiring every call site to specify all options.
- **Robustness:** The module is filled with `try...except` blocks to handle `OSError`, `json.JSONDecodeError`, and other potential issues gracefully. It logs warnings and returns `None` or empty data structures instead of crashing, making the data pipeline more resilient.
