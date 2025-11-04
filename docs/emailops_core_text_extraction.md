# `emailops.core_text_extraction`

**Primary Goal:** To act as a centralized and robust service for extracting raw text content from a wide variety of file formats. This module abstracts the complexity of handling different file types, providing a single, consistent interface to the rest of the application.

## Directory Mapping

```
.
└── emailops/
    └── core_text_extraction.py
```

---

## Core Functions & Connections

This module is designed as a service that encapsulates the messy details of file parsing. It is a cornerstone of the data ingestion pipeline.

### `extract_text(path, ...)`

- **Purpose:** This is the primary public function and the main entry point into the module. It acts as a **Facade**, hiding the complexity of the underlying extraction logic. It takes a file path, determines the file type from its extension, and delegates the extraction task to the appropriate specialized helper function.
- **Key Features:**
    - **File Type Dispatching:** It uses the file's suffix (e.g., `.pdf`, `.docx`) to decide which internal function (`_extract_pdf`, `_extract_word_document`) to call.
    - **Caching:** It implements an in-memory, thread-safe cache (`_extraction_cache`). Before processing a file, it checks if the result for that exact file path and `max_chars` limit is already in the cache and still valid (based on a TTL). This is a critical performance optimization that prevents the system from re-running expensive extraction operations on the same file multiple times.
    - **Configuration-Driven:** It respects the `allowed_file_patterns` from the central configuration, ensuring it doesn't attempt to process unauthorized file types.
    - **Graceful Failure:** If a file format is unsupported or an error occurs during extraction, it logs the issue and returns an empty string, preventing a single problematic file from crashing the entire ingestion pipeline.
- **Connections:** This function is heavily used by `emailops.core_conversation_loader._process_single_attachment` to get the text content of every attachment, which is then made available for indexing and analysis.

### `extract_text_async(path, ...)`

- **Purpose:** Provides an asynchronous wrapper around the synchronous `extract_text` function.
- **Implementation:** It uses `asyncio.get_event_loop().run_in_executor()`, which runs the blocking, CPU-bound `extract_text` function in a separate thread pool. This allows the main asynchronous event loop (e.g., in a web server or the GUI) to remain responsive while the file is being processed in the background.

---

## Internal Helper Functions (by File Type)

The core of the module is a set of private helper functions, each tailored to a specific file format.

- **`_extract_pdf(path, ...)`:** Uses the `pypdf` library. It's designed to be robust, with logic to handle encrypted PDFs (by trying an empty password) and to iterate through pages safely, respecting character limits.
- **`_extract_word_document(path, ...)`:** Uses the `python-docx` library for `.docx` files. For legacy `.doc` files, it demonstrates **platform-specific logic**: on Windows (`os.name == 'nt'`), it attempts to use the `pywin32` library to automate Microsoft Word for the most accurate extraction. As a fallback on other platforms, it tries to use the `textract` library.
- **`_extract_excel(path, ...)`:** Uses the `pandas` library, along with `openpyxl` (for `.xlsx`) and `xlrd` (for `.xls`). It iterates through each sheet in the workbook and converts the content to a CSV-like string format, prefixed with the sheet name for context.
- **`_extract_powerpoint(path, ...)`:** Uses the `python-pptx` library to iterate through slides and shapes, extracting text content.
- **`_extract_rtf(path, ...)`:** Uses the `striprtf` library.
- **`_extract_eml(path, ...)` and `_extract_msg(path, ...)`:** Handle raw email files. `_extract_eml` uses Python's built-in `email` package. `_extract_msg` uses the `extract-msg` library for Outlook's proprietary `.msg` format and includes important resource cleanup logic to close file handles.
- **`_html_to_text(html)`:** A helper for converting HTML content (found in some emails or `.html` files) into plain text. It prefers to use `BeautifulSoup` for accurate parsing but has a simpler regex-based fallback if the library isn't installed.

---

## Key Design Patterns

- **Facade Pattern:** As mentioned, `extract_text` provides a simple, unified interface to a complex and varied set of underlying subsystems (the different file parsing libraries).
- **Strategy Pattern:** The main `if/elif` block in `extract_text` acts as a dispatcher, selecting the correct "strategy" (the appropriate `_extract_*` function) based on the file type.
- **Dependency Encapsulation:** This module is the *only* place in the application that has direct dependencies on the suite of file parsing libraries (`pypdf`, `pandas`, etc.). This is excellent design because if a library needs to be swapped out or upgraded, the changes are localized to this single file.
- **Resilience and Graceful Degradation:** The module is built to be resilient. The extensive use of `try...except ImportError` means the application won't crash if an optional dependency for a specific file type is missing; it will simply be unable to process that format. Similarly, `try...except Exception` blocks within each helper ensure that a single corrupt file doesn't halt the entire system.