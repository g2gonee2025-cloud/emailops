# `emailops.utils`

**Primary Goal:** To provide a centralized collection of common, low-level utility functions that are used throughout the EmailOps application. This module helps to avoid code duplication and ensures that basic operations like file discovery, directory creation, and data sanitization are handled in a consistent and robust manner.

## Directory Mapping

```
.
└── emailops/
    └── utils.py
```

---

## Core Functions & Connections

This module contains a variety of helper functions that serve different parts of the application.

### File System Utilities

- **`find_conversation_dirs(root)`:**
    - **Purpose:** To scan the root export directory and identify all valid "conversation" subdirectories.
    - **Functionality:** It iterates through the items in the `root` path and returns a list of directories that contain a `Conversation.txt` file and do not start with a special character (`_` or `.`), which is a convention for ignoring folders like `_index` or `.wal`.
    - **Connections:** This is a fundamental utility used by `indexing_main.build_corpus` to discover the raw data that needs to be processed and indexed.

- **`ensure_dir(path)`:**
    - **Purpose:** A simple convenience function that ensures a directory exists, creating it (and any necessary parent directories) if it does not.
    - **Connections:** Used by various modules, such as `index_transaction`, before writing files to a directory that might not exist yet (e.g., the `.wal` or `backup` directories).

- **`read_text_file(path, ...)`:**
    - **Purpose:** A safe and robust function for reading the content of a text file.
    - **Functionality:** It defaults to reading with `utf-8-sig` encoding, which correctly handles the Byte Order Mark (BOM) often found in text files created on Windows. It also includes a `try...except` block to gracefully handle `OSError` or other issues, returning an empty string instead of crashing.
    - **Connections:** Used by `core_conversation_loader._load_conversation_text` to read the main `Conversation.txt` file.

### Data Sanitization

- **`_strip_control_chars(s, ...)`:**
    - **Purpose:** To remove non-printable ASCII control characters from a string. These characters can cause issues with JSON parsers, XML parsers, and other downstream systems.
    - **Functionality:** It uses a pre-compiled regular expression to find and remove characters in the ranges `\x00-\x1F` (excluding standard whitespace like tab, newline, and carriage return).
    - **Connections:** This is a low-level utility used by higher-level cleaning functions like `core_email_processing.clean_email_text` and `scrub_json`.

- **`scrub_json(data)` & `scrub_json_string(s)`:**
    - **Purpose:** To ensure that data is safe for JSON serialization or parsing.
    - **Functionality:** `scrub_json_string` is a simple alias for `_strip_control_chars`. `scrub_json` is a recursive function that can traverse a nested data structure (dicts and lists) and apply the character stripping to all string values within it.
    - **Connections:** Used by `core_conversation_loader` before loading JSON from `manifest.json` or `summary.json` to prevent parsing errors caused by stray control characters.

### General-Purpose Helpers

- **`safe_str(v, max_len, ...)`:**
    - **Purpose:** A defensive function to convert any value to a string while enforcing a maximum length.
    - **Functionality:** It handles `None` inputs gracefully (returning an empty string) and truncates strings that are longer than `max_len`. As noted in the source code, this is a consolidated function, which is good practice to avoid having multiple slightly different implementations of the same utility.
    - **Connections:** Used by `feature_summarize._normalize_analysis` to ensure that data extracted by the LLM doesn't exceed the schema's intended field lengths.

- **`monitor_performance(func)`:**
    - **Purpose:** A decorator for simple performance monitoring.
    - **Functionality:** It times the execution of the decorated function and logs a warning if it takes longer than one second. This function was consolidated from `util_processing.py`, which is a good refactoring step to centralize common utilities.
    - **Connections:** Used by `util_processing.TextPreprocessor` to monitor the performance of the text preparation step.

- **`logger`:**
    - The module initializes and exports a standard Python `logging.Logger` instance. Other modules can import and use this shared logger to ensure consistent log formatting and output.

## Key Design Patterns

- **Utility Module:** This module follows the common pattern of a "utility" or "helper" library. It contains a collection of small, stateless, and highly reusable functions that don't belong to any single business logic domain but are needed by many.
- **Defensive Programming:** Functions like `read_text_file` and `safe_str` are designed defensively. They anticipate potential problems (file not found, `None` input) and handle them gracefully instead of raising exceptions, which can make the higher-level code that uses them cleaner.
- **Code Consolidation (Refactoring):** The comments in the source code indicate that `safe_str` and `monitor_performance` were consolidated from other modules. This is an important refactoring practice that adheres to the "Don't Repeat Yourself" (DRY) principle, leading to a more maintainable codebase.