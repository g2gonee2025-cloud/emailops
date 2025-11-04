# `emailops.core_manifest`

**Primary Goal:** To serve as the single source of truth for loading, parsing, and extracting data from `manifest.json` files. This module provides canonical, robust functions that the rest of the application should use to access manifest data, ensuring consistency and preventing logic duplication.

## Directory Mapping

```
.
└── emailops/
    └── core_manifest.py
```

---

## Core Functions & Connections

This module is designed around a few key public functions that provide different levels of access to the manifest data.

### `load_manifest(convo_dir)`

- **Purpose:** This is the **canonical loader** for `manifest.json` files. It is designed to be extremely robust and resilient to common file corruption issues.
- **Robust Parsing Strategy:**
    1.  **UTF-8-SIG First:** It first tries to decode the file as `utf-8-sig`, which correctly handles the Byte Order Mark (BOM) that some editors add.
    2.  **Latin-1 Fallback:** If UTF-8 fails, it falls back to `latin-1` decoding, which is less likely to raise an error and can often recover text from partially corrupt files.
    3.  **Control Character Sanitization:** It aggressively strips non-printable control characters that would cause the JSON parser to fail.
    4.  **Strict JSON Parse:** It attempts a standard JSON parse.
    5.  **Backslash Repair:** If the strict parse fails, it uses a regular expression to fix a common issue where unescaped backslashes in file paths (a frequent problem with data from Windows systems) break the JSON structure. It then re-attempts the parse.
    6.  **Graceful Failure:** If all else fails, it logs an error and returns an empty dictionary, preventing the entire application from crashing due to a single corrupt manifest.
- **Connections:** This function is the foundation of the module. It's called by `core_conversation_loader.load_conversation` and the convenience function `get_conversation_metadata` to get the raw manifest data.

### `extract_metadata_lightweight(manifest)`

- **Purpose:** This is the **canonical metadata extractor** for indexing and search filtering. It pulls a small, essential set of fields from a parsed manifest dictionary.
- **Extracted Fields:**
    - `subject`: The conversation subject line.
    - `from`, `to`, `cc`: Lists of participants from the *first message* in the thread, used for filtering.
    - `start_date`, `end_date`: The date of the first and last messages, defining the conversation's time span.
- **Connections:** This function is critical for the indexing pipeline (`emailops.indexing_main`). The metadata it extracts is stored alongside the vector embeddings, allowing for powerful filtered searches (e.g., "find emails from John Doe in the last 30 days about 'Project X'").

### `extract_participants_detailed(manifest, ...)`

- **Purpose:** This function extracts a rich, deduplicated list of all participants in a conversation, designed specifically for use in summarization and analysis where understanding "who is who" is important.
- **Functionality:**
    - It iterates through the `from`, `to`, and `cc` fields of the messages in the manifest.
    - It deduplicates participants based on their email address (or normalized name as a fallback).
    - It returns a list of dictionaries, each with a standardized schema (`name`, `email`, `role`, `tone`, `stance`).
- **Connections:** This is used by `emailops.feature_summarize` to generate a "Dramatis Personae" section in the summary, providing context about the people involved in the conversation. It uses the `Participant` model from `common.models` for data integrity.

### `get_conversation_metadata(convo_dir)`

- **Purpose:** A convenience function that combines the two steps of loading and extracting into one call.
- **Functionality:** It simply calls `load_manifest()` and then passes the result to `extract_metadata_lightweight()`.
- **Connections:** This is the ideal function for any part of the application that needs quick access to the lightweight metadata for a conversation without having to manage the intermediate manifest dictionary.

---

## Key Design Patterns

- **Single Source of Truth:** The docstrings and module description repeatedly emphasize that these functions are the **canonical** way to interact with manifests. This is a deliberate design choice to enforce consistency and make the codebase easier to maintain. If the manifest schema changes, only this file needs to be updated.
- **Robustness and Graceful Degradation:** The `load_manifest` function is a prime example of defensive programming. It anticipates multiple common failure modes (encoding errors, malformed JSON) and has a clear strategy for dealing with each one, ensuring that the system can continue operating even with imperfect data.
- **Separation of Concerns:** The module separates the act of *loading* the raw data (`load_manifest`) from the act of *interpreting* it (`extract_metadata_lightweight`, `extract_participants_detailed`). This is a clean design that allows different parts of the application to get the level of detail they need without re-implementing the extraction logic.
- **Data Integrity:** The use of the `Participant` model (even if converted to dicts for output) and the careful deduplication logic in `extract_participants_detailed` show a focus on data quality and consistency.