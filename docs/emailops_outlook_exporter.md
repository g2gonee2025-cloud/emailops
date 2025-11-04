# `emailops.outlook_exporter`

**Primary Goal:** To provide a robust and efficient mechanism for exporting email conversations from Microsoft Outlook into a format suitable for ingestion by the EmailOps system.

## Directory Mapping

```
.
└── emailops/
    └── outlook_exporter/
        ├── __init__.py
        ├── attachments.py
        ├── cli.py
        ├── conversation.py
        ├── exporter.py
        ├── manifest_builder.py
        ├── mapitags.py
        ├── smtp_resolver.py
        ├── state.py
        └── utils.py
```

---

## Core Components & Connections

This module is designed to be run as a standalone CLI tool for exporting emails from Outlook.

### `exporter.py`

- **Purpose:** This is the main entry point for the exporter. It contains the `OutlookExporter` class, which orchestrates the entire export process.
- **Connections:**
    - Uses `conversation.py` to generate conversation keys for threading.
    - Uses `manifest_builder.py` to create the `manifest.json` file for each conversation.
    - Uses `attachments.py` to save attachments and extract their text content.
    - Uses `state.py` to manage the export state and support incremental exports.

### `conversation.py`

- **Purpose:** This module is responsible for generating a unique key for each conversation, which is used for threading.
- **Alignment with Canonical Blueprint:** The `get_conversation_key` function has been updated to align with the canonical blueprint's requirements for threading. It now uses the following properties in order of preference:
    1. `ConversationID`
    2. `PR_REFERENCES` and `PR_IN_REPLY_TO_ID` (RFC-style threading)
    3. `ConversationIndex`
    4. Subject-based fallback

### `manifest_builder.py`

- **Purpose:** This module is responsible for creating the `manifest.json` file for each conversation.
- **Alignment with Canonical Blueprint:** The `_extract_new_message_content` function has been updated to "mask" quoted text instead of removing it. Quoted text is now wrapped in `<quoted_text>...</quoted_text>` tags, preserving the full content of the email while clearly delineating the conversational history.

### `attachments.py`

- **Purpose:** This module is responsible for saving attachments and extracting their text content.
- **Alignment with Canonical Blueprint:** The `save_attachments_for_items` function has been updated to use the `core_text_extraction` module to extract text from attachments. The extracted text is then stored in the `manifest.json` file.

### `util_processing.py`

- **Purpose:** This module contains various utility functions for processing text.
- **Alignment with Canonical Blueprint:** A `redact_pii` function has been added to this module to redact common PII like email addresses and phone numbers. This function is integrated into the `TextPreprocessor` class to ensure that all text processed for indexing is automatically sanitized.
