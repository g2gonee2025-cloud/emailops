# `emailops.core_email_processing`

**Primary Goal:** To provide a suite of heuristic-based utilities for cleaning, parsing, and structuring raw email text. This module is essential for transforming messy, real-world email content into a clean format suitable for analysis, indexing, and processing by language models.

## Directory Mapping

```
.
└── emailops/
    └── core_email_processing.py
```

---

## Core Functions & Connections

This module uses a set of pre-compiled regular expressions to perform its tasks efficiently. The functions are designed to be conservative, prioritizing the preservation of substantive content over aggressive cleaning.

### `clean_email_text(text, ...)`

- **Purpose:** This is the primary cleaning function. It takes a raw email body as a string and applies a series of transformations to remove noise and standardize the content.
- **Cleaning Steps:**
    1.  **Header Removal:** Strips out common email header lines (e.g., `From:`, `Subject:`, `Date:`).
    2.  **Signature/Footer Removal:** Scans the *end* of the email text for common signature patterns (e.g., "Best regards," "Sent from my iPhone") and legal disclaimers, removing them. This is limited to the last few thousand characters to avoid accidentally removing content from the middle of a long email.
    3.  **Forwarding Separator Removal:** Deletes lines like "--- Original Message ---".
    4.  **Quoted Reply Stripping:** Removes the `>` characters that prefix quoted replies in an email thread.
    5.  **PII Redaction:** Redacts email addresses and URLs, replacing them with placeholders like `[email@domain]` and `[URL]`. This is a critical step for privacy.
    6.  **Whitespace Normalization:** Collapses multiple spaces, tabs, and newlines to create a more uniform text structure.
    7.  **Control Character Stripping:** Removes non-printable control characters that can interfere with downstream processing.
- **Connections:**
    - **Uses `utils._strip_control_chars()`:** Delegates the final, low-level character stripping to a utility function.
    - This function is likely called by the indexing pipeline (`emailops.indexing_main`) before text is chunked and embedded, ensuring that the vector database is not polluted with irrelevant noise.

### `extract_email_metadata(text)`

- **Purpose:** To perform a "best-effort" extraction of key metadata from the header block of an email.
- **Functionality:**
    - It first isolates the header block (everything before the first double newline).
    - It unfolds multi-line headers (as per RFC 5322) into single lines.
    - It then uses regular expressions to find and extract values for `From`, `To`, `Cc`, `Bcc`, `Date`, and `Subject`.
- **Connections:** This provides a fallback mechanism for metadata extraction when a fully parsed email object is not available. It can supplement the primary metadata loaded from `manifest.json`.

### `split_email_thread(text)`

- **Purpose:** To split a single text block containing an entire email thread into a list of individual messages. This is crucial for understanding the conversational flow and attributing statements to the correct message.
- **Heuristics Used:**
    1.  It uses a regular expression to find common separators like "On ... wrote:" or "--- Original Message ---".
    2.  It splits the text based on these separators.
    3.  **Chronological Sorting:** As a key refinement, it then attempts to parse a `Date:` header from each resulting text block. If multiple blocks have valid dates, it sorts them chronologically. This ensures that the messages are returned in the correct order (oldest to newest), which is vital for any kind of sequential analysis.
- **Connections:** This function is fundamental for any feature that needs to analyze the turn-by-turn nature of a conversation, such as the summarizer (`emailops.feature_summarize`) or when reconstructing a thread for an LLM to draft a reply.

---

## Key Design Patterns

- **Heuristic-Based Processing:** This module does not use a full, compliant RFC 5322 parser. Instead, it uses a set of robust regular expressions and heuristics. This approach is often faster and more resilient to malformed emails, even if it's less precise than a formal parser.
- **Pre-compiled Regex:** All regular expression patterns are compiled at the module level when the file is first imported. This is a standard performance optimization in Python, as it avoids the overhead of recompiling the same pattern every time a function is called.
- **Separation of Concerns:** The module focuses exclusively on text manipulation. It does not deal with file I/O, configuration, or business logic. It simply provides a toolbox of functions for processing email strings.
- **Conservative Cleaning:** The design philosophy is to avoid false positives. For example, signature removal is only performed on the tail end of the email. This is a good trade-off, as it's better to leave a small amount of noise in the text than to accidentally delete part of the actual message body.