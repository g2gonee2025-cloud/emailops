# `emailops.feature_summarize`

**Primary Goal:** To perform a deep, structured analysis of an email conversation and generate a comprehensive summary. This module goes beyond simple text summarization by implementing a multi-agent "facts ledger" approach to create a rich, queryable data structure.

## Directory Mapping

```
.
└── emailops/
    └── feature_summarize.py
```

---

## High-Level Orchestration

The main entry point for this module's functionality is the `analyze_conversation_dir` function.

### `analyze_conversation_dir(thread_dir, ...)`

- **Purpose:** This function orchestrates the entire summarization process for a single conversation directory.
- **Orchestration Flow:**
    1.  **Load Data:** It reads the raw `Conversation.txt` file from the specified directory.
    2.  **Clean Text:** It passes the raw text to `core_email_processing.clean_email_text` to prepare it for the language model.
    3.  **Analyze:** It calls the core `analyze_email_thread_with_ledger()` function to perform the multi-agent analysis.
    4.  **Enrich (Optional):** If `merge_manifest` is true, it calls `_merge_manifest_into_analysis()` to supplement the LLM's output with structured data (like participant emails and precise dates) from the `manifest.json` file. This acts as a "correction" step, grounding the analysis in factual metadata.
    5.  **Normalize:** It runs the final, merged analysis through `_normalize_analysis()` one last time to ensure the final output strictly adheres to the required schema, applying size caps and coercing enum values.
- **Connections:** This is the primary function called by `emailops.cli.cmd_summarize` and `emailops.cli.cmd_summarize_many`.

---

## The Multi-Agent Summarization Pipeline

The core logic resides in `analyze_email_thread_with_ledger`, which implements a "chain of verification" using multiple LLM calls, each with a distinct role.

### Pass 1: The Analyst

- **Purpose:** To perform the initial, broad analysis of the email thread.
- **Process:**
    1.  A detailed system prompt instructs the LLM to act as a "senior insurance account manager" and extract a comprehensive "facts ledger."
    2.  The prompt includes a rich JSON schema defining the desired output structure, covering everything from `participants` and `key_dates` to `commitments_made` and `risk_indicators`.
    3.  It calls `llm_client_shim.complete_json()` to get a structured JSON response.
- **Robustness:** This pass includes a crucial fallback. If the `complete_json` call fails or returns invalid JSON, it retries using `complete_text` and then attempts to parse the JSON from the resulting text using the highly robust `_try_load_json` helper. This makes the initial analysis very resilient to LLM formatting errors.

### Pass 2: The Critic

- **Purpose:** To review the Analyst's work for errors and omissions.
- **Process:**
    1.  A new prompt is constructed, instructing the LLM to act as a "quality control specialist."
    2.  This prompt includes both the original email thread and the JSON output from the Analyst.
    3.  It calls `complete_json` with a different schema, asking the LLM to identify `missed_items`, `accuracy_issues`, and provide an overall `completeness_score`.

### Pass 3: The Improver (Conditional)

- **Purpose:** To correct the initial analysis based on the Critic's feedback.
- **Process:**
    1.  This pass only runs if the Critic's `completeness_score` is below a certain threshold (e.g., 85) or if it identifies `critical_gaps`.
    2.  A third prompt is constructed, instructing the LLM to act as an "expert analyst."
    3.  This prompt contains the initial analysis, the critic's feedback, and the original thread. The LLM's task is to generate a *new, improved* analysis that addresses the flagged issues.
    4.  The final result is produced by `_union_analyses`, a function that intelligently merges the improved analysis with the initial one, ensuring that correct information from the first pass is not accidentally discarded.

---

## Data Handling and Normalization

- **`_normalize_analysis(data, ...)`:** This is a critical function for ensuring data quality. It takes the raw dictionary output from an LLM and rigorously coerces it into the correct schema. It sets default values for missing keys, truncates lists that are too long (e.g., `MAX_PARTICIPANTS`), and, most importantly, validates and coerces enum values (e.g., converting "in-progress" or "inprogress" to the canonical "in_progress").
- **`_try_load_json(data)`:** A highly defensive JSON parsing function. It can handle raw strings, byte strings, and pre-parsed dictionaries. It knows how to extract JSON from within markdown fences (e.g., ` ```json ... ``` `) and can even find the first syntactically balanced `{...}` object in a messy string, making it very effective at salvaging data from imperfect LLM outputs.
- **`_merge_manifest_into_analysis(...)`:** This function enriches the LLM's analysis. While the LLM is good at interpreting tone and summarizing content, it might miss or hallucinate precise details like email addresses or timestamps. This function cross-references the analysis with the ground-truth data from `manifest.json` (loaded via `core_manifest.load_manifest`) to fill in or correct these fields.

## Output Formatting

- **`format_analysis_as_markdown(analysis)`:** Takes the final, normalized analysis dictionary and formats it into a clean, human-readable Markdown document. This is used by the CLI to present the results to the user.
- **`_append_todos_csv(...)`:** This utility extracts the `next_actions` from the analysis and appends them to a root `todo.csv` file, providing a simple way to aggregate tasks from multiple conversations.

## Key Design Patterns

- **Chain of Verification:** The Analyst -> Critic -> Improver pipeline is a powerful pattern that uses multiple, specialized LLM calls to refine and self-correct, leading to a much higher quality output than a single prompt could achieve.
- **Schema-Driven Development:** The entire process is driven by the JSON schemas provided to the LLM. This ensures that the output is structured, predictable, and machine-readable.
- **Robust Data Handling:** The module is designed with the expectation that LLM output can be messy. Functions like `_try_load_json` and `_normalize_analysis` are essential for cleaning and structuring this data into a reliable format.
- **Separation of Concerns:** The module separates the core analysis logic (`analyze_email_thread_with_ledger`) from the data loading and formatting concerns (`analyze_conversation_dir`, `format_analysis_as_markdown`), making the code easier to test and maintain.