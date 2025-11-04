# `emailops.cli`

**Primary Goal:** Acts as the main command-line orchestrator for the EmailOps system, providing a unified interface for various high-level operations. It delegates tasks to specialized modules, ensuring a clean separation of concerns.

## Directory Mapping

```
.
└── emailops/
    └── cli.py
```

---

## Core Functions & Connections

This module serves as a "thin helper," parsing user commands and invoking the appropriate backend functionality. It ensures that environment variables and configurations are correctly propagated to child processes.

### `main()`

- **Purpose:** The main entry point for the entire CLI application. It builds the command-line argument parser, parses the user's command, and calls the corresponding function (`ns.func`).
- **Connections:**
    - Calls `build_cli()` to set up the `argparse` structure.
    - Executes the specific `cmd_*` function determined by the user's input (e.g., `cmd_index`, `cmd_reply`).
    - Implements global exception handling to provide clear error messages and return codes.

### `build_cli()`

- **Purpose:** Constructs the `ArgumentParser` object that defines the entire command-line interface, including all subcommands and their arguments.
- **Connections:**
    - Uses helper functions (`_add_index_parser`, `_add_reply_parser`, etc.) to define the arguments for each subcommand. This keeps the main build function clean and modular.

---

## Subcommands & Delegations

Each subcommand corresponds to a major feature of the EmailOps system and delegates its logic to a more specialized module.

### `cmd_index(ns)`

- **Purpose:** Handles the `index` command. It initiates the process of building or updating the vector index for email conversations.
- **Connections:**
    - **Delegates to:** `emailops.indexing_main` by calling `_run_email_indexer()`.
    - `_run_email_indexer()` runs the indexer in a separate subprocess. This is a critical design choice to isolate the indexer's dependencies and prevent `sys.argv` conflicts.
    - It uses `core_validators.validate_command_args` to ensure the command being run is safe before execution.

### `cmd_reply(ns)`

- **Purpose:** Handles the `reply` command. It drafts a reply email (`.eml`) for a specific conversation.
- **Connections:**
    - **Delegates to:** `emailops.feature_search_draft.draft_email_reply_eml()`.
    - This function encapsulates the complex logic of retrieving context, generating a response with an LLM, and formatting it as an `.eml` file.
    - Uses `services.atomic_file_service.AtomicFileService` to write the output file safely.

### `cmd_fresh(ns)`

- **Purpose:** Handles the `fresh` command. It drafts a new email from scratch based on a user's query and specified recipients.
- **Connections:**
    - **Delegates to:** `emailops.feature_search_draft.draft_fresh_email_eml()`.
    - Leverages the same underlying retrieval and generation pipeline as `cmd_reply` but for composing new messages.
    - It calls `feature_search_draft.parse_filter_grammar` to validate the query syntax early.

### `cmd_chat(ns)`

- **Purpose:** Handles the `chat` command. It allows for an interactive chat session grounded in the context retrieved from the email knowledge base.
- **Connections:**
    - **Delegates to:**
        - `emailops.feature_search_draft._low_level_search()` to first retrieve relevant context snippets.
        - `emailops.feature_search_draft.chat_with_context()` to generate a conversational response based on the retrieved context.

### `cmd_summarize(ns)`

- **Purpose:** Handles the `summarize` command. It generates a summary for a single email conversation.
- **Connections:**
    - **Delegates to:** `emailops.feature_summarize.analyze_conversation_dir()`.
    - The result is then formatted using `format_analysis_as_markdown()`.

### `cmd_summarize_many(ns)`

- **Purpose:** Handles the `summarize-many` command. It summarizes multiple conversations in parallel for efficiency.
- **Connections:**
    - Uses Python's `multiprocessing` module (with a 'spawn' context for safety) to distribute the summarization work across multiple CPU cores.
    - The actual work for each conversation is performed by the `_summarize_worker` function, which in turn calls `analyze_conversation_dir()` from the `feature_summarize` module. This is a classic map-reduce pattern.

---

## Key Design Patterns

- **Orchestrator Pattern:** `cli.py` doesn't implement the core logic itself; it orchestrates calls to other modules that do.
- **Subprocess Isolation:** The indexer is run in a separate process to prevent dependency conflicts and ensure a clean environment.
- **Safe Multiprocessing:** For batch operations like `summarize-many`, it uses the `spawn` start method and top-level worker functions to avoid issues with state sharing and pickling that can occur with the default `fork` method on non-Windows systems.
- **Configuration Hub:** It relies on `core_config.get_config()` to fetch configuration and `_ensure_env()` to propagate it to child processes, centralizing configuration management.