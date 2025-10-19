# `emailops_gui.py` - EmailOps Graphical User Interface

## 1. Overview

This module provides a comprehensive, production-grade graphical user interface (GUI) for the EmailOps application, built using Python's `tkinter` library. It exposes all major features of the EmailOps suite in a user-friendly, tabbed interface.

**Core Features:**
-   Advanced Search with filters and ranking
-   Drafting replies and new emails
-   Conversational chat with session management
-   Conversation browsing and file management
-   Index building with real-time progress
-   Configuration management
-   System diagnostics and health checks
-   Text chunking and thread analysis

---

## 2. Architecture and Core Components

### 2.1. Main Application Class: `EmailOpsApp`

The entire GUI is encapsulated within the `EmailOpsApp` class, which inherits from `tk.Tk`.

**Key Responsibilities:**
-   Initializes the main application window.
-   Manages application state, including settings and task control.
-   Builds the main UI components (menu, header, tabs).
-   Handles background tasks like logging and progress updates.

### 2.2. State Management

-   **`AppSettings` Dataclass**: A dataclass that holds all persistent application settings. It can be loaded from and saved to a JSON file (`~/.emailops_gui.json`).
-   **`TaskController` Class**: Manages the state of long-running, asynchronous tasks. It provides mechanisms for starting, stopping, and canceling tasks, as well as tracking their progress.

### 2.3. Logging Infrastructure

-   **`QueueHandler` Class**: A custom logging handler that directs log messages to a `queue.Queue`.
-   **`configure_logging()`**: Sets up the root logger to use the `QueueHandler`, allowing log messages from any module to be displayed in the GUI's log tab.

---

## 3. UI Structure

The GUI is organized into a series of tabs, each dedicated to a specific function.

### 3.1. Main Tabs

-   **Search**: For searching the email index.
-   **Draft Reply**: For generating a reply to an existing conversation.
-   **Draft Fresh**: For composing a new email.
-   **Chat**: For conversational interaction with the email data.
-   **Conversations**: For browsing and managing email conversations.
-   **Batch Operations**: For performing actions on multiple conversations at once.
-   **Index**: For building and managing the search index.
-   **Configuration**: For managing application and environment settings.
-   **Diagnostics**: For running system health checks.
-   **Chunking**: For preparing conversations for indexing.
-   **Analyze**: For summarizing and analyzing email threads.
-   **Logs**: For displaying application logs.

### 3.2. Key UI Elements

-   **`LabeledSpinbox`**: A custom widget combining a `ttk.Label` and a `ttk.Spinbox`.
-   **`run_with_progress` Decorator**: A decorator that wraps methods to run them in a separate thread, managing UI state (disabling buttons, showing progress bars) during execution.

---

## 4. Core Functionality by Tab

### 4.1. Search Tab
-   **UI**: Search query input, `k` and `sim_threshold` spinboxes, advanced filter options (collapsible), results treeview, and snippet display.
-   **Workflow**:
    1.  User enters a query and clicks "Search".
    2.  `_on_search()` is called, which runs in a background thread via the `run_with_progress` decorator.
    3.  Advanced filters (from, to, subject, date) are collected if the advanced section is open.
    4.  The `_search` function from `search_and_draft` is called.
    5.  Results are displayed in the treeview.
    6.  Selecting a result in the treeview displays its text snippet.

### 4.2. Draft Reply Tab
-   **UI**: Conversation ID combobox, optional query input, token and policy controls, and a text area for the generated reply.
-   **Workflow**:
    1.  User selects a conversation ID and clicks "Generate Reply".
    2.  `_on_draft_reply()` is called.
    3.  `draft_email_reply_eml` from `search_and_draft` is invoked.
    4.  The generated email body and metadata are displayed.
    5.  The user can save the generated reply as a `.eml` file.

### 4.3. Draft Fresh Tab
-   **UI**: "To", "Cc", and "Subject" fields, an input for the drafting intent/instructions, and a text area for the generated email.
-   **Workflow**:
    1.  User fills in the recipient and subject details and provides a query.
    2.  `_on_draft_fresh()` calls `draft_fresh_email_eml` from `search_and_draft`.
    3.  The generated email is displayed and can be saved as a `.eml` file.

### 4.4. Chat Tab
-   **UI**: Session management controls (ID, load, save, reset), a query input, and a text area for the chat history.
-   **Workflow**:
    1.  User enters a question and clicks "Ask".
    2.  `_on_chat()` is called.
    3.  A `ChatSession` object is used to manage history.
    4.  The `chat_with_context` function from `search_and_draft` is called.
    5.  The user's question and the assistant's answer are appended to the chat history.

### 4.5. Conversations Tab
-   **UI**: A treeview to list all conversations, with buttons to view conversation details, open attachments, or open the conversation folder.
-   **Workflow**:
    1.  "List Conversations" button calls `_on_list_convs()`.
    2.  `list_conversations_newest_first` from `search_and_draft` is called.
    3.  The treeview is populated with conversation details.
    4.  Double-clicking a conversation opens its `Conversation.txt` in a new window.

### 4.6. Index Tab
-   **UI**: Controls for batch size, number of workers, and force re-indexing, along with a progress bar and status label.
-   **Workflow**:
    1.  "Build / Update Index" button calls `_on_build_index()`.
    2.  This function constructs and runs the `email_indexer` module as a subprocess.
    3.  The GUI monitors the subprocess's output to display real-time progress.

### 4.7. Configuration Tab
-   **UI**: A scrollable form with fields for all major configuration settings (GCP, indexing, email).
-   **Workflow**:
    1.  User modifies the settings and clicks "Apply Configuration".
    2.  `_apply_config()` updates the `AppSettings` object, saves it to JSON, and updates the corresponding environment variables.

### 4.8. Diagnostics Tab
-   **UI**: Buttons to run various diagnostic checks and a text area to display the results.
-   **Workflow**:
    -   Each button calls a corresponding function (`_run_diagnostics`, `_check_deps`, etc.).
    -   These functions call the underlying logic from the `doctor` module.
    -   The results are formatted and displayed in the text area with color-coded tags for success, warning, and error.

---

## 5. Dependencies

-   **`tkinter`**: The core GUI library.
-   **`emailops` modules**: `doctor`, `email_indexer`, `processor`, `text_chunker`, `summarize_email_thread`, `config`, `utils`, `validators`, `search_and_draft`.
