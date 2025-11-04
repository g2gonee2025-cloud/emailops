# `emailops.feature_search_draft`

**Primary Goal:** To serve as the central intelligence of the EmailOps application. This module orchestrates the entire Retrieval-Augmented Generation (RAG) pipeline, from searching the knowledge base to drafting high-quality, context-aware emails. It contains the core logic for the `reply`, `fresh`, and `chat` commands.

## Directory Mapping

```
.
└── emailops/
    └── feature_search_draft.py
```

---

## High-Level Orchestration Functions

These are the main entry points that `emailops.cli` calls to perform a complete user-facing task.

### `draft_email_reply_eml(...)`

- **Purpose:** To generate a complete `.eml` reply for an existing conversation.
- **Orchestration Flow:**
    1.  **Load Conversation Data:** It starts by loading the existing conversation's manifest and text from the specified `conv_id`.
    2.  **Derive Query:** If the user doesn't provide an explicit query, it intelligently derives one by looking at the *last inbound message* in the thread, making the system feel more intuitive.
    3.  **Gather Context:** It calls `_gather_context_for_conv()` to perform a focused search for relevant information *within that specific conversation*.
    4.  **Draft Email:** It passes the query and the retrieved context to `draft_email_structured()` to generate the email content.
    5.  **Construct EML:** It uses the generated content, along with recipient and subject information derived from the conversation history, to call `_build_eml()` and create the final, ready-to-send `.eml` file.

### `draft_fresh_email_eml(...)`

- **Purpose:** To generate a new `.eml` file from scratch based on a user's request.
- **Orchestration Flow:**
    1.  **Parse Query:** It first calls `parse_filter_grammar()` to separate any structured filters (e.g., `from:`, `subject:`) from the free-text part of the user's query.
    2.  **Gather Context:** It calls `_gather_context_fresh()` to perform a broad search across the *entire knowledge base*, constrained by any filters found in the query.
    3.  **Draft Email:** It passes the query and context to `draft_email_structured()` for generation.
    4.  **Construct EML:** It calls `_build_eml()` using the user-provided recipients and subject to create the final `.eml` file.

### `chat_with_context(...)`

- **Purpose:** To provide a direct, conversational answer to a user's question, grounded in retrieved context.
- **Orchestration Flow:**
    1.  The `emailops.cli` first calls `_search()` to get a list of relevant context snippets.
    2.  These snippets, along with the user's query and chat history, are passed to `chat_with_context()`.
    3.  This function formats the inputs into a prompt and calls the LLM via `complete_json()` to get a structured answer containing the text, citations, and any missing information.

---

## The RAG Pipeline: Key Stages

### 1. Context Gathering (Retrieval)

- **`search()`:** This is the heart of the retrieval system. It implements a sophisticated, multi-stage pipeline to find the most relevant information:
    1.  **Load Index:** They load the `mapping.json` and the `embeddings.npy` files from the index directory, using caching (`_get_cached_mapping`, `_ensure_embeddings_ready`) for performance.
    2.  **Prefilter:** They apply any structured filters from the query (`apply_filters`) to narrow down the search space *before* performing the expensive vector search.
    3.  **Vector Search:** They embed the user's query and perform a vector similarity search against the document embeddings to get an initial set of candidates.
    4.  **Recency Boost:** The scores are boosted based on the age of the documents (`_boost_scores_for_indices`), favoring more recent information.
    5.  **Deduplication:** Duplicate content is removed by checking the `content_hash` of each chunk, ensuring diversity in the results.
    6.  **Reranking:** The candidates are reranked by creating short summaries and embedding them, providing a second pass of semantic relevance checking.
    7.  **Diversification (MMR):** A Maximal Marginal Relevance (MMR) algorithm (`_mmr_select`) is used to select a final set of documents that is both relevant to the query and diverse in content, avoiding a context filled with redundant information.
    8.  **Content Loading:** Finally, the full text for the selected documents is loaded from disk.

### 2. Email Drafting (Generation)

- **`draft_email_structured(...)`:** This function implements a "multi-agent" or "chain-of-thought" workflow to produce a high-quality draft.
    1.  **Initial Draft:** It first prompts the LLM in JSON mode with the user's query and the retrieved context, asking for a structured output that includes the draft text, citations, and mentioned attachments. It has a retry mechanism that increases temperature and falls back to a text-based prompt if the JSON generation fails.
    2.  **Critic Pass:** The initial draft is then passed to a second LLM call, acting as a "critic." The critic's job is to review the draft for quality, accuracy, and adherence to instructions, outputting a structured list of issues and suggested improvements.
    3.  **Auditor Loop:** The draft is then passed to an "auditor" that scores it against a predefined rubric (e.g., `factuality_rating`, `citation_quality`). If the scores are below a target threshold, the system enters a loop:
        - It prompts the LLM again, this time as a "senior comms specialist," providing the draft and the low audit scores, and asks it to *improve* the draft specifically to address the low scores.
        - The improved draft is then re-audited. This loop continues until the draft passes the quality check or a maximum number of attempts is reached.
    4.  **Attachment Selection:** Based on the final draft's citations and mentions, it intelligently selects the most relevant files to attach (`_select_attachments_from_citations`, `_select_attachments_from_mentions`).

---

## Security & Safety

- **Prompt Injection Defense:** The module takes prompt injection seriously. The system prompts explicitly instruct the LLM to **never** follow instructions found within the context snippets. Additionally, the `_hard_strip_injection` function uses a regex (`_INJECTION_PATTERN_RE`) to proactively remove known injection patterns from the text before it's sent to the LLM.
- **Input Validation:** The code includes checks to prevent overly long queries and to validate the structure of the context snippets, adding a layer of defense against malformed inputs.

## Key Design Patterns

- **Retrieval-Augmented Generation (RAG):** The entire module is a sophisticated implementation of the RAG pattern.
- **Multi-Agent System / Chain of Verification:** The Draft -> Critic -> Audit -> Improve workflow is a powerful pattern that uses the LLM to iteratively refine and self-correct its own output, leading to much higher quality results than a single-shot prompt.
- **Facade Pattern:** High-level functions like `draft_email_reply_eml` provide a simple interface to the complex, multi-stage pipeline, making them easy to use from the CLI or other parts of the application.
- **Caching:** The use of in-memory, thread-safe caches for query embeddings and the index mapping file significantly improves performance for repeated or similar requests.