# `emailops.vertex_agent_builder`

**Primary Goal:** To provide a high-level, production-ready client for interacting with Google Cloud's Vertex AI Agent Builder (formerly Discovery Engine). This module encapsulates the complexity of the underlying Google Cloud client libraries, offering a clean, task-oriented API for advanced features like grounded generation, fact-checking, and conversational search.

## Directory Mapping

```
.
└── emailops/
    └── vertex_agent_builder.py
```

---

## Core Component: `VertexAgentBuilder` Class

This class is the central point of interaction with the Agent Builder service.

- **Purpose:** To manage the connection and requests to a specific Agent Builder datastore.
- **Initialization:**
    - It is instantiated with a `project_id`, `location`, and `datastore_id`.
    - Upon initialization, it creates client instances for the various Discovery Engine services (`SearchServiceClient`, `ConversationalSearchServiceClient`, etc.).
- **Connections:** This class depends on the application's core configuration (`core_config`) to get the necessary GCP project details. The `create_agent_builder` factory function provides a convenient way to instantiate it from the global config.

---

## Key Features & Methods

This module exposes several powerful AI capabilities through the methods of the `VertexAgentBuilder` class.

### Grounded Generation

- **`answer_query(query, ...)`:**
    - **Purpose:** This is the primary method for performing grounded Question-Answering (Q&A). It sends a query to the Agent Builder and receives an answer that is explicitly grounded in the documents within the specified datastore.
    - **Functionality:** It constructs an `AnswerQueryRequest`, including specifications for citation extraction and a system preamble. It then calls the `answer_query` method of the `SearchServiceClient`.
    - **Return Value:** It returns a `GroundedAnswer` dataclass, which neatly packages the generated `answer` text, a list of `citations` (with document IDs and snippets), and a `grounding_score`. This structured response is crucial for building trustworthy AI systems, as it allows the application to show users *why* the model gave a particular answer.

### Conversational Search

- **`create_session(...)`, `list_sessions(...)`, `multi_turn_search(...)`:**
    - **Purpose:** This group of methods provides the tools for building a stateful, multi-turn conversational experience.
    - **Functionality:**
        1.  `create_session` is called to start a new conversation, which returns an `AgentSession` object containing a unique `session.name`.
        2.  This `session.name` is then passed to subsequent calls of `multi_turn_search`. This tells the Agent Builder to interpret the new query in the context of the previous turns of the conversation.
        3.  For example, if the first query is "What's our D&O coverage?" and the second is "What about E&O?", `multi_turn_search` allows the model to understand that "What about" refers to the coverage for "E&O" in the context of the previous question.
    - **Connections:** This provides a much more advanced search capability than the simple vector search implemented in `feature_search_draft`, as it leverages the conversational understanding built into the Vertex AI platform.

### Fact-Checking

- **`check_grounding(answer_candidate, facts, ...)`:**
    - **Purpose:** To verify whether a given statement is factually supported by a list of source texts. This is a powerful feature for combating model hallucinations.
    - **Functionality:** It takes a generated `answer_candidate` and a list of `facts` (strings of text from source documents) and sends them to the `check_grounding` API endpoint. The service then determines if the claims in the answer can be found within the provided facts.
    - **Return Value:** It returns a `GroundingCheck` object, which contains a boolean `is_grounded` flag, a confidence score, and, importantly, a list of any `unsupported_claims`.
    - **Connections:** This method can be integrated into the drafting pipeline (like in `feature_search_draft`) as an additional verification step after the "critic" and "auditor" to provide an even higher degree of confidence in the final output.

### Semantic Ranking

- **`rank_documents(query, documents, ...)`:**
    - **Purpose:** To re-rank a list of documents based on their semantic relevance to a query, rather than just keyword matching.
    - **Functionality:** It takes a query and a list of documents (each with an `id` and `content`) and sends them to the `rank` API endpoint. The service returns the same documents but reordered according to their relevance.
    - **Connections:** This provides a more sophisticated alternative to the local reranking logic in `feature_search_draft`. It could be used as a replacement for, or in addition to, the existing cross-encoder reranking step to leverage Google's powerful semantic models.

## Key Design Patterns

- **Facade Pattern:** The `VertexAgentBuilder` class is a classic Facade. It provides a simplified, task-oriented API (`answer_query`, `check_grounding`) that hides the complex, verbose, and protocol-buffer-based requests required by the underlying Google Cloud `discoveryengine` client libraries.
- **Data Transfer Objects (DTOs):** The use of dataclasses like `GroundedAnswer`, `GroundingCheck`, and `RankedDocument` is a best practice. Instead of returning raw API response dictionaries, the methods return strongly-typed, self-documenting objects, which makes the code that calls them cleaner, safer, and easier to understand.
- **Provider Abstraction:** While this module is specific to Vertex AI, it follows the same principle as the `llm_runtime` by encapsulating all provider-specific logic. This keeps the main application logic clean and makes it clear which part of the system is responsible for interacting with this particular external service.