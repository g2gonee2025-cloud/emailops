# `emailops.llm_runtime`

**Primary Goal:** To serve as the concrete implementation layer for all interactions with external Language Model (LLM) and embedding providers. This module handles the low-level details of API calls, authentication, rate limiting, and provider-specific logic, exposing a set of standardized functions to the rest of the application via the `llm_client_shim`.

## Directory Mapping

```
.
└── emailops/
    ├── llm_client_shim.py
    └── llm_runtime.py
```

---

## Core Functions & Connections

This module is the engine room of the application's AI capabilities. It directly interacts with external services.

### `complete_text(...)` & `complete_json(...)`

- **Purpose:** These functions are responsible for generating text from a language model. `complete_text` returns free-form text, while `complete_json` is specialized for generating structured JSON output, often guided by a provided schema.
- **Functionality:**
    - **Provider Logic:** They contain the specific code needed to call the Vertex AI Gemini API.
    - **Configuration:** They construct a `GenerationConfig` object to control parameters like `max_output_tokens` and `temperature`.
    - **Safety Settings:** They explicitly disable all content safety filters (e.g., for hate speech, harassment), which is a common requirement for enterprise use cases where the input data is controlled and false positives from safety filters can be disruptive.
    - **JSON Fallback:** `complete_json` includes a robust fallback mechanism. If the model fails to return valid JSON in its native JSON mode, it automatically re-prompts the model in text mode and then attempts to parse the JSON from the resulting text using `_extract_json_from_text`. This makes the function highly resilient to model errors.
- **Connections:** These functions are the concrete implementations that are called by the wrappers in `llm_client_shim.py`.

### `embed_texts(...)`

- **Purpose:** This is the master function for converting text into vector embeddings. It's a prime example of the **Strategy Pattern**.
- **Functionality:**
    - **Provider Dispatching:** Based on the `provider` argument (e.g., "vertex", "openai", "local"), it dispatches the call to a specific, private helper function (e.g., `_embed_vertex`, `_embed_openai`, `_embed_local`).
    - **Normalization:** After receiving the vectors from the provider-specific function, it passes them to `_normalize()`, which performs L2 normalization to convert them into unit vectors. This is a crucial step for ensuring consistent and mathematically sound similarity searches.
    - **Input Validation:** It performs sanity checks on the final output array to ensure the dimensions are correct and the values are finite, preventing corrupted data from entering the index.
- **Connections:** This function is the implementation behind `llm_client_shim.embed_texts` and is the engine that powers the "embedding" stage of the `indexing_main` pipeline.

---

## Provider-Specific Implementations

The module contains a set of private functions, each dedicated to a single LLM provider. This encapsulates the unique API and authentication requirements of each service.

- **`_embed_vertex(...)`:** Handles the complexity of Google's Vertex AI, including logic for both the modern `google-genai` and legacy `vertexai.language_models` APIs.
- **`_embed_openai(...)`, `_embed_azure_openai(...)`, `_embed_cohere(...)`, etc.:** Each of these functions imports its specific client library (e.g., `openai`, `cohere`), retrieves the necessary API keys from environment variables, and formats the request according to that provider's specification.

---

## Resilience and Scalability

This module is built for production use and includes several advanced features for resilience and scalability.

- **`@with_retry` & `@circuit_breaker`:** The core `complete_*` functions are decorated with these wrappers (from `services.resilience`).
    - **`@with_retry`:** Automatically retries an API call if it fails with a "retryable" error (like a rate limit or temporary server error), using exponential backoff to avoid overwhelming the service.
    - **`@circuit_breaker`:** If a service fails repeatedly, the circuit breaker will "trip" and stop sending requests for a configured period. This prevents the application from wasting resources on a known-unresponsive service and allows the service time to recover.
- **Rate Limiting (`_check_rate_limit`)**: Before making an API call, the code calls this function, which uses a thread-safe deque to track the timestamps of recent calls. If the number of calls in the last minute exceeds the configured limit, it sleeps just long enough for the oldest call to expire, effectively throttling the application to stay within API limits.
- **Project Rotation (`_rotate_to_next_project`)**: This is a sophisticated feature for scaling beyond the limits of a single cloud account.
    - It loads a list of validated service accounts from a configuration file.
    - If an API call fails with a quota-related error (`_should_rotate_on`), it automatically rotates to the *next* account in the list, re-initializes the Vertex AI client with the new credentials, and retries the request.
    - This allows the application to "burst" across multiple accounts, achieving much higher throughput than would be possible with a single project.

## Key Design Patterns

- **Strategy Pattern:** The `embed_texts` function is a perfect example. The main function defines the overall algorithm (embed -> normalize -> validate), but the specific embedding "strategy" is chosen and delegated at runtime based on the provider.
- **Resilience Patterns:** The explicit use of Retry, Circuit Breaker, and Rate Limiting patterns makes the module robust against the transient failures common in distributed systems.
- **Lazy Initialization:** The configuration (`_config`) and the list of validated accounts (`_validated_accounts`) are loaded only when they are first needed, and the results are cached. This is done using a thread-safe, double-checked locking pattern to ensure it happens safely and efficiently in a multi-threaded environment.