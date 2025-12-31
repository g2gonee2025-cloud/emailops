# Preliminary Bulk Code Review Report

> **Note:** The review process is incomplete due to API rate limits. This report summarizes findings from the logs so far.

- **Files Analyzed Successfully:** 46
- **Total Issues Found:** 437
- **Failed Files:** 16

## Top Issues by File

| File | Issues |
|------|--------|
| `orchestrator.py` | 19 |
| `_hybrid_helpers.py` | 18 |
| `queue.py` | 16 |
| `indexer.py` | 15 |
| `policy_enforcer.py` | 15 |
| `rag.py` | 14 |
| `grounding.py` | 13 |
| `graph_discovery.py` | 13 |
| `facts_ledger.py` | 13 |
| `auth.py` | 13 |
| `routes_chat.py` | 12 |
| `summarizer.py` | 12 |
| `session.py` | 12 |
| `observability.py` | 11 |
| `__init__.py` | 11 |
| `routes_summarize.py` | 11 |
| `client.py` | 11 |
| `routes_auth.py` | 10 |
| `health.py` | 10 |
| `routes_search.py` | 10 |
| `query_classifier.py` | 10 |
| `loader.py` | 10 |
| `routes_admin.py` | 9 |
| `email_processing.py` | 9 |
| `text_extraction.py` | 9 |
| `exceptions.py` | 9 |
| `routes_draft.py` | 9 |
| `routes_answer.py` | 9 |
| `vector_search.py` | 9 |
| `audit_config.py` | 9 |
| `guardrails_client.py` | 8 |
| `atomic_io.py` | 8 |
| `results.py` | 8 |
| `dependencies.py` | 8 |
| `injection_defense.py` | 8 |
| `states.py` | 7 |
| `defenses.py` | 6 |
| `config.py` | 5 |
| `graphs.py` | 5 |
| `types.py` | 4 |
| `redis.py` | 4 |
| `search.py` | 4 |
| `queue_registry.py` | 3 |
| `models.py` | 3 |
| `redacted.py` | 3 |
| `context.py` | 2 |

## Failed Files (Max Retries)

- `main.py`
- `routes_ingest.py`
- `graph.py`
- `query_expansion.py`
- `nodes.py`
- `hybrid_search.py`
- `graph_search.py`
- `async_cache.py`
- `filter_resolution.py`
- `reranking.py`
- `filters.py`
- `fts_search.py`
- `vector_store.py`
- `validators.py`
- `models.py`
- `chunker.py`
