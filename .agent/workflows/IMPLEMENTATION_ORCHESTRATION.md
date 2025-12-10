---
description: Outlook Cortex Implementation QS
---

# 🚀 Outlook Cortex Implementation Orchestration

> **Status:** Active
> **Blueprint Reference:** `OUTLOOK_CORTEX_BLUEPRINT_v3.3.md`

---

## 🛑 PROTOCOL & RULES (NON-NEGOTIABLE)

### 1. 🔐 Credentials & Environment
*   **Source of Truth:** All credentials (DB, Redis, Spaces, LLM Gateway) are in the `.env` file.
*   **Injection:** Load via project-level environment variables or `doctl`.
*   **Rule:** **NEVER MOCK CREDENTIALS**. Always use live connection strings for validation.

### 2. ✅ The "Validation First" Gate
*   **Before moving to Step N+1**: The Agentic Coder **must** validate Step N.
*   **Validation Standard:**
    1.  Code is implemented.
    2.  Unit tests pass.
    3.  **Live E2E Verification succeeded** (e.g., a real row was written to Postgres, a real file was parsed, a real embedding was generated).
*   **Failure Protocol:** If validation fails, stop. Fix Step N. Do not skip.

### 3. 🛡️ Anti-Drift & Best Practices
*   **No Mocks:** Do not create mock implementations for placeholders. If a module is needed, implement the skeleton with typed interfaces.
*   **Error Handling:** No generic `try/except: pass`. Use `raise` or typed `CortexError`.
*   **Directory Structure:** Strictly follow Blueprint `§2.2`. Do not invent folders.

---

## 📋 MASTER IMPLEMENTATION CHECKLIST

### Phase 1: Infrastructure & Foundation (`§0` - `§4`)
*Goal: A connected, observable environment with a valid database schema.*

- [ ] **1.1 Workspace Setup**
    - [ ] Initialize repo structure per `§2.2`.
    - [ ] Setup `pyproject.toml` with dependencies (`fastapi`, `sqlalchemy`, `pgvector`, `alembic`, `pydantic`, `opentelemetry`, `langgraph`).
    - [ ] **Validation:** Run `ls -R` and verify exact match with Blueprint.

- [ ] **1.2 Configuration Engine**
    - [ ] Implement `cortex.config.models` (Pydantic v2).
    - [ ] Implement `cortex.config.loader` (Singleton).
    - [ ] **Validation:** Script `verify_config.py` loads config and prints (masked) DB URL from `.env`.

- [ ] **1.3 Observability Layer**
    - [ ] Implement `cortex.observability` (`init`, `@trace_operation`).
    - [ ] **Validation:** Run a script with a decorated function; verify JSON log output contains `trace_id`.

- [ ] **1.4 Database & Schema**
    - [ ] Configure `cortex.db` session/engine.
    - [ ] Create Migration `001`: Initial Tables (`threads`, `messages`, `attachments`, `chunks`).
    - [ ] Create Migration `002`: FTS (`tsv` columns + triggers).
    - [ ] Create Migration `003`: Vectors (`hnsw` index).
    - [ ] Create Migration `004`: RLS Policies.
    - [ ] **Validation:** Run `alembic upgrade head`. Connect via `psql`, insert a dummy row, verify RLS isolation and Trigger execution.

### Phase 2: Ingestion Pipeline (`§5` - `§6`)
*Goal: Raw files -> Cleaned, Redacted, Normalized Data in Postgres.*

- [ ] **2.1 Security Validators**
    - [ ] Implement `cortex.security.validators` (Path traversal, Ext check).
    - [ ] **Validation:** Unit test rejecting `../../etc/passwd`.

- [ ] **2.2 Manifest Handling (B1)**
    - [ ] Implement `core_manifest.py` & `validation.py`.
    - [ ] **Validation:** Run `scan_and_refresh` on a test folder. Verify `manifest.json` generation.

- [ ] **2.3 Text Processing & PII**
    - [ ] Implement `text_preprocessor.py` & `pii.py`.
    - [ ] **Validation:** Process string "Call 555-0199". Assert output contains `<<PHONE>>`.

- [ ] **2.4 Parsers & Extractors**
    - [ ] Implement `parser_email.py` (EML/MBOX).
    - [ ] Implement `attachments.py` & `text_extraction.py`.
    - [ ] **Validation:** Parse complex EML with attachment. Verify attachment text extraction.

- [ ] **2.5 Ingestion Job Runner**
    - [ ] Implement `mailroom.py` and `writer.py`.
    - [ ] **Live Validation:** Run ingestion job on a real EML file. Query DB to confirm `messages` and `chunks` records exist.

### Phase 3: Intelligence Runtime (`§7`)
*Goal: Resilient LLM connectivity and Embedding generation.*

- [ ] **3.1 Chunking Engine**
    - [ ] Implement `chunker.py` (Quote masking, overlap).
    - [ ] **Validation:** Chunk a long conversation. Verify no split mid-sentence and correct metadata.

- [ ] **3.2 LLM Runtime (DOKS)**
    - [ ] Implement `cortex.llm.runtime` (Retries, Circuit Breaker).
    - [ ] Implement `cortex.llm.doks_scaler` (DigitalOcean logic).
    - [ ] Implement `cortex.llm.client` shim.
    - [ ] **Live Validation:** Run `client.embed_texts(["test"])` against live endpoint. Verify 3840-dim (or config-match) vector returned.

- [ ] **3.3 Async Workers**
    - [ ] Implement `workers/` structure and handlers (`reindex`, `ingest`).
    - [ ] **Live Validation:** Push job to Redis. Verify Worker logs receipt and success.

### Phase 4: Retrieval & RAG (`§8` - `§10`)
*Goal: Semantic Search and Agentic Workflows.*

- [ ] **4.1 Search Engines**
    - [ ] Implement `fts_search.py` and `vector_search.py`.
    - [ ] Implement `hybrid_search.py` (RRF Fusion).
    - [ ] **Live Validation:** Ingest unique term "Zanzibar". Query "Zanzibar" via Hybrid search. Verify hit.

- [ ] **4.2 Agentic Graphs**
    - [ ] Implement `graph_answer_question`.
    - [ ] Implement `graph_draft_email`.
    - [ ] Implement `graph_summarize_thread`.
    - [ ] **Live Validation:** Run `answer_question` on ingested data. Verify grounded response.

- [ ] **4.3 API Layer**
    - [ ] Implement FastAPI Routes (`/search`, `/answer`, `/draft`).
    - [ ] **Live Validation:** `curl POST /answer` and receive JSON 200 OK.

### Phase 5: Production Readiness (`§11` - `§13`)
*Goal: Safety, CLI tools, and Deployment artifacts.*

- [ ] **5.1 Safety & Guardrails**
    - [ ] Implement `injection_defense.py` & `policy_enforcer.py`.
    - [ ] **Validation:** Verify prompt with "Ignore instructions" is stripped/sanitized.

- [ ] **5.2 Doctor & CLI**
    - [ ] Implement `cortex_cli` and `cmd_doctor.py`.
    - [ ] **Live Validation:** Run `cortex doctor --check-all`. Must return Exit Code 0.

---

## 📝 AGENTIC LEDGER (Post-Implementation Log)

*Agentic Coder: Append new rows here after completing a Phase or major component.*

| Date | Component | Status | Verified (Live)? | Edge Cases / Concerns / Technical Debt |
| :--- | :--- | :--- | :--- | :--- |
| | | | | |
| | | | | |
| | | | | |
