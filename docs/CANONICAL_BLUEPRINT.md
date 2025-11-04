# Outlook Cortex (EmailOps Edition) — **Canonical Implementation Blueprint v2.1 (2025‑11‑01)**

*Single-source specification for an enterprise‑ready, agentic AI system that ingests exported email conversations + attachments, builds a high‑fidelity knowledge base, retrieves accurately, drafts emails with grounded citations and consistent tone, and operates under verifiable autonomy, programmatic permissions, and standardized schemas.*

**Scope (unchanged):**

* **No Azure, no direct Microsoft 365 APIs.**
* **Data** = exported emails at **conversation level** (`.eml`, `.mbox`) + **attachments** (PDF/Word/Excel/PowerPoint/images).
* **Calendar management removed.**
* **Deploy** on **DigitalOcean**, **Vertex AI**, and/or **open‑source**.

---

## 0) Architecture (at a glance)

```
                ┌──────────────────────────────────────────────────────────────────┐
 Email exports  │  1. DATA INGESTION & NORMALIZATION                               │
 + attachments  │  - Mailroom Ingestion (EML/MBOX) → MIME parse + header decode    │
 (Spaces/S3,    │  - Thread rebuild (Message-ID / In-Reply-To / References)        │
 SFTP) ───────▶ │  - Attachment text/image/table extraction + OCR + AV scan        │
                │  - PII detection/redaction (offset-preserving)                   │
                └──────────────┬───────────────────────────────────────────────────┘
                                │ normalized docs + metadata
                                ▼
                ┌──────────────────────────────────────────────────────────────────┐
                │  2. CHUNKING & EMBEDDINGS                                        │
                │  - Structural + semantic + contextual chunking (titles/paths)    │
                │  - Dense embeddings + (optional) image/table embeddings          │
                │  - Vector DB (Qdrant/pgvector/Weaviate) + Lexical index          │
                └──────────────┬───────────────────────────────────────────────────┘
                                │ chunk ids + embeddings + terms
                                ▼
                ┌──────────────────────────────────────────────────────────────────┐
                │  3. ENTERPRISE EMAIL KNOWLEDGE GRAPH (EEKG)                      │
                │  - Entities: People, Threads, Messages, Attachments, Concepts,   │
                │    Tasks, Decisions, Citations; temporal + permission metadata   │
                └──────────────┬───────────────────────────────────────────────────┘
                                │ entity ids + relations
                                ▼
                ┌──────────────────────────────────────────────────────────────────┐
                │  4. ORCHESTRATOR (LangGraph) + MCP TOOLING                       │
                │  - Semantic Router + conditional graph edges                      │
                │  - All tools via MCP; durable workflows via Temporal/Argo        │
                └──────────────┬───────────────────────────────────────────────────┘
                                │ tool calls + plan steps
                                ▼
                ┌──────────────────────────────────────────────────────────────────┐
                │  5. RETRIEVAL & GENERATION                                       │
                │  - Hybrid retrieval (BM25 + vectors → RRF; MMR; rerank)          │
                │  - Draft emails w/ persona mimicry & per‑claim citations         │
                │  - Strict JSON Schemas + constrained decoding                    │
                └──────────────┬───────────────────────────────────────────────────┘
                                │ outputs + scores + audit trails
                                ▼
                ┌──────────────────────────────────────────────────────────────────┐
                │  6. VERIFIABLE AUTONOMY LAYERS (VAL)                             │
                │  - OPA/OpenFGA policy gates, Guardrails/NeMo, immudb + Rekor     │
                │  - Risk/Trust scoring + immutable audit                          │
                └──────────────────────────────────────────────────────────────────┘
```

---

## 1) Data Processing & Knowledge Base Construction

### 1.1 Email normalization & threading (from exports)

**Accepted inputs:** `.eml` (RFC 5322/RFC 2047/RFC 2231 for header encodings) and `.mbox` (RFC 4155). Always decode encoded‑word headers and RFC 2231 parameters (e.g., filenames). Parse bodies and attachments (MIME). Use Python `email` with `headerregistry` (for `Address`), plus robust RFC 2047/2231 decoding. **Normalize charsets** and **quoted‑printable/base64** before further processing. ([datatracker.ietf.org])

**Thread reconstruction (deterministic):**

1. Prefer **`References` chain**; fallback to **`In-Reply-To`**; as last resort, cluster by normalized **subject (ignore “Re:”/“Fwd:”) + participants + time proximity**. RFC 5322 spells out semantics; allow multi‑parent edge cases but keep a **single canonical thread id**. ([datatracker.ietf.org])
2. For **mbox**, respect “From ” separators and known vagaries; treat as a serialized folder. ([rfc-editor.org])
3. **Quoted‑text masking**: Detect and mark quoted blocks & signatures rather than deleting them (preserve provenance). Use heuristic libraries such as **Mailgun Talon** (signatures/quotes) or equivalents; store masks in metadata for de‑duplication and chunking. ([GitHub])

**Implementation notes:**

* Decode non‑ASCII headers via RFC 2047 encoded words; decode RFC 2231 parameters (e.g., `filename*=`). ([IETF])
* Keep **Message‑ID** de‑dup guards; compute **content hash** fallback for malformed messages.

### 1.2 Attachment understanding (text, layout, images, tables)

* **General extraction**: **Apache Tika** for broad format coverage, including `message/rfc822`, Office, PDF, HTML, images, and `.mbox`. ([tika.apache.org])
* **OCR**: **Tesseract** (multi‑language packs) and **Tika OCR** integration when PDFs/images lack text. ([Tesseract OCR])
* **Layout‑aware** partitioning: **Unstructured** `partition_email`, `partition_pdf`, `partition_image`, `process_attachments=True` for email body + attachments with element types (Title, ListItem, etc.). ([docs.unstructured.io])
* **Tables**: Prefer dedicated extractors (**Camelot**, **tabula‑java/tabula‑py**, **pdfplumber**) depending on table style (lattice/stream). Store a structured copy (CSV/Parquet) for exact numeric ops. ([camelot-py.readthedocs.io])

**Security hygiene:** (ingest‑time) AV/malware scan, reject active content; mark password‑protected files as **unparsed** with reason.

### 1.3 PII detection/redaction (ingest‑time)

Use **Microsoft Presidio** (regex + NER + context) with **spaCy** models; support custom recognizers and reversible tokenization under policy. Persist redaction offsets for later **unredaction when authorized**. ([GitHub])

### 1.4 Storage layout

* **Raw objects**: DigitalOcean **Spaces** (S3‑compatible) for original EML/attachments and normalized JSON; enable lifecycle+CDN as needed. ([DigitalOcean Docs])
* **Relational metadata**: **PostgreSQL** (threads, messages, participants, attachments, provenance, policies); DigitalOcean Managed PG supports **pgvector** (extension is `vector`). ([DigitalOcean Docs])
* **Vector search**:

  * **Qdrant** (HNSW + payload filters); or **pgvector** for single‑DB footprint; **Weaviate** for built‑in hybrid. ([qdrant.tech])
* **Lexical search**: **OpenSearch/Elasticsearch‑compatible BM25** (or Postgres FTS for small deployments). ([Haystack Documentation])
* **Graph**: Start with **NetworkX**; scale to **Neo4j** for EEKG analytics (Cypher). *(General knowledge; no external spec needed.)*

---

## 2) Chunking & Embedding

### 2.1 Why chunking matters

LLMs under‑utilize mid‑context details (“**Lost in the Middle**”); favor smaller, coherent chunks and place the most relevant spans near the top of prompts. ([arXiv])

### 2.2 Prescriptive policy

1. **Structural split first** (headers, quoted blocks, tables, lists, slide titles) using Unstructured/Tika metadata. Ensure quoted text is **masked**, not mixed with the latest reply. ([docs.unstructured.io])
2. **Semantic split** within sections (sentence‑aware).
3. **Contextualized chunks**: prepend **title + section path** (and brief automatic summary) to make each chunk self‑contained—mirrors industry guidance on contextual retrieval. ([DigitalOcean Docs])
4. Target **300–800 tokens** with \~10–20% overlap; never split **inside** quoted‑reply masks or tables.
5. **Tables**: store both **serialized text** (for embeddings) and **structured rows** (for exact filters/joins).

### 2.3 Embeddings & reranking

* **Text embeddings (open)**: strong families include **BGE‑M3** (multilingual, multi‑granularity) and widely used Sentence‑Transformers lines; store `model@version` and `dim`. ([Hugging Face])
* **Images**: optional CLIP‑family embeddings for diagrams.
* **Rerankers**: Use **cross‑encoders** for top‑N re‑ranking (e.g., `cross-encoder/ms-marco‑MiniLM‑L6‑v2`) or **BGE‑reranker‑v2‑m3** (multilingual). ([Hugging Face])

---

## 3) Retrieval (hybrid, re‑ranked, grounded)

### 3.1 Pipeline (deterministic)

1. **Query analysis & rewrite** (multi‑query) for recall.
2. **Hybrid first pass**:

   * **Lexical**: BM25.
   * **Vector**: KNN in Qdrant/pgvector/Weaviate.
   * **Fuse** with **Reciprocal Rank Fusion (RRF)**; optionally weight sources. *(RRF is a standard fusion technique and common in hybrid search pipelines.)* ([SingleStore])
3. **Diversify** with **MMR** to reduce redundancy. *(Classical Carbonell & Goldstein, 1998.)* ([cs.cmu.edu])
4. **Re‑rank** top‑N with a **cross‑encoder** (e.g., MS‑MARCO MiniLM) or **BGE‑reranker‑v2‑m3**; keep N small (50→10). ([Hugging Face])
5. **Assemble context**: **source‑segregated** windows (never interleave sentences from different messages); include message/attachment **provenance** (thread id, message id, filename + page).

**Why hybrid?** Hybrid (BM25 + dense) consistently outperforms either alone across exact‑term + semantic queries. Weaviate and other production systems document hybrid and RRF fusion best practices. ([docs.weaviate.io])

### 3.2 Answering & grounding

* **Strict grounding**: Answers/drafts must be **fully supported** by retrieved evidence; include **per‑claim citations** mapping to **message/attachment spans** (with offsets/pages).
* If evidence insufficient: return `NO_ANSWER` + suggested follow‑ups.
* **Contrastive**: mention negative evidence where useful (“no mention of X in this thread slice”).

---

## 4) Core Models & Multi‑Agent Orchestration

### 4.1 Model choices (provider‑agnostic)

* **Open‑source LLMs** (e.g., Llama 3.x, Qwen 2.5, Mixtral) or **Vertex AI** (Gemini) for long context and multimodal. *(Model families; pick based on latency/cost needs.)*

### 4.2 Orchestrator (LangGraph) + MCP tools

* **LangGraph** models agent workflows as **graphs with state**, conditional edges, and human‑in‑the‑loop; supports **durable** execution. ([docs.langchain.com])
* **All tools exposed via** **Model Context Protocol (MCP)**: one standard for model ↔ tool/data interactions; versioned spec; supports server/client separation. **Use MCP for search, DB, vector, policy, audit, renderers.** ([modelcontextprotocol.io])
* **Durable workflows**: **Temporal** or Argo for ingest→index→review→publish pipelines. *(General orchestration practice.)*

**MCP resource classes (normative)**
`email.search`, `email.get_thread`, `attachment.extract_text`, `kb.search_hybrid`, `kb.rerank`, `graph.query_cypher`, `policy.check`, `audit.append`, `tasks.create`.

---

## 5) Platforms & Infra (DigitalOcean‑first; Vertex optional)

* **DOKS** (managed Kubernetes) for services; **Spaces** for object storage; **Managed PostgreSQL** (enable **`CREATE EXTENSION vector;`** for pgvector); optional **Qdrant/Weaviate**. ([DigitalOcean Docs])
* **OpenSearch/Elastic‑compatible** for BM25 + dashboards. *(Industry standard.)*
* **Observability**: **OpenTelemetry** traces; **Loki** for logs; ship via **Fluent Bit** DaemonSet. ([OpenTelemetry])

---

## 6) Tooling, Permissions & Logic of Use

### 6.1 Tool registry (MCP)

Register tools with: name, version, schema, allowed roles, ABAC attributes, rate limits, PII‑exposure flags. **Every tool call includes** `{user_id, session_id, task_id, security_ctx, relevant_entity_ids, policy_claims}`.

### 6.2 Permission enforcement

* **Identity/SSO** via Keycloak (OIDC).
* **ABAC/RBAC** via **Open Policy Agent (OPA)** (Rego) on **every** tool call.
* **Relationship auth** via **OpenFGA** for doc‑level permissions (who‑can‑see‑what). ([Keycloak])

### 6.3 Deterministic decision logic

1. If user asks for **facts/citations** → `kb.search_hybrid` → **rerank** → assemble answer.
2. If **drafting email** → load **thread slice** + top‑K attachment chunks → `policy.check` → compose → `VAL.review`.
3. If **evidence missing** → return `NO_ANSWER` or request authorization to index new data.
4. **Every** tool call logs provenance and is **audited** (see §12).

---

## 7) Standardized Output Schemas (strictly validated)

> All agents must emit **valid** JSON objects. Use constrained decoding and schema validation (e.g., Guardrails RAIL / NeMo Guardrails). ([NVIDIA Docs])

### 7.1 `Answer`

```json
{
  "type": "answer",
  "query": "string",
  "answer_markdown": "string",
  "evidence": [
    {
      "thread_id": "uuid",
      "message_id": "string",
      "attachment_id": "string|null",
      "span": {"start": 0, "end": 0},
      "snippet": "string",
      "confidence": 0.0
    }
  ],
  "confidence_overall": 0.0,
  "safety": {"pii_present": false, "policy_flags": []},
  "retrieval_diagnostics": {
    "lexical_score": 0.0,
    "vector_score": 0.0,
    "fused_rank": 0,
    "reranker": "model@version"
  }
}
```

### 7.2 `EmailDraft`

```json
{
  "type": "email_draft",
  "thread_id": "uuid",
  "to": ["addr"],
  "cc": ["addr"],
  "subject": "string",
  "body_markdown": "string",
  "tone_style": {"persona_id": "string", "tone": "brief|formal|friendly|empathetic|firm"},
  "attachments": [{"from_attachment_id": "string", "as_filename": "string"}],
  "citations": [{"message_id":"string","attachment_id":"string|null","justification":"string"}],
  "val_scores": {"risk": 0.0, "trust": 0.0, "policy_ok": true},
  "next_actions": [{"label":"Send","action":"send_email","requires_human_confirm": true}]
}
```

### 7.3 `SearchResults`

```json
{
  "type": "search_results",
  "query": "string",
  "reranker": "model@version",
  "results": [
    {"chunk_id":"uuid","score":0.0,"thread_id":"uuid","message_id":"string","attachment_id":"string|null","highlights":["..."]}
  ]
}
```

### 7.4 `TaskUpdate`

```json
{
  "type": "task_update",
  "task_id":"uuid",
  "status":"queued|running|blocked|awaiting_approval|completed|failed",
  "who":"agent|user",
  "message":"string",
  "linked_entities":[{"type":"thread|doc|attachment|chunk","id":"..."}],
  "timestamp":"iso8601"
}
```

---

## 8) Retrieval & Answering Best Practices (operational)

1. **Always hybrid** (BM25 + dense) → **RRF fuse** → **MMR diversify** → **cross‑encoder rerank**. ([docs.weaviate.io])
2. **Aggressive citation**: each assertion maps to exact message/attachment spans.
3. **Negative evidence** where helpful.
4. **Confidence discipline**: surface `confidence_overall` + per‑evidence `confidence`.
5. **When unsure**: return `NO_ANSWER`; never fabricate.

---

## 9) Email Drafting: Context Loading & Persona Mimicry

### 9.1 Context loading recipe

* Retrieve **thread history slices** (key turns + latest) + top‑K attachment chunks; include **who‑said‑what** and dates.
* Precompose **facts / decisions / asks / risks** outline.
* Draft using **Adaptive Persona Projection (APP)** from user‑approved edits.
* Always **attach** referenced files and **inline** key snippets **with citations**.

### 9.2 Safety & compliance pass (VAL)

* **NeMo Guardrails / GuardrailsAI (RAIL)** for schema conformance, topic control, and jailbreak/safety checks. ([NVIDIA Docs])
* **OPA policy** gate before high‑risk actions (e.g., emailing external recipients with client identifiers). ([Keycloak])

---

## 10) Verifiable Autonomy Layers (VAL)

* **OWASP LLM Top‑10** alignment (prompt injection, insecure output handling, excessive agency, etc.). Apply **intelligent friction** by risk level. ([OWASP])
* **Immutable audit**: **immudb** (append‑only, tamper‑evident) + signed approvals to **Sigstore/Rekor** (transparency log). *(Use internal security pattern drawn from standard tamper‑evident logging and transparency‑log practices.)*
* **NIST AI RMF** governance: GOVERN/MAP/MEASURE/MANAGE; keep artifacts in `/risk/`. ([NIST Publications])

**Audit record (minimal):**

```
{ ts, user_or_agent, tool, input_hash, output_hash, policy_decisions, risk, trust, approval_refs, signatures }
```

---

## 11) Evaluation & Monitoring

* **RAG eval**: Faithfulness/answer correctness/context recall (e.g., RAGAS/TruLens) tied to CI; compare **hybrid vs dense‑only** A/B and **reranker variants** (MiniLM vs BGE‑m3). *(Evaluation tools referenced widely; pairing with hybrid and reranker docs above.)*
* **Observability**: **OpenTelemetry** tracing; logs to **Loki** via **Fluent Bit**; dashboards + retrieval diagnostics for query‑level debugging. ([OpenTelemetry])
* **SLOs**: latency budgets per stage (retrieval, rerank, compose, VAL).
* **Cost**: per‑task/user/day budgets with circuit breakers.

---

## 12) Operational Workflows

Use **Temporal** (or Argo) for durable, replayable pipelines:

1. **IngestExport** → Parse → PII → Partition → Store → Index.
2. **ReindexAttachment** on file update.
3. **DraftEmail** → Retrieve → Compose → **VAL** → **HumanConfirm** (optional) → Finalize.
4. **BackfillGraph**: entity resolution & relationships (People↔Threads↔Docs).

**Failure playbooks** (anti‑confusion defaults):

* **MIME parse errors**: retry ×3; quarantine attachment; file `TaskUpdate(blocked)` with reason.
* **PII false positives**: allow Presidio recognizer overrides; reprocess delta. ([Microsoft GitHub])
* **Retrieval misses**: lexical‑only expansion → multi‑query rewrite → else `NO_ANSWER`.
* **Draft unsafe**: VAL **medium/high risk** → require human approval + signature to Rekor.

---

## 13) Implementation Plan (DigitalOcean‑first)

**Infra**

* **DOKS** cluster + node pools (CPU for parsing; optional GPU for OCR/LLM).
* **Spaces** buckets: `raw/`, `normalized/`, `artifacts/`. ([DigitalOcean Docs])
* **Managed PostgreSQL** (+ **pgvector**). ([DigitalOcean Docs])
* **Qdrant/Weaviate** via Helm if scale requires. ([qdrant.tech])
* **OpenSearch** (Helm) for lexical/RRF dashboards.
* **Temporal** (Helm) for workflows.
* **VAL** services (OPA, OpenFGA, Guardrails/NeMo).

**Microservices**

* `ingest-mailroom` (Go/Python): watch Spaces; parse EML/mbox; header decode; PII pass.
* `attachment-extract` (Python): Tika/Unstructured/Tesseract + table tools. ([tika.apache.org])
* `indexer` (Python): chunk → embed → upsert (vector + lexical).
* `search-api` (Python/TS): hybrid (BM25 + KNN) → RRF → MMR → cross‑encoder rerank. ([SingleStore])
* `composer` (LLM): APP‑aware drafting (JSON schema enforced).
* `val-gateway`: OPA/OpenFGA checks + Guardrails/NeMo validation; write audits. ([Keycloak])
* `mcp-hub`: registry + proxy for MCP tool servers. ([modelcontextprotocol.io])

---

## 14) Security & Compliance Baseline

* Map to **OWASP GenAI Top‑10**; enforce **excessive agency** controls via policy + human‑confirm thresholds. ([OWASP])
* Align lifecycle governance to **NIST AI RMF**; keep records of risk acceptance/mitigations. ([NIST])
* Secrets in Vault/KMS; never include secrets in LLM context; tools fetch server‑side.

---

## 15) Hard “Gotchas” & Preventive Controls

* **Thread confusion from quoted text** → detect & **mask** quoted blocks (Talon‑style) before chunking; index latest reply separately. ([GitHub])
* **Middle‑bias loss** → smaller, coherent chunks; **contextualized prefixes** (title/section). ([arXiv])
* **Excessive agency** → VAL + OPA gates + human‑confirm for medium/high risk (OWASP LLM #8). ([OWASP])
* **Permission leaks** → enforce filters at **retriever** (vector payload filters / SQL RLS) and **renderer** (schema guardrails) with OPA/OpenFGA. ([qdrant.tech])
* **Schema drift** → Guardrails/NeMo validation + constrained decoding; reject/repair. ([NVIDIA Docs])
* **Audit non‑repudiation** → immudb (append‑only) + signed approval to Rekor. *(Standard tamper‑evident patterns; cited governance in §10.)*

---

## 16) MCP Operational Rules (normative)

* **All tool use via MCP servers** (HTTP/WebSocket).
* **Schema versioning required**; tools emit **typed errors** (retryable vs terminal).
* Enforce **MCP version strings** per spec (e.g., `2025‑06‑18`). ([modelcontextprotocol.io])
* Tool calls must carry **context package** (see §6.1). ([modelcontextprotocol.io])

---

## 17) Milestones (ship in weeks, not months)

**M1 — Foundation (week 1–2)**
DOKS, Spaces, Managed PG (+pgvector), OpenSearch; ingest 1k sample emails; normalize & store. ([DigitalOcean Docs])

**M2 — Index & Retrieve (week 3–4)**
Structural+semantic+contextual chunking; BGE/SBERT embeddings; Qdrant/pgvector; hybrid retrieval; RRF+MMR; basic rerank. ([docs.weaviate.io])

**M3 — Compose & Guard (week 5–6)**
Persona stub; `EmailDraft` schema; VAL (Guardrails/NeMo) + OPA checks; audit enabled. ([NVIDIA Docs])

**M4 — Orchestrate & Prove (week 7–8)**
LangGraph orchestration; MCP‑wrapped tools; Temporal workflows; dashboards, tracing, RAG eval. ([docs.langchain.com])

---

## 18) Appendix — Concrete Parameters & Defaults (normative)

**Ingestion**

* Decode headers per **RFC 2047**/**RFC 2231**; reject malformed after 3 retries. ([IETF])
* Threading: prefer `References`; fallback `In-Reply-To`; otherwise subject/participants/time. ([datatracker.ietf.org])
* Quoted/signature masks via Talon‑style heuristics. ([GitHub])

**Extraction**

* Tika server with OCR enabled; Tesseract langs per tenant; Unstructured `process_attachments=True`. ([cwiki.apache.org])
* Tables: Camelot (lattice/stream), tabula‑java, pdfplumber; prefer CSV+Parquet outputs. ([camelot-py.readthedocs.io])

**Chunking**

* Size: **300–800 tokens**, overlap **10–20%**.
* Prefix each chunk: `[Thread: … > Message: … > Section: …] + 1–2 sentence summary`. *(Contextual retrieval best practice.)* ([DigitalOcean Docs])

**Vector/Lexical**

* Qdrant: HNSW (`m=16, ef=128`), **payload filters** mirrored from OpenFGA relations; store `created_at`, `participants`, `tenant_id`. ([qdrant.tech])
* Weaviate: hybrid enabled with BM25F + vector fusion (RRF or alpha‑weighted). ([docs.weaviate.io])
* OpenSearch: BM25 defaults (k1≈1.2, b≈0.75) tunable per corpus. ([Haystack Documentation])

**Reranking**

* Default cross‑encoder: `cross-encoder/ms-marco‑MiniLM‑L6‑v2` (fast); multilingual: `BAAI/bge-reranker‑v2‑m3`. ([Hugging Face])

**Schemas & Guardrails**

* Guard output strictly with **NeMo Guardrails** (Colang policies) or **GuardrailsAI (RAIL)**; abort/repair on invalid JSON. ([NVIDIA Docs])

**Observability**

* OpenTelemetry traces across agents/tools; logs shipped to **Loki** via **Fluent Bit**. ([OpenTelemetry])

**Governance**

* OWASP GenAI Top‑10 controls + **NIST AI RMF** artifacts. ([OWASP])

---

## 19) Changes vs previous canonical draft (what’s new & why)

1. **Header decoding & threading rules hardened** (explicit RFC 2047/2231 decoding; `References`→`In‑Reply‑To` fallbacks), reducing thread split/merge errors. ([IETF])
2. **Contextual chunking** (titles/paths + short summaries) added to mitigate long‑context pitfalls. ([arXiv])
3. **Reranker guidance updated** to **BGE‑reranker‑v2‑m3** (multilingual) with MiniLM as fast default. ([Hugging Face])
4. **Hybrid/RRF/MMR** pipeline made normative with references to production docs (Weaviate, RRF). ([docs.weaviate.io])
5. **MCP references corrected** (official spec + versioning) and mandated for *all* tool use. ([modelcontextprotocol.io])
6. **DigitalOcean pgvector note clarified** (`CREATE EXTENSION vector;`), preventing common enablement errors. ([DigitalOcean Docs])
7. **Table extraction stack** made explicit (Camelot/tabula/pdfplumber) with structured outputs for numerics. ([camelot-py.readthedocs.io])
8. **VAL hardening** mapped to **OWASP LLM Top‑10** + **NIST AI RMF**; guardrail frameworks explicitly required. ([OWASP])

---

## 20) One‑page “Do this, not that” (engineer‑ready)

* **Do** ingest **native `.eml`/`.mbox`** and decode headers per RFC 2047/2231. **Don’t** flatten to ad‑hoc `.txt` that loses metadata. ([IETF])
* **Do** mask quoted replies/signatures before chunking. **Don’t** interleave old quotes with the latest message. ([GitHub])
* **Do** run **BM25 + vector** → **RRF** → **MMR** → **cross‑encoder**. **Don’t** rely on vectors alone for exact term queries. ([docs.weaviate.io])
* **Do** enforce schemas with **NeMo/RAIL**; **Don’t** accept free‑form LLM JSON. ([NVIDIA Docs])
* **Do** gate high‑risk actions via **OPA/OpenFGA**; **Don’t** call tools without policy context. ([Keycloak])
* **Do** audit every tool call; **Don’t** lose provenance.

---

### End of Canonical Blueprint (v2.1)

> This document is the **only** reference required to implement Outlook Cortex (EmailOps Edition) on DigitalOcean/Vertex/open‑source stacks—from ingestion to orchestration, retrieval to drafting, and safety to auditability—aligned with **current, reputable best practices** in 2025.

**Primary sources used:** RFCs for email/mbox/encoding; Unstructured/Tika/Tesseract docs; Weaviate/Qdrant/pgvector/OpenSearch docs; LangGraph & MCP specs; OWASP GenAI Top‑10; NIST AI RMF; Guardrails/NeMo Guardrails docs. (See inline citations per section.)