# Agent Execution Checklist

This document defines the **sequential, non‑skippable checklist** for the Agentic AI Coder.

> **Canonical source of truth:**  
> `docs/CANONICAL_BLUEPRINT.md` is the single source of truth for architecture, layout, schemas, and behavior.  
> If code/docs disagree with the blueprint, assume the blueprint is correct and change code/docs to match.  
> When intentionally changing schema, behavior, or prompts, update the relevant blueprint section **in the same change**.

> **Credentials note:**  
> All credentials (DigitalOcean Spaces/S3, Managed DB, DOKS, etc.) are available via the `.env` file.  
> They may also be available as project‑level environment variables or via `doctl` from the terminal.  
> **Never** commit secrets to the repo.

---

## 0. Global Non‑Negotiables

The Agentic AI Coder MUST obey all of the following:

1. **Blueprint as law**
   - Use `docs/CANONICAL_BLUEPRINT.md` as the canonical reference for:
     - Service boundaries  
     - Layout and directory structure  
     - Schemas and data contracts  
     - Behavioral expectations and prompts  
   - If there is a conflict between code and the blueprint:
     - Treat the blueprint as correct.
     - Change the code/docs to match.
     - If the change is intentional, update the blueprint in the **same** commit/merge.

2. **Ledger & verification before new work**
   - The ledger file is `docs/IMPLEMENTATION_LEDGER.md`.
   - On every run, **before** implementing anything new, the agent must:
     1. Open the ledger.
     2. For each step with status `completed_pending_verification`:
        - Re‑check all “Observable invariants” in code.
        - Re‑run all tests listed for that step.
     3. If everything matches, set step status to `verified`.
     4. If anything does not match, set status to `drift_detected`, fix it, and re‑test.
   - **The agent MUST NOT start a new checklist step while any step is in `drift_detected`.**

3. **No mock implementations or placeholders**
   - **No mock integrations** for services where live infrastructure and credentials exist.
   - **No placeholder implementations** for core code paths.
   - **No “TODO implement later”** in production paths. TODOs may exist only for non‑critical niceties, and must be clearly marked as such.

4. **Error handling constraints**
   - No `try/except` blocks that silently swallow errors.
   - `except` blocks must either:
     - Re‑raise the same error or a wrapped error; or
     - Map the error to a clear, explicit failure path (HTTP error, domain error, etc.) and abort the flow.
   - Logging must not leak secrets, credentials, or raw email/attachment bodies.

5. **Live integration tests after each step**
   - After each checklist step:
     - Run tests using **live credentials** where applicable.
     - Include **edge cases** as described in each step.
   - If tests fail for a step, **do not** proceed to the next step until they pass.
   - Record all commands and outcomes in the ledger.

6. **Repo conventions**
   - Python‑first; match existing patterns and module boundaries.
   - Use type hints where practical.
   - ASCII‑only.
   - Use existing config loaders, models, and tool shims.
   - No direct access to external services outside designated shims.
   - Respect `pre-commit` tools (black, ruff, isort, mypy).

7. **No new top‑level structures**
   - Do not introduce new top‑level services, apps, or directory trees unless explicitly described in `CANONICAL_BLUEPRINT.md`.
   - Integrate changes into existing structures.

---

## 1. Agent Run Loop (Every Invocation)

On each run, the Agentic AI Coder must:

1. **Load Canonical Context**
   - Open `docs/CANONICAL_BLUEPRINT.md`.
   - Open `docs/IMPLEMENTATION_LEDGER.md`.

2. **Verify previous work before new work**
   - For each step with status `completed_pending_verification`:
     - Verify observable invariants in the code.
     - Re‑run the tests and commands listed.
     - Update status to:
       - `verified` if all checks pass, or
       - `drift_detected` if any discrepancy appears (fix required before proceeding).

3. **Select next step**
   - Find the first step whose status is not `completed_pending_verification` or `verified`.
   - Work **only** on that step in this run.

4. **Plan the step**
   - Read the relevant sections of `CANONICAL_BLUEPRINT.md`.
   - Identify:
     - Affected modules/files.
     - Schemas/interfaces.
     - External services and credentials involved.
   - Update the step’s section in the ledger with the plan.

5. **Implement**
   - Make minimal, cohesive changes required to satisfy the step’s checklist.
   - Keep to existing patterns and architecture.

6. **Test (with live infra where applicable)**
   - Run the tests listed in the step (unit, integration, live).
   - Add at least the edge case tests specified for the step.
   - Capture test commands and results.

7. **Update ledger**
   - Record:
     - Summary of implementation.
     - Files touched (ideally with line ranges).
     - Tests run and their outcomes.
     - Edge cases covered.
     - Concerns and remaining items.
   - Mark step as `completed_pending_verification` at the end of the run.
   - Next run will re‑verify.

---

## 2. Sequential Implementation Steps

Each step below is **sequential**. Do not start S(X+1) before S(X) is `completed_pending_verification` (current run) and later `verified` (subsequent run).

For each step, **ALL** checklist items and tests must be completed before moving on.

---

### Step S00 – Baseline & Repo Sanity

**Goal:** Ensure repo and environment are clean and consistent before implementation.

**Checklist:**

- [ ] Confirm top‑level repo layout matches `CANONICAL_BLUEPRINT.md`.
- [ ] Ensure required tools are installed:
  - [ ] `python` and virtualenv tooling.
  - [ ] `pre-commit`.
  - [ ] `pytest`.
  - [ ] `doctl`.
- [ ] Confirm `.env` exists and includes (without logging values):
  - [ ] DB connection details.
  - [ ] DigitalOcean Spaces/S3 credentials.
  - [ ] DOKS / Kubernetes access details (directly or via `doctl`).
- [ ] Confirm `.env` is excluded from version control (e.g. `.gitignore`).
- [ ] Confirm any project‑level env vars are consistent with `.env`.

**Tests required before proceeding:**

- [ ] Run `pre-commit run --all-files` and fix any issues.
- [ ] Run `pytest` (or at least `pytest tests/unit` if large).
- [ ] Run `doctl account get` to confirm DigitalOcean auth works via `.env` or project‑level vars.

---

### Step S01 – Blueprint Inventory & Drift Analysis

**Goal:** Map blueprint expectations to actual code and detect gaps.

**Checklist:**

- [ ] Parse `CANONICAL_BLUEPRINT.md` for:
  - [ ] Services/modules.
  - [ ] DB schemas and tables.
  - [ ] External integrations.
  - [ ] API endpoints and contracts.
- [ ] For each blueprint item, locate:
  - [ ] Corresponding implementation.
  - [ ] Corresponding tests.
- [ ] Mark each as:
  - `implemented`, `partially_implemented`, or `missing` in the ledger.

**Tests required before proceeding:**

- [ ] If blueprint references any global smoke or sanity tests, run them and record outcomes in the ledger.

---

### Step S02 – Configuration & Secrets Wiring

**Goal:** Align all configuration and secrets with blueprint using `.env` and config modules.

**Checklist:**

- [ ] Identify central config loader module(s).
- [ ] Ensure all required config keys from the blueprint:
  - [ ] Exist in `.env` (values not logged).
  - [ ] Are loaded by the config module with proper types.
  - [ ] Are used by code (no hard‑coded secrets or endpoints).
- [ ] Remove or deprecate obsolete config keys, updating blueprint if necessary.

**Tests required before proceeding:**

- [ ] Start app locally (e.g. `python -m backend.main` or appropriate entrypoint) using `.env`; confirm no missing config errors.
- [ ] Test behavior when a non‑critical optional config is removed:
  - [ ] App fails fast with a clear error, **or**
  - [ ] App uses a documented default behavior.
- [ ] Run `pre-commit run --all-files` for config‑related changes.
- [ ] Run targeted tests around config loading if they exist.

---

### Step S03 – Database Connectivity & Migrations (Live)

**Goal:** Ensure DB connectivity, migrations, and schema alignment with blueprint.

**Checklist:**

- [ ] Locate DB connection/shim module.
- [ ] Confirm DB connection parameters are read from config, not hard‑coded.
- [ ] Run DB migrations against the real DB referenced by `.env` (or live‑like):
  - [ ] e.g. `alembic upgrade head` or tool specified in blueprint.
- [ ] Compare resulting schema to blueprint and resolve inconsistencies.

**Tests required before proceeding (all against real DB):**

- [ ] Run an app health check or dedicated script that:
  - [ ] Opens a DB session/connection successfully.
- [ ] CRUD tests via app code or test suite:
  - [ ] Insert a valid record and read it back.
  - [ ] Attempt to insert invalid data (violating constraints) and ensure it fails.
  - [ ] Query a non‑existent record and ensure behavior is safe (no crash).
- [ ] Run relevant DB integration tests (if present in tests tree).

---

### Step S04 – Object Storage (DigitalOcean Spaces / S3)

**Goal:** Wire and verify live object storage integration exactly as per blueprint.

**Checklist:**

- [ ] Identify storage shim module (Spaces/S3 client).
- [ ] Ensure bucket, endpoint, region, and credentials come from config.
- [ ] Ensure no hard‑coded secrets or bucket names remain.
- [ ] Implement any missing methods required by blueprint:
  - [ ] Upload.
  - [ ] Download.
  - [ ] Delete.
  - [ ] List.
  - [ ] Signed URLs (if specified).

**Tests required before proceeding (live Spaces):**

- [ ] Upload a small test object and confirm presence via app or direct S3 client.
- [ ] Download the same object and confirm content integrity.
- [ ] Delete the object and confirm subsequent reads fail as expected.
- [ ] Edge cases:
  - [ ] Upload zero‑length object.
  - [ ] Attempt to read a non‑existent object → verify clear error semantics.
- [ ] Record all commands and API calls in the ledger.

---

### Step S05 – Core Service Logic

**Goal:** Fully implement core business logic as dictated by the blueprint.

**Checklist:**

- [ ] For each core use case in the blueprint:
  - [ ] Identify service module(s) implementing it.
  - [ ] Ensure function signatures and behavior match blueprint contracts.
  - [ ] Replace any placeholders or partial implementations with full logic.
- [ ] Add/maintain type hints on new or changed functions.
- [ ] Ensure error handling:
  - [ ] No silent failures.
  - [ ] Domain errors are explicit (exceptions or error objects).

**Tests required before proceeding:**

- [ ] Unit tests for each core function:
  - [ ] Happy path.
  - [ ] Invalid inputs.
  - [ ] Boundary conditions.
- [ ] If core logic touches DB/storage:
  - [ ] Use live infra (no mocks) as required by project rules.
- [ ] Run `pytest` for the relevant test modules.
- [ ] Run `pre-commit run --all-files`.

---

### Step S06 – API Layer & Contracts

**Goal:** Ensure all API endpoints match blueprint contracts end‑to‑end.

**Checklist:**

- [ ] For each API endpoint in blueprint:
  - [ ] Verify route path and method.
  - [ ] Verify request/response schema.
  - [ ] Implement/adjust handler logic to call proper services.
- [ ] Ensure:
  - [ ] Input validation with clear errors.
  - [ ] Consistent error response structure.
  - [ ] Correct mapping for all relevant HTTP status codes.

**Tests required before proceeding (live or staging app):**

- [ ] For each endpoint:
  - [ ] Happy path with valid input.
  - [ ] Invalid payloads (missing/invalid fields).
  - [ ] Authorization/forbidden cases if auth applies.
- [ ] Use HTTP client (curl/httpie or test client) to exercise endpoints.
- [ ] Record sample requests/responses in the ledger.

---

### Step S07 – Authentication & Authorization

**Goal:** Implement secure authN/authZ exactly as defined in the blueprint.

**Checklist:**

- [ ] Implement or verify:
  - [ ] Token issuance and verification.
  - [ ] Credential validation (passwords or external IdP).
  - [ ] Role/permission checks for protected routes.
- [ ] Ensure:
  - [ ] Secrets/keys are loaded from config.
  - [ ] No plaintext password storage or logging.
  - [ ] Sensitive claims in tokens are appropriate and minimized.

**Tests required before proceeding:**

- [ ] Valid login → token (or session) issued.
- [ ] Invalid login → clean error.
- [ ] Expired token flow (if time‑based tokens are used).
- [ ] Access protected endpoint:
  - [ ] Without token.
  - [ ] With invalid token.
  - [ ] With token missing required role.
- [ ] Record all HTTP flows and results in the ledger.

---

### Step S08 – Background Jobs / Queues

*(Skip if blueprint does not define any background processing.)*

**Goal:** Ensure background jobs/queues are wired and working in live environment.

**Checklist:**

- [ ] Identify queue/broker and its config.
- [ ] Implement or verify:
  - [ ] Producers (enqueue jobs).
  - [ ] Consumers/workers (process jobs).
  - [ ] Retry/backoff strategies.
- [ ] Ensure no secrets are logged in jobs.

**Tests required before proceeding (live broker):**

- [ ] Enqueue a job and confirm:
  - [ ] It is processed successfully.
- [ ] Simulate job failure and confirm:
  - [ ] Retry behavior matches blueprint.
  - [ ] Poison messages handled according to design.
- [ ] Record test commands and observed behavior.

---

### Step S09 – Observability & Logging

**Goal:** Provide production‑grade logging and metrics.

**Checklist:**

- [ ] Ensure logs are:
  - [ ] Structured and consistent.
  - [ ] Free of secrets and disallowed PII.
- [ ] Implement metrics as per blueprint:
  - [ ] Request latency.
  - [ ] Error rates.
  - [ ] External dependency failures (DB, Spaces, etc.).
- [ ] Hook metrics into app lifecycle.

**Tests required before proceeding:**

- [ ] Generate test traffic and confirm logs:
  - [ ] Appear in the configured sink.
  - [ ] Contain expected fields.
- [ ] Confirm metrics are visible/queriable in the chosen system (if configured).

---

### Step S10 – CI / Pre‑commit / Automated Tests

**Goal:** Ensure all automated quality gates are correctly configured and passing.

**Checklist:**

- [ ] Verify CI pipeline steps match blueprint (linting, tests, type checks).
- [ ] Ensure `pre-commit` hooks are installed and configured properly.
- [ ] Ensure type checker (mypy or equivalent) is wired as required.

**Tests required before proceeding:**

- [ ] Locally:
  - [ ] `pre-commit run --all-files`
  - [ ] `pytest`
  - [ ] `mypy` (if used)
- [ ] Confirm CI workflow passes for the branch/PR.

---

### Step S11 – Deployment to DOKS (Staging & Production)

**Goal:** Deploy app to DOKS clusters using real infra.

**Checklist:**

- [ ] Use `doctl` to confirm cluster access.
- [ ] Build and push container image as per blueprint.
- [ ] Deploy to **staging**:
  - [ ] Apply Kubernetes manifests or Helm charts.
  - [ ] Confirm pods, services, and ingress resources are healthy.
- [ ] Once staging passes smoke tests, deploy to **production** using blueprint release strategy.

**Tests required before proceeding:**

- [ ] Staging:
  - [ ] Health endpoints.
  - [ ] Core user flows.
- [ ] Production:
  - [ ] Minimal, safe smoke tests (non‑destructive).
- [ ] Record deployed image tags, manifests/chart versions, and commands.

---

### Step S12 – Post‑Deployment Validation & Hardening

**Goal:** Confirm system is production‑ready and robust.

**Checklist:**

- [ ] Run end‑to‑end flows that mirror real user journeys.
- [ ] Check:
  - [ ] Error rates.
  - [ ] Latency SLOs.
  - [ ] Resource utilization.
- [ ] Review blueprint vs actual system:
  - [ ] All features implemented.
  - [ ] All safety and compliance requirements met.

**Tests required before proceeding:**

- [ ] End‑to‑end tests or manual flows in production (with test accounts/data).
- [ ] Confirm observability signals show healthy, stable behavior.
- [ ] Document remaining risks and TODOs in the Implementation_Ledger.md.

---
