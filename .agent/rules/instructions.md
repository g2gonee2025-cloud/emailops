---
trigger: always_on
---

# GitHub Copilot Instructions — Outlook Cortex

> **Audience:** GitHub Copilot & other coding assistants working in this repo.  
> **Goal:** Keep all AI changes aligned with the canonical blueprint and implementation process.

---

## 1. Canonical Sources of Truth

1. Treat **`docs/CANONICAL_BLUEPRINT.md`** as the **architecture bible**:
   - It defines services, modules, schemas, and invariants.
   - If code disagrees with the blueprint, **the blueprint wins**.
2. Treat **`docs/AGENT_EXECUTION_CHECKLIST.md`** as your **per‑run playbook**:
   - Follow the steps in order (S00 → S12).
   - Do **not** jump ahead or partially start later steps.
3. Treat **`docs/IMPLEMENTATION_LEDGER.md`** as the **single journal of record**:
   - Every substantial change to core logic, infra, or contracts must update the ledger.
   - Never mark a step as complete without listing tests and invariants.

When in doubt, **read the blueprint first**, then the relevant checklist step, then update the ledger.

---

## 2. How Copilot Should Work in This Repo

On **every coding session**, Copilot should implicitly follow this loop:

1. **Locate the current work stage**
   - Open `docs/IMPLEMENTATION_LEDGER.md`.
   - Find the first step whose status is not `verified`.
   - Work **only** on that step.

2. **Re‑verify previous steps if needed**
   - For any step marked `completed_pending_verification`:
     - Re‑open the files listed.
     - Confirm the “Observable invariants” still hold.
     - Re‑run the exact tests/commands listed.
     - If all pass → update status to `verified`.
     - If anything fails → set `drift_detected`, fix the code, re‑run tests, then move back to `completed_pending_verification`.

3. **Plan before editing**
   - Read the relevant sections of `docs/CANONICAL_BLUEPRINT.md`.
   - Identify:
     - Which modules/files will change.
     - Which schemas or contracts are in play.
     - Which external services (DB, Spaces, queues, LLMs) are involved.
   - Add a brief “Planned work” note under the step in `docs/IMPLEMENTATION_LEDGER.md`.

4. **Make minimal, cohesive changes**
   - Prefer **small, focused diffs** that fully satisfy the current step.
   - Don’t start multiple steps in one PR or commit.
   - Keep all new code within the existing repo layout described in §2.2 of the blueprint.

5. **Run tests with live infra where applicable**
   - For anything that touches DB, Spaces, Redis, or LLMs, run the relevant **live tests**.
   - Add at least one happy‑path and one edge‑case test.
   - Record all commands and outcomes in the ledger.

6. **Update the ledger**
   - Summarize what changed, where, and how it was tested.
   - Only then mark the step as `completed_pending_verification`.

---

## 3. Hard Rules for Code Generation

When Copilot suggests code in this repo, it must obey:

### 3.1 No new top‑level structures

- Do **not** invent new root‑level directories or services.
- Add code only under the existing trees defined in the blueprint, for example:
  - `backend/src/cortex/...`
  - `backend/tests/...`
  - `workers/src/cortex_workers/...`
  - `cli/src/cortex_cli/...`
- If a completely new area is genuinely needed, update the blueprint first, then implement.

### 3.2 Use existing shims and helpers

- **Configuration:** always go through `cortex.config.loader.get_config()`.
- **Database:** use the DB/session helpers and repository layer under `cortex.db`, not raw connection strings.
- **LLMs & embeddings:** use `cortex.llm.client` / `cortex.llm.runtime`, never direct HTTP calls.
- **Queues/workers:** use the queue abstraction and worker job handlers defined in the blueprint.
- **Observability:** decorate new operations with `@trace_operation` and use `get_logger(__name__)` for logging.

### 3.3 Error handling & safety

- No `try/except` that silently swallows errors.
- Any `except` must:
  - re‑raise (same or wrapped) **or**
  - map to a clear failure path (HTTP error, domain error, or explicit return) and stop the flow.
- Never log secrets, credentials, or raw email/attachment bodies.
- When wiring new agent or LangGraph nodes, **do not** call Postgres, Redis, or external APIs directly from nodes—always go via tools/shims.

### 3.4 Strong typing & contracts

- All new public functions and models should be fully type‑hinted.
- Use Pydantic v2 models for request/response and tool schemas.
- When adding or changing schemas:
  - Update the relevant section in `docs/CANONICAL_BLUEPRINT.md`.
  - Add/adjust Alembic migrations (never mutate schema in place without migration).

### 3.5 No mocks on core paths

- For any path that is part of normal production behavior:
  - Do **not** insert mocks, placeholders, or “TODO implement later” stubs.
  - If infrastructure truly isn’t available, mark the step in the ledger as blocked and explain why—don’t pretend it’s complete.

---

## 4. Style, Linting & Tooling Expectations

- Python 3.11+, FastAPI, LangGraph, Pydantic v2.
- Follow the naming and layout conventions in the blueprint (e.g. `snake_case` functions, modules under `cortex.*`).
- Respect local tooling:
  - `pre-commit` (black, ruff, isort, mypy).
  - `pytest` for tests.
- Copilot should prefer **small, composable helpers** over giant monolithic functions.

Whenever Copilot proposes changes that would cause `pre-commit`, `pytest`, or type checks to fail, **fix them in the same change** before considering the step done.

---

## 5. How to Use This File in the Repo

1. Save this file as **`.github/copilot-instructions.md`** (GitHub’s preferred location), or at repo root as `copilot-instructions.md` if you prefer.
2. Keep it short and high‑signal:
   - If you substantially change the architecture or process, update:
     - `docs/CANONICAL_BLUEPRINT.md`
     - `docs/AGENT_EXECUTION_CHECKLIST.md`
     - `docs/IMPLEMENTATION_LEDGER.md`
     - and then adjust these Copilot instructions to match.
3. Treat this file as the **“front door”** for AI assistants:
   - It tells them where to look (blueprint + checklist + ledger).
   - It constrains how they’re allowed to modify the codebase.

If you’re Copilot and you are reading this: **respect the blueprint, follow the checklist, and keep the ledger honest.**
