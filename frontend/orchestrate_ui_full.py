#!/usr/bin/env python3
"""
orchestrate_ui_full.py

Wave-based orchestration for launching Jules API sessions to build a best-in-class UI.

Prereqs:
  pip install aiohttp

Auth:
  export JULES_API_KEY="..."
  (or JULES_API_KEY_ALT)

Usage:
  python orchestrate_ui_full.py --owner g2gonee2025-cloud --repo emailops --batch 1 --dry-run
  python orchestrate_ui_full.py --owner g2gonee2025-cloud --repo emailops --batch 1

Notes:
- This script *creates* Jules sessions (and PRs if automationMode=AUTO_CREATE_PR). It does not merge PRs.
- To minimize merge conflicts, waves are designed so each wave edits distinct primary files.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import aiohttp

API_ROOT = "https://jules.googleapis.com/v1alpha"
SESSIONS_URL = f"{API_ROOT}/sessions"
SOURCES_URL = f"{API_ROOT}/sources"

DEFAULT_CONCURRENCY = 3  # conservative; increase if you do not see 429s
POLL_INTERVAL_SECONDS = 10
DEFAULT_MAX_SESSION_MINUTES = 45

SESSION_TERMINAL_STATES = {"COMPLETED", "FAILED"}
SESSION_STOP_STATES = {
    "PAUSED",
    "AWAITING_USER_FEEDBACK",
}  # requires human intervention
SESSION_PLAN_STATE = "AWAITING_PLAN_APPROVAL"

ApprovalMode = str  # "none" | "auto" | "manual"

logger = logging.getLogger("jules_ui_orchestrator")


def _parse_env_file(env_path: Path) -> str | None:
    """Parse a .env file and return the API key if found."""
    if not env_path.exists():
        return None
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k in ("JULES_API_KEY_ALT", "JULES_API_KEY") and v:
                return v
    except Exception:
        return None
    return None


def load_api_key() -> str:
    """Load Jules API key from env or .env file."""
    key = os.environ.get("JULES_API_KEY_ALT") or os.environ.get("JULES_API_KEY")
    if key:
        return key.strip()

    # Lightweight .env support
    for env_path in (Path(".env"), Path("frontend/.env")):
        key = _parse_env_file(env_path)
        if key:
            return key

    return ""


def auth_headers(api_key: str) -> dict[str, str]:
    # Docs show both x-goog-api-key and X-Goog-Api-Key; use lowercase canonical.
    return {"x-goog-api-key": api_key}


async def list_sources(
    http: aiohttp.ClientSession, api_key: str
) -> list[dict[str, Any]]:
    """List all sources, handling pagination."""
    headers = auth_headers(api_key)
    sources: list[dict[str, Any]] = []
    page_token: str | None = None

    while True:
        params: dict[str, Any] = {"pageSize": 100}
        if page_token:
            params["pageToken"] = page_token

        async with http.get(SOURCES_URL, headers=headers, params=params) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"ListSources failed ({resp.status}): {text[:500]}")
            data = json.loads(text) if text.strip() else {}
            sources.extend(data.get("sources", []))
            page_token = data.get("nextPageToken")
            if not page_token:
                break

    return sources


def _source_repo_tuple(source_obj: dict[str, Any]) -> tuple[str, str] | None:
    gh = source_obj.get("githubRepo") or {}
    owner = gh.get("owner")
    repo = gh.get("repo")
    if isinstance(owner, str) and isinstance(repo, str):
        return owner, repo
    return None


def _source_default_branch(source_obj: dict[str, Any]) -> str | None:
    gh = source_obj.get("githubRepo") or {}
    default_branch = gh.get("defaultBranch") or {}
    name = default_branch.get("displayName")
    if isinstance(name, str) and name:
        return name
    return None


async def discover_source_name(
    http: aiohttp.ClientSession,
    api_key: str,
    owner: str,
    repo: str,
    explicit_source: str | None = None,
) -> tuple[str, str | None]:
    """
    Discover the correct Jules source resource name for a given GitHub repo.

    Returns: (source_name, default_branch_display_name)
    """
    if explicit_source:
        # Caller already knows the resource name; we won't validate to avoid extra failure modes.
        return explicit_source, None

    sources = await list_sources(http, api_key)
    matches = []
    for s in sources:
        tup = _source_repo_tuple(s)
        if tup and tup[0].lower() == owner.lower() and tup[1].lower() == repo.lower():
            matches.append(s)

    if not matches:
        sample = ", ".join(
            [
                f"{_source_repo_tuple(s)} -> {s.get('name')}"
                for s in sources[:5]
                if _source_repo_tuple(s)
            ]
        )
        raise RuntimeError(
            f"No Jules source found for {owner}/{repo}. "
            f"Connect the repo in the Jules UI, then retry. Sample sources: {sample}"
        )

    # If multiple matches exist, pick the first; they should be identical repo.
    chosen = matches[0]
    source_name = chosen.get("name")
    if not isinstance(source_name, str) or not source_name:
        raise RuntimeError(
            f"Matched source for {owner}/{repo} but missing 'name' field: {chosen}"
        )

    return source_name, _source_default_branch(chosen)


def build_prompt(task_def: dict[str, str]) -> str:
    return (
        f"Mission: {task_def['task']}\n"
        f"Primary target: {task_def['file']}\n\n"
        "Constraints:\n"
        f"- Apply changes primarily to {task_def['file']}.\n"
        "- Keep scope minimal to complete the mission.\n"
        "- Maintain TypeScript strictness and fix any lint errors introduced.\n"
        "- Add or update tests when relevant.\n\n"
        f"Instructions:\n{task_def['prompt']}\n"
    )


async def create_session(
    http: aiohttp.ClientSession,
    api_key: str,
    source_name: str,
    starting_branch: str,
    task_def: dict[str, str],
    approval_mode: ApprovalMode,
) -> dict[str, Any]:
    headers = {**auth_headers(api_key), "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "title": f"[UI] {task_def['task']} — {Path(task_def['file']).name}",
        "prompt": build_prompt(task_def),
        "sourceContext": {
            "source": source_name,
            "githubRepoContext": {"startingBranch": starting_branch},
        },
        "automationMode": "AUTO_CREATE_PR",
    }

    if approval_mode in ("auto", "manual"):
        payload["requirePlanApproval"] = True

    try:
        async with http.post(SESSIONS_URL, headers=headers, json=payload) as resp:
            text = await resp.text()
            if resp.status == 429:
                return {
                    "status": "rate_limited",
                    "task": task_def["task"],
                    "file": task_def["file"],
                    "error": text,
                }
            if resp.status not in (200, 201):
                return {
                    "status": "failed",
                    "task": task_def["task"],
                    "file": task_def["file"],
                    "error": f"{resp.status}: {text}",
                }

            data = json.loads(text) if text.strip() else {}
            # Create returns a Session object (name/id/url/state).
            return {
                "status": "created",
                "task": task_def["task"],
                "file": task_def["file"],
                "session_name": data.get("name"),
                "session_url": data.get("url"),
                "initial_state": data.get("state"),
                "raw": data,
            }
    except Exception as e:
        return {
            "status": "error",
            "task": task_def["task"],
            "file": task_def["file"],
            "error": str(e),
        }


async def approve_plan(
    http: aiohttp.ClientSession, api_key: str, session_name: str
) -> bool:
    """POST /v1alpha/{session=sessions/*}:approvePlan"""
    headers = {**auth_headers(api_key), "Content-Type": "application/json"}
    url = f"{API_ROOT}/{session_name}:approvePlan"
    async with http.post(url, headers=headers, json={}) as resp:
        # docs say empty response on success
        if resp.status in (200, 204):
            return True
        text = await resp.text()
        logger.warning(
            "approvePlan failed for %s (%s): %s", session_name, resp.status, text[:300]
        )
        return False


def extract_pr_urls(session_obj: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for out in session_obj.get("outputs", []) or []:
        pr = (out or {}).get("pullRequest")
        if isinstance(pr, dict):
            u = pr.get("url")
            if isinstance(u, str) and u:
                urls.append(u)
    # de-dup while preserving order
    seen = set()
    uniq = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
    return uniq


async def get_session(
    http: aiohttp.ClientSession, api_key: str, session_name: str
) -> dict[str, Any]:
    headers = auth_headers(api_key)
    url = f"{API_ROOT}/{session_name}"
    async with http.get(url, headers=headers) as resp:
        text = await resp.text()
        if resp.status == 429:
            raise RuntimeError("rate_limited")
        if resp.status != 200:
            raise RuntimeError(
                f"GET {session_name} failed ({resp.status}): {text[:500]}"
            )
        return json.loads(text) if text.strip() else {}


async def _handle_plan_approval(
    http: aiohttp.ClientSession,
    api_key: str,
    session_name: str,
    sess_url: str,
    approval_mode: ApprovalMode,
    printed_approval_hint: bool,
) -> bool:
    """Handle the logic for approving a session plan."""
    if approval_mode == "auto":
        ok = await approve_plan(http, api_key, session_name)
        if ok:
            logger.info("Approved plan for %s", session_name)
    elif not printed_approval_hint:
        logger.warning(
            "Session awaiting plan approval: %s (%s)", session_name, sess_url
        )
        return True
    return printed_approval_hint


async def poll_session(
    http: aiohttp.ClientSession,
    api_key: str,
    session_name: str,
    approval_mode: ApprovalMode,
    max_minutes: int,
) -> dict[str, Any]:
    """Poll session until terminal or stop state, auto/manual approval as configured."""
    deadline = time.monotonic() + (max_minutes * 60)
    printed_approval_hint = False

    while time.monotonic() < deadline:
        try:
            sess = await get_session(http, api_key, session_name)
        except RuntimeError as e:
            if str(e) == "rate_limited":
                await asyncio.sleep(POLL_INTERVAL_SECONDS * 2)
            else:
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
            continue

        state = sess.get("state")
        sess_url = sess.get("url")

        if state == SESSION_PLAN_STATE:
            printed_approval_hint = await _handle_plan_approval(
                http,
                api_key,
                session_name,
                sess_url,
                approval_mode,
                printed_approval_hint,
            )
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            continue

        if state in SESSION_TERMINAL_STATES or state in SESSION_STOP_STATES:
            pr_urls = extract_pr_urls(sess)
            return {
                "session_name": session_name,
                "state": state,
                "session_url": sess_url,
                "pr_urls": pr_urls,
                "raw": sess,
            }

        await asyncio.sleep(POLL_INTERVAL_SECONDS)

    return {
        "session_name": session_name,
        "state": "TIMEOUT",
        "session_url": None,
        "pr_urls": [],
        "raw": {},
    }


def validate_wave_no_conflicts(wave: list[dict[str, str]]) -> None:
    """Ensure each wave targets distinct primary files to reduce PR conflicts."""
    files = [t["file"] for t in wave]
    dupes = sorted({f for f in files if files.count(f) > 1})
    if dupes:
        raise ValueError(f"Wave has duplicate target files (conflict risk): {dupes}")


async def run_wave(
    http: aiohttp.ClientSession,
    api_key: str,
    source_name: str,
    starting_branch: str,
    approval_mode: ApprovalMode,
    wave_name: str,
    wave_tasks: list[dict[str, str]],
    concurrency: int,
    max_session_minutes: int,
) -> dict[str, Any]:
    validate_wave_no_conflicts(wave_tasks)
    sem = asyncio.Semaphore(concurrency)

    async def launch_one(t: dict[str, str]) -> dict[str, Any]:
        async with sem:
            return await create_session(
                http, api_key, source_name, starting_branch, t, approval_mode
            )

    # Launch sessions; stop early on 429
    created: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    rate_limited: dict[str, Any] | None = None

    tasks = [asyncio.create_task(launch_one(t)) for t in wave_tasks]
    try:
        for fut in asyncio.as_completed(tasks):
            res = await fut
            if res.get("status") == "rate_limited":
                rate_limited = res
                break
            if res.get("status") == "created":
                created.append(res)
            else:
                failures.append(res)
    finally:
        if rate_limited is not None:
            # Cancel unfinished tasks
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    if rate_limited is not None:
        return {
            "wave": wave_name,
            "status": "rate_limited",
            "rate_limited": rate_limited,
            "created": created,
            "failures": failures,
            "polled": [],
        }

    # Poll all created sessions concurrently (each poll loop sleeps)
    session_names = [
        c.get("session_name") for c in created if isinstance(c.get("session_name"), str)
    ]
    polls = [
        poll_session(http, api_key, s, approval_mode, max_session_minutes)
        for s in session_names
    ]
    polled = await asyncio.gather(*polls) if polls else []

    return {
        "wave": wave_name,
        "status": "ok",
        "created": created,
        "failures": failures,
        "polled": polled,
    }


def _count_tasks(batch: list[tuple[str, list[dict[str, str]]]]) -> int:
    return sum(len(wave_tasks) for _, wave_tasks in batch)


def _flatten_batch(
    batch: list[tuple[str, list[dict[str, str]]]],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for _, wave_tasks in batch:
        out.extend(wave_tasks)
    return out


def _ensure_unique_files_in_wave(batch: list[tuple[str, list[dict[str, str]]]]) -> None:
    for _, wave_tasks in batch:
        validate_wave_no_conflicts(wave_tasks)


def make_task(file: str, task: str, prompt: str) -> dict[str, str]:
    return {"file": file, "task": task, "prompt": prompt}


# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
API_TS = "frontend/src/lib/api.ts"
MAIN_TSX = "frontend/src/main.tsx"
MESSAGE_LIST_TSX = "frontend/src/components/thread/MessageList.tsx"
DOCTOR_PANEL_TSX = "frontend/src/components/admin/DoctorPanel.tsx"

# ------------------------------------------------------------------------------
# JOB DEFINITIONS: 200 total (25 + 45 + 80 + 50)
# ------------------------------------------------------------------------------

# Batch 1 (25): Bedrock
B1_W1 = [
    make_task(
        "frontend/package.json",
        "Install core deps & scripts",
        "Add devDependencies: vitest, jsdom, @testing-library/react, @testing-library/jest-dom, @testing-library/user-event, @playwright/test. "
        "Add dependencies: react-router-dom, @tanstack/react-query, @tanstack/react-query-devtools (optional), "
        "@radix-ui/react-dialog, @radix-ui/react-select, @radix-ui/react-tabs, @radix-ui/react-tooltip, @radix-ui/react-switch, "
        "@radix-ui/react-checkbox, @radix-ui/react-dropdown-menu, @radix-ui/react-scroll-area, @radix-ui/react-separator, @radix-ui/react-avatar, "
        "framer-motion, react-virtuoso, zod, react-hook-form, @hookform/resolvers, recharts, clsx, tailwind-merge, lucide-react. "
        "Add scripts: test (vitest run), test:watch (vitest), test:e2e (playwright test).",
    ),
    make_task(
        "frontend/vite.config.ts",
        "Configure Vitest",
        "Configure Vite+Vitest for React+TS: test.environment='jsdom', globals=true, setupFiles=['./vitest.setup.ts']. "
        "Ensure alias '@' -> './src' works in both dev and tests.",
    ),
    make_task(
        "frontend/vitest.setup.ts",
        "Vitest setup",
        "CREATE FILE. Add `import '@testing-library/jest-dom/vitest'` and any minimal setup required for RTL. Keep it lightweight.",
    ),
    make_task(
        "frontend/playwright.config.ts",
        "Playwright config",
        "CREATE FILE. Configure Playwright with baseURL http://localhost:5173 and a webServer that runs the frontend dev server. "
        "Add projects for chromium and webkit.",
    ),
    make_task(
        ".github/workflows/frontend-ci.yml",
        "Frontend CI",
        "CREATE FILE. Add a GitHub Actions workflow that runs (in frontend/): npm ci, npm run lint, npm run build, npm run test. "
        "Cache npm. Use Node 20.",
    ),
]

B1_W2 = [
    make_task(
        "frontend/src/lib/queryClient.ts",
        "Add QueryClient",
        "CREATE FILE. Export a configured TanStack QueryClient (staleTime 60000, refetchOnWindowFocus false, retry 1). "
        "Export a <QueryProvider> wrapper component.",
    ),
    make_task(
        API_TS,
        "Harden API client",
        "Refactor API client so EVERY request uses getHeaders() (auth coverage, including runDoctor). "
        "Add request<T> wrapper: handles response.ok, parses JSON safely, throws typed ApiError {status, message, details?}. "
        "Support AbortController signals and avoid logging sensitive data.",
    ),
    make_task(
        "frontend/src/contexts/AuthContext.tsx",
        "Auth consistency",
        "Update AuthContext to use request<T>() from api.ts. Handle 401 by clearing auth state and navigating to /login via React Router.",
    ),
    make_task(
        "frontend/src/contexts/toastContext.tsx",
        "Global API error toasts",
        "Enhance toast context to support success/error/info variants and typed API-error payloads. "
        "Listen for a custom window event (e.g., `api:error`) and show a global error toast.",
    ),
    make_task(
        MAIN_TSX,
        "Wire providers",
        "Ensure provider order: BrowserRouter -> QueryProvider -> ToastProvider -> AuthProvider -> App. "
        "Avoid multiple routers/providers and ensure strict mode is consistent.",
    ),
]

B1_W4 = [
    make_task(
        "frontend/src/tests/testUtils.tsx",
        "Test utilities",
        "CREATE FILE. Provide renderWithProviders helper: wraps MemoryRouter, QueryProvider, ToastProvider, AuthProvider. "
        "Expose helpers for setting auth state in tests.",
    ),
    make_task(
        "frontend/src/tests/smoke.test.tsx",
        "Smoke test",
        "CREATE FILE. Render <App/> and assert the navigation/sidebar exists.",
    ),
    make_task(
        "frontend/src/tests/routes.test.tsx",
        "Routing tests",
        "CREATE FILE. Test deep linking for /thread/:id and that /dashboard redirects to /login when unauthenticated.",
    ),
    make_task(
        "frontend/src/tests/authContext.test.tsx",
        "Auth tests",
        "CREATE FILE. Test login/logout transitions and that 401 handling clears token and navigates to /login.",
    ),
    make_task(
        "frontend/src/tests/api.test.ts",
        "API wrapper tests",
        "CREATE FILE. Unit tests for request<T> typed errors, status propagation, and auth header attachment.",
    ),
]

B1_W5 = [
    make_task(
        "frontend/src/components/LoginView.tsx",
        "Login works with router",
        "Ensure LoginView navigates via React Router on success, shows toast errors, and works with updated AuthContext + API wrapper.",
    ),
    make_task(
        "frontend/src/components/DashboardView.tsx",
        "Dashboard mounts",
        "Ensure DashboardView renders under new Layout/routes without crashing. Keep placeholder data; do not introduce fake timers here.",
    ),
    make_task(
        "frontend/src/index.css",
        "Global CSS baseline",
        "Ensure Tailwind directives exist. Add baseline CSS variables and typography defaults for dark glass theme.",
    ),
    make_task(
        "frontend/nginx.conf",
        "SPA + security headers",
        "Ensure SPA fallback (try_files ... /index.html). Add basic security headers (X-Content-Type-Options, X-Frame-Options, Referrer-Policy).",
    ),
    make_task(
        "frontend/README.md",
        "Frontend docs",
        "Update README with dev/test/e2e commands and how to configure auth env vars if needed.",
    ),
]

BATCH_1: list[tuple[str, list[dict[str, str]]]] = [
    ("B1.W4 Unit/Integration Tests", B1_W4),
    ("B1.W5 Cleanup", B1_W5),
]

# Batch 2 (45): Design System
RADIX_COMPONENTS = [
    "Dialog",
    "Select",
    "Tabs",
    "Tooltip",
    "Switch",
    "Checkbox",
    "DropdownMenu",
    "ScrollArea",
    "Separator",
    "Avatar",
]
BASIC_COMPONENTS = [
    "Button",
    "Input",
    "Label",
    "Textarea",
    "Badge",
    "Card",
    "Skeleton",
    "Table",
    "Alert",
    "Progress",
]

B2_W1 = [
    make_task(
        f"frontend/src/components/ui/{c}.tsx",
        f"UI {c} (Radix)",
        f"Create a <{c}> wrapper using Radix UI primitives. Ensure accessibility (keyboard, focus). "
        "Style with Tailwind glassmorphism tokens. Use forwardRef and typed props.",
    )
    for c in RADIX_COMPONENTS
]

B2_W2 = [
    make_task(
        f"frontend/src/components/ui/{c}.tsx",
        f"UI {c} (Tailwind)",
        f"Create a <{c}> component using Tailwind. Support variants/sizes as appropriate. Dark-mode glass aesthetic. Typed props; forwardRef when needed.",
    )
    for c in BASIC_COMPONENTS
]

TEST_A = [
    "Dialog",
    "Select",
    "Tabs",
    "Switch",
    "Checkbox",
    "Button",
    "Input",
    "Card",
    "Badge",
    "Alert",
]
TEST_B = [
    "Tooltip",
    "DropdownMenu",
    "ScrollArea",
    "Avatar",
    "Label",
    "Textarea",
    "Skeleton",
    "Table",
    "Progress",
    "Separator",
]

B2_W3 = [
    make_task(
        f"frontend/src/components/ui/{c}.test.tsx",
        f"Test {c}",
        f"CREATE FILE. Vitest + RTL unit tests for <{c}>: render, basic interactions, key aria attributes. Keep tests stable.",
    )
    for c in TEST_A
]

B2_W4 = [
    make_task(
        f"frontend/src/components/ui/{c}.test.tsx",
        f"Test {c}",
        f"CREATE FILE. Vitest + RTL unit tests for <{c}>. Include keyboard interaction tests where relevant.",
    )
    for c in TEST_B
]

B2_W5 = [
    make_task(
        "frontend/tailwind.config.js",
        "Tailwind theme polish",
        "Refine Tailwind theme tokens for glassmorphism (colors via CSS variables, border/shadow). Extend existing config; do not break build.",
    ),
    make_task(
        "frontend/src/index.css",
        "CSS variables",
        "Define standard CSS variables: --background, --foreground, --primary, --destructive, --muted, etc. Ensure components reference these.",
    ),
    make_task(
        "frontend/src/components/ui/GlassCard.tsx",
        "GlassCard refactor",
        "Refactor GlassCard to use the new Card primitives and cn() utility. Ensure consistent spacing and borders.",
    ),
    make_task(
        "frontend/src/lib/utils.ts",
        "Add formatting helpers",
        "Add small formatting utilities (formatDate, formatNumber, truncate) while keeping cn(). Ensure exports are tree-shakeable.",
    ),
    make_task(
        "frontend/src/components/ui/Loader.tsx",
        "Add Loader",
        "CREATE FILE. Standard Loader/spinner component; accessible (aria-label) and theme-consistent.",
    ),
]

BATCH_2: list[tuple[str, list[dict[str, str]]]] = [
    ("B2.W1 Radix wrappers", B2_W1),
    ("B2.W2 Basic components", B2_W2),
    ("B2.W3 Unit tests A", B2_W3),
    ("B2.W4 Unit tests B", B2_W4),
    ("B2.W5 Theme polish", B2_W5),
]

# Batch 3 (80): Feature Refactor
VIEWS: list[dict[str, str]] = [
    {
        "view": "DashboardView",
        "view_file": "frontend/src/components/DashboardView.tsx",
        "hook": "useDashboardMetrics",
        "hook_file": "frontend/src/hooks/useDashboardMetrics.ts",
        "schema_file": "frontend/src/schemas/dashboard.ts",
        "test_file": "frontend/src/tests/DashboardView.test.tsx",
        "e2e_file": "frontend/e2e/dashboard.spec.ts",
    },
    {
        "view": "SearchView",
        "view_file": "frontend/src/components/SearchView.tsx",
        "hook": "useSearch",
        "hook_file": "frontend/src/hooks/useSearch.ts",
        "schema_file": "frontend/src/schemas/search.ts",
        "test_file": "frontend/src/tests/SearchView.test.tsx",
        "e2e_file": "frontend/e2e/search.spec.ts",
    },
    {
        "view": "DraftView",
        "view_file": "frontend/src/components/DraftView.tsx",
        "hook": "useDraft",
        "hook_file": "frontend/src/hooks/useDraft.ts",
        "schema_file": "frontend/src/schemas/draft.ts",
        "test_file": "frontend/src/tests/DraftView.test.tsx",
        "e2e_file": "frontend/e2e/draft.spec.ts",
    },
    {
        "view": "SummarizeView",
        "view_file": "frontend/src/components/SummarizeView.tsx",
        "hook": "useSummarize",
        "hook_file": "frontend/src/hooks/useSummarize.ts",
        "schema_file": "frontend/src/schemas/summarize.ts",
        "test_file": "frontend/src/tests/SummarizeView.test.tsx",
        "e2e_file": "frontend/e2e/summarize.spec.ts",
    },
    {
        "view": "AdminDashboard",
        "view_file": "frontend/src/components/AdminDashboard.tsx",
        "hook": "useAdmin",
        "hook_file": "frontend/src/hooks/useAdmin.ts",
        "schema_file": "frontend/src/schemas/admin.ts",
        "test_file": "frontend/src/tests/AdminDashboard.test.tsx",
        "e2e_file": "frontend/e2e/admin.spec.ts",
    },
    {
        "view": "LoginView",
        "view_file": "frontend/src/components/LoginView.tsx",
        "hook": "useLogin",
        "hook_file": "frontend/src/hooks/useLogin.ts",
        "schema_file": "frontend/src/schemas/login.ts",
        "test_file": "frontend/src/tests/LoginView.test.tsx",
        "e2e_file": "frontend/e2e/login.spec.ts",
    },
]

B3_W1 = [
    make_task(
        v["hook_file"],
        f"Hook {v['hook']}",
        f"CREATE FILE. Implement {v['hook']} using TanStack Query. Encapsulate API calls from api.ts. "
        "Return { data, isLoading, error } plus mutation helpers where needed.",
    )
    for v in VIEWS
]

B3_W2 = [
    make_task(
        v["schema_file"],
        f"Schema {v['view']}",
        "CREATE FILE. Define Zod schemas for data/forms used in this view. Export inferred TS types and helper validators.",
    )
    for v in VIEWS
]

B3_W3 = [
    make_task(
        v["view_file"],
        f"Refactor {v['view']}",
        f"Refactor {v['view']} to use {v['hook']} + Zod schema + Design System components. "
        "Remove manual fetch calls; implement consistent loading (Skeleton) and error (Alert + toast) states.",
    )
    for v in VIEWS
]

B3_W4 = [
    make_task(
        v["test_file"],
        f"Test {v['view']}",
        f"CREATE FILE. Integration tests for {v['view']} using RTL/Vitest. Mock {v['hook']} to test loading/error/success rendering.",
    )
    for v in VIEWS
]

B3_W5 = [
    make_task(
        v["e2e_file"],
        f"E2E {v['view']}",
        f"CREATE FILE. Playwright test for primary {v['view']} flow. Keep deterministic; prefer data-testid selectors.",
    )
    for v in VIEWS
]

B3_W6 = [
    make_task(
        "frontend/src/lib/threadUtils.ts",
        "Thread utils",
        "CREATE FILE. Utilities for sorting/flattening thread chunks/messages. Pure functions.",
    ),
    make_task(
        "frontend/src/lib/ingestUtils.ts",
        "Ingestion utils",
        "CREATE FILE. Utilities to validate/normalize ingestion JSON manifests and payloads.",
    ),
    make_task(
        "frontend/src/lib/searchUtils.ts",
        "Search utils",
        "CREATE FILE. Utilities to serialize/parse search query params (URL state).",
    ),
    make_task(
        "frontend/src/lib/chartUtils.ts",
        "Chart utils",
        "CREATE FILE. Helpers to map metrics to Recharts datasets, plus formatters.",
    ),
    make_task(
        MESSAGE_LIST_TSX,
        "MessageList component",
        "CREATE FILE. Virtualized message list using react-virtuoso.",
    ),
    make_task(
        "frontend/src/components/search/FilterBar.tsx",
        "FilterBar component",
        "CREATE FILE. Filter bar for SearchView using Select/Input; sync with URL params.",
    ),
    make_task(
        "frontend/src/components/dashboard/KPIGrid.tsx",
        "KPIGrid component",
        "CREATE FILE. KPI grid for dashboard metrics.",
    ),
    make_task(
        "frontend/src/components/dashboard/IngestionChart.tsx",
        "IngestionChart component",
        "CREATE FILE. Recharts chart for ingestion throughput/latency.",
    ),
    make_task(
        "frontend/src/components/ask/ChatWindow.tsx",
        "ChatWindow component",
        "CREATE FILE. Chat transcript component using ScrollArea; supports streaming placeholder.",
    ),
    make_task(
        "frontend/src/components/ingestion/JobTable.tsx",
        "JobTable component",
        "CREATE FILE. Table for ingestion jobs with status badges and retry action.",
    ),
    make_task(
        "frontend/src/components/draft/TemplateSelector.tsx",
        "TemplateSelector component",
        "CREATE FILE. Draft template selector using Select; supports saved templates.",
    ),
    make_task(
        DOCTOR_PANEL_TSX,
        "DoctorPanel component",
        "CREATE FILE. Doctor results panel with copy/download actions.",
    ),
    make_task(
        "frontend/src/components/admin/ConfigPanel.tsx",
        "ConfigPanel component",
        "CREATE FILE. Config display/editor with redaction and validation.",
    ),
    make_task(
        "frontend/src/hooks/useDebounce.ts",
        "useDebounce hook",
        "CREATE FILE. Debounce hook with tests.",
    ),
    make_task(
        "frontend/src/hooks/useLocalStorage.ts",
        "useLocalStorage hook",
        "CREATE FILE. Persisted state hook (SSR-safe) with tests.",
    ),
]

B3_W7 = [
    make_task(
        v["view_file"],
        f"Verify {v['view']}",
        "Verify the refactor: fix lint/type issues, ensure loading/error states are consistent, and ensure no UX regressions.",
    )
    for v in VIEWS
] + [
    make_task(
        "frontend/src/App.tsx",
        "Verify routing integration",
        "Verify all routes render under Layout and deep links work (/thread/:id). Add Suspense fallbacks if missing.",
    ),
    make_task(
        "frontend/src/components/Layout.tsx",
        "Verify Layout",
        "Ensure Layout has skip-link target, responsive behavior, and no overflow issues.",
    ),
    make_task(
        "frontend/src/components/Sidebar.tsx",
        "Verify Sidebar",
        "Ensure NavLink active styles, keyboard nav, and aria-current handling.",
    ),
    make_task(
        MAIN_TSX,
        "Verify provider order",
        "Ensure BrowserRouter + QueryProvider + Auth + Toast are correctly nested.",
    ),
    make_task(
        "frontend/src/lib/queryClient.ts",
        "Verify Query defaults",
        "Tune retry/refetch defaults and ensure query keys are stable.",
    ),
    make_task(
        API_TS,
        "Verify API client",
        "Ensure ApiError typing, header attachment, and no sensitive console logging.",
    ),
    make_task(
        MESSAGE_LIST_TSX,
        "Verify virtualization",
        "Ensure virtualization scroll behavior and performance.",
    ),
    make_task(
        "frontend/src/components/search/FilterBar.tsx",
        "Verify URL state",
        "Ensure URL search params round-trip correctly and refresh persists state.",
    ),
    make_task(
        DOCTOR_PANEL_TSX,
        "Verify DoctorPanel",
        "Ensure results are readable, copy works, and error display is friendly.",
    ),
    make_task(
        "frontend/src/hooks/useDebounce.ts",
        "Verify useDebounce",
        "Ensure hook cleans up timers and handles changing values.",
    ),
    make_task(
        "frontend/src/hooks/useMediaQuery.ts",
        "Add useMediaQuery",
        "CREATE FILE. Media query hook for responsive UI; include tests.",
    ),
]

B3_REMAINING = [
    make_task(
        MESSAGE_LIST_TSX,
        "MessageList component",
        "CREATE FILE. Virtualized message list using react-virtuoso.",
    ),
    make_task(
        "frontend/src/components/dashboard/IngestionChart.tsx",
        "IngestionChart component",
        "CREATE FILE. Recharts chart for ingestion throughput/latency.",
    ),
    make_task(
        "frontend/src/components/draft/TemplateSelector.tsx",
        "TemplateSelector component",
        "CREATE FILE. Draft template selector using Select; supports saved templates.",
    ),
    make_task(
        DOCTOR_PANEL_TSX,
        "DoctorPanel component",
        "CREATE FILE. Doctor results panel with copy/download actions.",
    ),
]

BATCH_3: list[tuple[str, list[dict[str, str]]]] = [
    ("B3.W6 Remaining Components", B3_REMAINING),
    ("B3.W7 Verification", B3_W7),
]

# Batch 4 (50): Deep Polish
B4_W1 = [
    make_task(
        v["view_file"],
        f"A11y audit {v['view']}",
        "Audit this view for WCAG 2.1 AA: keyboard navigation, focus order, aria labels, headings/landmarks, color contrast, and empty states.",
    )
    for v in VIEWS
] + [
    make_task(
        "frontend/src/components/ui/Dialog.tsx",
        "A11y Dialog",
        "Verify focus trap, Escape, aria-describedby/title, and initial focus behavior.",
    ),
    make_task(
        "frontend/src/components/ui/Select.tsx",
        "A11y Select",
        "Verify keyboard navigation, aria attributes, and screen reader labeling.",
    ),
    make_task(
        "frontend/src/components/ui/Tabs.tsx",
        "A11y Tabs",
        "Verify arrow key navigation, aria-selected, and focus management.",
    ),
    make_task(
        "frontend/src/components/ui/Tooltip.tsx",
        "A11y Tooltip",
        "Verify tooltip trigger semantics and aria-describedby usage.",
    ),
    make_task(
        "frontend/src/components/Sidebar.tsx",
        "A11y Sidebar",
        "Ensure nav has proper landmarks/labels and active item is announced.",
    ),
    make_task(
        "frontend/src/App.tsx",
        "A11y skip-link",
        "Add skip-to-content link and main landmark; ensure focus moves correctly.",
    ),
]

B4_W2 = [
    make_task(
        f"frontend/e2e/scenario_{i:02d}.spec.ts",
        f"E2E scenario {i:02d}",
        "CREATE FILE. Playwright multi-step scenario test. Use stable selectors and avoid flakiness. Cover edge cases and error handling.",
    )
    for i in range(1, 16)
]

B4_W3 = [
    make_task(
        v["view_file"],
        f"Motion polish {v['view']}",
        "Add subtle framer-motion transitions for main content and list items. Respect prefers-reduced-motion.",
    )
    for v in VIEWS
] + [
    make_task(
        "frontend/src/components/ui/Toast.tsx",
        "Motion Toast",
        "Add smooth enter/exit animations for Toasts. Respect prefers-reduced-motion.",
    ),
]

B4_W4 = [
    make_task(
        "frontend/vite.config.ts",
        "Code-splitting",
        "Optimize rollupOptions/manualChunks for vendor chunking (react, radix, tanstack, charts) without breaking build.",
    ),
    make_task(
        "frontend/nginx.conf",
        "Security headers",
        "Add CSP header (report-only first if safer), Referrer-Policy, Permissions-Policy. Ensure SPA fallback remains.",
    ),
    make_task(
        "frontend/index.html",
        "SEO/meta polish",
        "Set proper <title>, lang attribute, meta description, and theme-color.",
    ),
    make_task(
        "frontend/src/lib/api.ts",
        "Abort + sanitize",
        "Add AbortController support to request<T>() and ensure no sensitive logs in production builds.",
    ),
    make_task(
        "frontend/src/components/ThreadView.tsx",
        "Thread perf",
        "Reduce re-renders (memo/useMemo) and ensure virtualized MessageList is used.",
    ),
    make_task(
        "frontend/src/components/SearchView.tsx",
        "Search perf",
        "Optimize result list rendering; use memoization and stable keys. Avoid redundant queries.",
    ),
    make_task(
        "frontend/src/components/IngestionView.tsx",
        "Ingestion perf",
        "Optimize job table rendering and polling (cancellable, efficient).",
    ),
    make_task(
        "frontend/src/main.tsx",
        "Web vitals",
        "Add web-vitals instrumentation gated behind a debug flag; log to console or placeholder endpoint.",
    ),
    make_task(
        "frontend/public/robots.txt",
        "Robots",
        "CREATE FILE. Provide sensible robots.txt (consider disallowing /admin).",
    ),
    make_task(
        "frontend/package.json",
        "Bundle analysis script",
        "Add optional scripts: build:analyze (rollup-plugin-visualizer or similar). Do not run in CI by default.",
    ),
]

BATCH_4: list[tuple[str, list[dict[str, str]]]] = [
    ("B4.W1 Accessibility", B4_W1),
    ("B4.W2 Advanced E2E", B4_W2),
    ("B4.W3 Motion", B4_W3),
    ("B4.W4 Performance & security", B4_W4),
]

BATCHES: dict[str, list[tuple[str, list[dict[str, str]]]]] = {
    "1": BATCH_1,
    "2": BATCH_2,
    "3": BATCH_3,
    "4": BATCH_4,
}

EXPECTED_COUNTS = {"1": 25, "2": 45, "3": 80, "4": 50}


def print_dry_run(batch_id: str, batch: list[tuple[str, list[dict[str, str]]]]) -> None:
    logger.info("DRY RUN for Batch %s", batch_id)
    total = _count_tasks(batch)
    logger.info("Total jobs: %s", total)
    for wave_name, wave_tasks in batch:
        logger.info("- %s: %s jobs", wave_name, len(wave_tasks))
        for t in wave_tasks:
            logger.info("    • %s -> %s", t["file"], t["task"])


def _process_wave_result(wave_result: dict[str, Any], wave_name: str) -> bool:
    """Process the result of a wave, logging PRs and checking for failures."""
    if wave_result.get("status") == "rate_limited":
        logger.error(
            "Rate limited during %s. Reduce concurrency and retry.", wave_name
        )
        return True

    pr_urls: list[str] = []
    for sess in wave_result.get("polled", []) or []:
        for u in sess.get("pr_urls", []) or []:
            pr_urls.append(u)
    if pr_urls:
        logger.info("PRs from %s:", wave_name)
        for u in pr_urls:
            logger.info("  %s", u)
    else:
        logger.info(
            "No PR URLs detected for %s (may still be in outputs, or no change).",
            wave_name,
        )

    creation_failures = [
        f
        for f in (wave_result.get("failures", []) or [])
        if f.get("status") in ("failed", "error", "rate_limited")
    ]
    non_success_states = {
        "FAILED",
        "PAUSED",
        "AWAITING_USER_FEEDBACK",
        "TIMEOUT",
    }
    bad_sessions = [
        s
        for s in (wave_result.get("polled", []) or [])
        if s.get("state") in non_success_states
    ]

    if creation_failures or bad_sessions:
        logger.error(
            "Wave %s has failures/paused/timeout sessions. Stopping batch. Re-run with --continue-on-failure to proceed.",
            wave_name,
        )
        return True

    return False


async def run_batch(
    batch_id: str,
    batch: list[tuple[str, list[dict[str, str]]]],
    api_key: str,
    source_name: str,
    starting_branch: str,
    approval_mode: ApprovalMode,
    concurrency: int,
    max_session_minutes: int,
    pause_between_waves: bool,
    out_dir: Path,
    continue_on_failure: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    batch_report_path = out_dir / f"ui_batch_{batch_id}_{run_id}.json"

    results: dict[str, Any] = {
        "batch": batch_id,
        "source": source_name,
        "branch": starting_branch,
        "approval_mode": approval_mode,
        "concurrency": concurrency,
        "max_session_minutes": max_session_minutes,
        "waves": [],
    }

    timeout = aiohttp.ClientTimeout(total=60 * 10)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        for idx, (wave_name, wave_tasks) in enumerate(batch, start=1):
            logger.info(
                "=== Wave %s/%s: %s (%s tasks) ===",
                idx,
                len(batch),
                wave_name,
                len(wave_tasks),
            )

            wave_result = await run_wave(
                http=http,
                api_key=api_key,
                source_name=source_name,
                starting_branch=starting_branch,
                approval_mode=approval_mode,
                wave_name=wave_name,
                wave_tasks=wave_tasks,
                concurrency=concurrency,
                max_session_minutes=max_session_minutes,
            )

            results["waves"].append(wave_result)
            batch_report_path.write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )

            should_stop = _process_wave_result(wave_result, wave_name)
            if should_stop and not continue_on_failure:
                break

            if pause_between_waves and idx < len(batch):
                if sys.stdin.isatty():
                    await asyncio.to_thread(
                        input,
                        "Merge the PRs from this wave (if any), then press Enter to continue to the next wave...",
                    )
                else:
                    logger.warning(
                        "pause-between-waves enabled but no TTY; continuing without pause."
                    )

    logger.info("Batch report written to: %s", batch_report_path)
    return batch_report_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True, help="GitHub owner (org/user)")
    parser.add_argument("--repo", required=True, help="GitHub repository name")
    parser.add_argument(
        "--batch",
        required=True,
        choices=["1", "2", "3", "4"],
        help="Which batch to run",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional explicit Jules source name (e.g., sources/github-myorg-myrepo)",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Optional starting branch override (defaults to source defaultBranch or 'main')",
    )
    parser.add_argument(
        "--approval",
        choices=["none", "auto", "manual"],
        default="auto",
        help="Plan approval mode",
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--max-session-minutes", type=int, default=DEFAULT_MAX_SESSION_MINUTES
    )
    parser.add_argument(
        "--pause-between-waves",
        dest="pause_between_waves",
        action="store_true",
        default=sys.stdin.isatty(),
        help="Pause between waves (default: on for interactive terminals)",
    )
    parser.add_argument(
        "--no-pause-between-waves",
        dest="pause_between_waves",
        action="store_false",
        help="Do not pause between waves",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue to next wave even if failures occur",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out-dir", default="jules_reports", help="Directory to write JSON reports"
    )
    args = parser.parse_args()

    api_key = load_api_key()
    if not api_key:
        sys.exit("Missing Jules API key. Set JULES_API_KEY (or JULES_API_KEY_ALT).")

    batch = BATCHES[args.batch]

    # Validate each wave has unique target files
    _ensure_unique_files_in_wave(batch)

    if args.dry_run:
        print_dry_run(args.batch, batch)
        return

    out_dir = Path(args.out_dir)

    async def _run() -> None:
        timeout = aiohttp.ClientTimeout(total=60 * 5)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            source_name, default_branch = await discover_source_name(
                http=http,
                api_key=api_key,
                owner=args.owner,
                repo=args.repo,
                explicit_source=args.source,
            )

        starting_branch = args.branch or default_branch or "main"
        logger.info("Using source=%s startingBranch=%s", source_name, starting_branch)

        await run_batch(
            batch_id=args.batch,
            batch=batch,
            api_key=api_key,
            source_name=source_name,
            starting_branch=starting_branch,
            approval_mode=args.approval,
            concurrency=args.concurrency,
            max_session_minutes=args.max_session_minutes,
            pause_between_waves=args.pause_between_waves,
            out_dir=out_dir,
            continue_on_failure=args.continue_on_failure,
        )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
