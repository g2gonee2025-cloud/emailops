"""OpenAI-compatible API provider (works with DO, OpenRouter, etc.)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import aiohttp

from .base import ReviewProvider, ReviewResult

logger = logging.getLogger(__name__)


class RateLimiter:
    """Handles rate limiting with exponential backoff."""

    def __init__(self, max_retries: int = 5, base_delay: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._lock = asyncio.Lock()
        self._global_wait_until = 0.0

    async def wait_if_needed(self) -> None:
        async with self._lock:
            now = time.time()
            if self._global_wait_until > now:
                wait_time = self._global_wait_until - now
                logger.warning("Rate limit active, waiting %.1fs...", wait_time)
                await asyncio.sleep(wait_time)

    async def set_global_wait(self, seconds: float) -> None:
        async with self._lock:
            self._global_wait_until = max(
                self._global_wait_until, time.time() + seconds
            )

    def get_delay(self, attempt: int) -> float:
        delay = self.base_delay * (2**attempt)
        jitter = delay * 0.1 * (0.5 - (time.time() % 1))
        return min(delay + jitter, 120)


REVIEW_PROMPT = """
<task>
You are a Senior Code Reviewer. Analyze the following {language} file for potential issues.
DO NOT suggest fixes. Only identify and describe problems.
</task>

<categories>
Report issues in these categories:
1. LOGIC_ERRORS: Incorrect conditionals, wrong comparisons, off-by-one, infinite loops, unreachable code
2. NULL_SAFETY: Missing null/undefined checks, unguarded optional access, NoneType/TypeError risks
3. EXCEPTION_HANDLING: Unhandled exceptions, swallowed errors, missing cleanup, wrong error types
4. SECURITY: Hardcoded secrets, injection vulnerabilities, unsafe operations, XSS/CSRF risks
5. PERFORMANCE: Inefficient loops, N+1 queries, memory leaks, redundant computations, unnecessary re-renders
6. STYLE: Major style violations, naming issues, dead code, magic numbers, accessibility issues
7. TYPE_ERRORS: Incorrect types, missing types on public APIs, type mismatches, any-abuse
</categories>

<context>
File: {file_path}
Language: {language}
Related Context:
{imports_context}
</context>

<code>
{file_content}
</code>

<output_format>
Respond ONLY with a JSON object:
{{
  "file": "<filename>",
  "issues": [
    {{"category": "<LOGIC_ERRORS|NULL_SAFETY|EXCEPTION_HANDLING|SECURITY|PERFORMANCE|STYLE|TYPE_ERRORS>", "line": <int or null>, "description": "<issue>"}}
  ],
  "summary": "<1-sentence overall assessment>"
}}
If no issues found, return {{"file": "<filename>", "issues": [], "summary": "No issues detected."}}
</output_format>
"""


class OpenAICompatProvider(ReviewProvider):
    """OpenAI-compatible API provider."""

    name = "openai"

    def __init__(self, model: str = "openai-gpt-5"):
        self.model = model
        self.api_key = os.getenv("LLM_API_KEY") or os.getenv("DO_API_KEY")
        self.base_url = (
            os.getenv("LLM_ENDPOINT") or "https://inference.do-ai.run/v1"
        ).rstrip("/")
        self.rate_limiter = RateLimiter()
        self._session: aiohttp.ClientSession | None = None

        if not self.api_key:
            raise ValueError("Missing API key. Set LLM_API_KEY or DO_API_KEY.")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )
        return self._session

    async def review_file(
        self,
        file_path: Path,
        content: str,
        context: str,
        language: str,
    ) -> ReviewResult:
        """Review a file using OpenAI-compatible API."""
        rel_path = str(file_path)

        if len(content) > 50000:
            return ReviewResult(
                file=rel_path, skipped=True, skip_reason="File too large"
            )
        if not content.strip():
            return ReviewResult(file=rel_path, skipped=True, skip_reason="Empty file")

        prompt = REVIEW_PROMPT.format(
            file_path=rel_path,
            language=language,
            imports_context=context[:3000],
            file_content=content[:15000],
        )

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
        }

        session = await self._get_session()

        for attempt in range(self.rate_limiter.max_retries):
            await self.rate_limiter.wait_if_needed()

            try:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status == 429:
                        retry_after_raw = resp.headers.get("Retry-After")
                        try:
                            retry_after = (
                                float(retry_after_raw) if retry_after_raw else 30.0
                            )
                        except ValueError:
                            retry_after = 30.0
                        await self.rate_limiter.set_global_wait(retry_after)
                        delay = self.rate_limiter.get_delay(attempt)
                        logger.warning(
                            "Rate limited (429). Retry %d/%d in %.1fs",
                            attempt + 1,
                            self.rate_limiter.max_retries,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    resp.raise_for_status()
                    data = await resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        return ReviewResult(
                            file=rel_path,
                            error="Empty choices in API response",
                            model=self.model,
                        )
                    raw_content = choices[0].get("message", {}).get("content", "{}")
                    # Strip thinking tags
                    raw_content = re.sub(
                        r"<think>.*?</think>", "", raw_content, flags=re.DOTALL
                    )
                    parsed = json.loads(raw_content)
                    if not isinstance(parsed, dict):
                        return ReviewResult(
                            file=rel_path,
                            error="Invalid JSON response shape",
                            model=self.model,
                        )
                    issues = parsed.get("issues") or []
                    if not isinstance(issues, list):
                        return ReviewResult(
                            file=rel_path,
                            error="Invalid issues format in response",
                            model=self.model,
                        )
                    summary = parsed.get("summary", "")
                    if not isinstance(summary, str):
                        summary = str(summary)

                    return ReviewResult(
                        file=rel_path,
                        issues=issues,
                        summary=summary,
                        model=self.model,
                        language=language,
                    )

            except TimeoutError:
                logger.warning("Timeout on attempt %d for %s", attempt + 1, rel_path)
                if attempt < self.rate_limiter.max_retries - 1:
                    await asyncio.sleep(self.rate_limiter.get_delay(attempt))
                    continue
                return ReviewResult(
                    file=rel_path, error="Timeout after max retries", model=self.model
                )

            except aiohttp.ClientError as e:
                if attempt < self.rate_limiter.max_retries - 1:
                    await asyncio.sleep(self.rate_limiter.get_delay(attempt))
                    continue
                return ReviewResult(
                    file=rel_path, error=f"ClientError: {e}", model=self.model
                )

            except json.JSONDecodeError as e:
                return ReviewResult(
                    file=rel_path, error=f"JSONDecodeError: {e}", model=self.model
                )

        return ReviewResult(
            file=rel_path, error="Max retries exceeded", model=self.model
        )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
