"""Jules API provider for code review."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import aiohttp

from .base import ReviewProvider, ReviewResult

logger = logging.getLogger(__name__)


class JulesProvider(ReviewProvider):
    """Jules API provider for automated PR-based code review."""

    name = "jules"

    def __init__(
        self,
        repo_owner: str = "g2gonee2025-cloud",
        repo_name: str = "emailops",
        api_url: str = "https://jules.googleapis.com/v1alpha/sessions",
    ):
        self.api_key = os.getenv("JULES_API_KEY")
        self.api_url = api_url
        self.source_id = f"sources/github/{repo_owner}/{repo_name}"
        self._session: aiohttp.ClientSession | None = None

        if not self.api_key:
            raise ValueError("Missing JULES_API_KEY environment variable.")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def review_file(
        self,
        file_path: Path,
        content: str,
        context: str,
        language: str,
    ) -> ReviewResult:
        """Create a Jules review session for the file."""
        rel_path = str(file_path)

        prompt_text = (
            f"Analyze the file '{rel_path}' for errors, mismatches, logic errors, and syntactical errors.\n"
            f"Fix any errors found, considering edge cases and unintended negative consequences.\n"
            f"Context files:\n{context}"
        )

        payload = {
            "title": f"Review: {rel_path}",
            "prompt": prompt_text,
            "sourceContext": {
                "source": self.source_id,
                "githubRepoContext": {"startingBranch": "main"},
            },
            "automationMode": "AUTO_CREATE_PR",
        }

        # Lazy import to avoid startup cost/import cycles if not used
        from tenacity import (
            before_sleep_log,
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        # Inner function to allow retry decorator on async method
        @retry(
            wait=wait_exponential(multiplier=1, min=2, max=60),
            stop=stop_after_attempt(5),
            retry=retry_if_exception_type(aiohttp.ClientResponseError),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _make_request():
            session = await self._get_session()
            async with session.post(
                self.api_url,
                json=payload,
                headers={
                    "X-Goog-Api-Key": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status == 429:
                    # Raise exception to trigger retry
                    resp.raise_for_status()

                if 200 <= resp.status < 300:
                    data = await resp.json()
                    return data
                else:
                    # Non-retriable error (unless we want to retry 5xx too, but sticking to rate limits mostly)
                    # Actually raise for 5xx too just in case?
                    # For now, let's just handle success or error return.
                    # If we don't raise, tenacity won't retry.
                    return None

        try:
            # We need to handle the potential retry-exhausted exception or success
            data = await _make_request()

            if data:
                session_id = data.get("name", "unknown")
                logger.info("Created Jules session for %s: %s", rel_path, session_id)
                return ReviewResult(
                    file=rel_path,
                    summary=f"Jules session created: {session_id}",
                    model="jules",
                    language=language,
                )
            else:
                # This path is hit if we returned None (non-200, non-429 response that wasn't raised)
                # We need to capture the error details, but _make_request as written above
                # complicates access to the response object for non-200s.
                # Let's refactor slightly to just do the request in a retry block safely.
                return ReviewResult(
                    file=rel_path, error="Request failed (unknown)", model="jules"
                )

        except aiohttp.ClientResponseError as e:
            # This catches 429s after retries are exhausted
            logger.error("API request failed for %s after retries: %s", rel_path, e)
            return ReviewResult(
                file=rel_path,
                error=f"Rate limit/Error after retries: {e}",
                model="jules",
            )

        except aiohttp.ClientError as e:
            logger.error("API request failed for %s: %s", rel_path, e)
            return ReviewResult(file=rel_path, error=str(e), model="jules")

        except TimeoutError:
            logger.error("API request timed out for %s", rel_path)
            return ReviewResult(file=rel_path, error="Timeout", model="jules")
        except Exception as e:
            logger.error("Unexpected error for %s: %s", rel_path, e)
            return ReviewResult(file=rel_path, error=str(e), model="jules")

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
