"""Jules API provider for code review."""

from __future__ import annotations

import logging
import os
from asyncio import Lock
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
        starting_branch: str = "main",
        automation_mode: str = "AUTO_CREATE_PR",
    ):
        self.api_key = os.getenv("JULES_API_KEY")
        self.api_url = api_url
        self.source_id = f"sources/github/{repo_owner}/{repo_name}"
        self.starting_branch = starting_branch
        self.automation_mode = automation_mode

        self._session: aiohttp.ClientSession | None = None
        self._session_lock = Lock()

        if not self.api_key:
            raise ValueError("Missing JULES_API_KEY environment variable.")

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()
            return self._session

    async def review_file(
        self,
        file_path: Path,
        context: str,
        language: str,
    ) -> ReviewResult:
        """Create a Jules review session for the file."""
        file_path_str = str(file_path)

        prompt_text = (
            f"Analyze the file '{file_path_str}' for errors, mismatches, logic errors, and syntactical errors.\n"
            f"Fix any errors found, considering edge cases and unintended negative consequences.\n"
            f"Context files:\n{context}"
        )

        payload = {
            "title": f"Review: {file_path_str}",
            "prompt": prompt_text,
            "sourceContext": {
                "source": self.source_id,
                "githubRepoContext": {"startingBranch": self.starting_branch},
            },
            "automationMode": self.automation_mode,
        }

        from asyncio import TimeoutError as AsyncTimeoutError
        from json import JSONDecodeError

        # Lazy import to avoid startup cost/import cycles if not used
        from tenacity import (
            RetryError,
            before_sleep_log,
            retry,
            retry_if_exception,
            stop_after_attempt,
            wait_exponential,
        )

        def is_retriable(e: BaseException) -> bool:
            """Determine if an exception is retriable."""
            if isinstance(
                e,
                (
                    aiohttp.ClientConnectorError,
                    aiohttp.ServerDisconnectedError,
                    AsyncTimeoutError,
                ),
            ):
                return True
            if isinstance(e, aiohttp.ClientResponseError):
                return e.status == 429 or e.status >= 500
            return False

        @retry(
            wait=wait_exponential(multiplier=1, min=2, max=60),
            stop=stop_after_attempt(5),
            retry=retry_if_exception(is_retriable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
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
                resp.raise_for_status()  # Raise for non-2xx status codes
                try:
                    return await resp.json()
                except JSONDecodeError:
                    # Handle empty or non-JSON response for 2xx status codes
                    if resp.status in (200, 201, 202, 204):
                        return {}  # Treat as a valid, but empty, JSON object
                    raise  # Re-raise for unexpected JSON errors on other 2xx codes

        try:
            data = await _make_request()

            if data is not None:
                session_id = data.get("name", "unknown")
                logger.info(
                    "Created Jules session for %s: %s", file_path_str, session_id
                )
                return ReviewResult(
                    file=file_path_str,
                    summary=f"Jules session created: {session_id}",
                    model="jules",
                    language=language,
                )
            else:
                # This case should ideally not be reached due to raise_for_status,
                # but as a fallback:
                return ReviewResult(
                    file=file_path_str,
                    error="Request failed (unknown reason)",
                    model="jules",
                )

        except RetryError as e:
            logger.error(
                "API request for %s failed after multiple retries: %s", file_path_str, e
            )
            return ReviewResult(
                file=file_path_str,
                error=f"Request failed after retries: {e}",
                model="jules",
            )
        except aiohttp.ClientResponseError as e:
            logger.error(
                "API request for %s failed with status %d: %s",
                file_path_str,
                e.status,
                e.message,
            )
            return ReviewResult(
                file=file_path_str, error=f"HTTP {e.status}: {e.message}", model="jules"
            )
        except (TimeoutError, aiohttp.ClientError) as e:
            logger.error("API request for %s failed: %s", file_path_str, e)
            return ReviewResult(
                file=file_path_str, error=f"Request failed: {e}", model="jules"
            )
        except JSONDecodeError as e:
            logger.error("Failed to decode JSON response for %s: %s", file_path_str, e)
            return ReviewResult(
                file=file_path_str, error=f"Invalid JSON response: {e}", model="jules"
            )
        except Exception as e:
            logger.error(
                "An unexpected error occurred for %s: %s",
                file_path_str,
                e,
                exc_info=True,
            )
            return ReviewResult(
                file=file_path_str,
                error=f"An unexpected error occurred: {e}",
                model="jules",
            )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
