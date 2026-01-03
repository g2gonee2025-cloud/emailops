"""Code reviewer orchestrator."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
)

from scripts.review_cli.providers.base import ReviewResult

if TYPE_CHECKING:
    from scripts.review_cli.config import Config
    from scripts.review_cli.providers.base import ReviewProvider
    from scripts.review_cli.scanners.file_scanner import FileScanner

logger = logging.getLogger(__name__)


class CodeReviewer:
    """Orchestrates code review across multiple files."""

    def __init__(
        self,
        provider: ReviewProvider,
        scanner: FileScanner,
        config: Config,
    ):
        self.provider = provider
        self.scanner = scanner
        self.config = config
        self.results: list[ReviewResult] = []
        self._semaphore: asyncio.Semaphore | None = None

    async def _review_single_file(
        self, file_path: Path, progress: Progress, task_id: TaskID
    ) -> ReviewResult:
        """Review a single file with semaphore limiting."""
        if self._semaphore is None:
            raise RuntimeError("Semaphore not initialized. Call run() first.")

        async with self._semaphore:
            try:
                rel_path = file_path.relative_to(self.config.project_root)
            except ValueError:
                rel_path = file_path

            try:
                # PERFORMANCE: Check file size before reading
                if (
                    self.config.review.max_file_size > 0
                    and file_path.stat().st_size > self.config.review.max_file_size
                ):
                    progress.advance(task_id)
                    return ReviewResult(
                        file=str(rel_path),
                        skipped=True,
                        skip_reason=f"Exceeds max size of {self.config.review.max_file_size} bytes",
                    )
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                progress.advance(task_id)
                return ReviewResult(
                    file=str(rel_path), skipped=True, skip_reason="Binary file"
                )
            except OSError as e:
                progress.advance(task_id)
                return ReviewResult(file=str(rel_path), error=str(e))

            try:
                language = self.scanner.get_language(file_path)
                context = self.scanner.get_context(file_path)
            except Exception as e:
                progress.advance(task_id)
                logger.error("Failed to get language/context for %s: %s", rel_path, e)
                return ReviewResult(file=str(rel_path), error=str(e))

            model_name = getattr(
                self.provider, "model", getattr(self.provider, "name", "")
            )

            try:
                result = await self.provider.review_file(
                    file_path=rel_path,
                    content=content,
                    context=context,
                    language=language,
                )
            except Exception as exc:
                progress.advance(task_id)
                logger.error("Review failed for %s: %s", rel_path, exc)
                return ReviewResult(
                    file=str(rel_path),
                    error=str(exc),
                    model=model_name,
                    language=language,
                )

            progress.advance(task_id)

            # Log result
            if result.skipped:
                logger.debug("⏭ Skipped %s: %s", rel_path, result.skip_reason)
            elif result.error:
                logger.error(
                    "❌ %s: %s",
                    rel_path,
                    str(result.error)[:80] if result.error else "Unknown error",
                )
            elif result.has_issues:
                logger.warning("⚠️ %s: %d issues", rel_path, len(result.issues))
            else:
                logger.info("✅ %s: No issues", rel_path)

            return result

    async def run(
        self, on_result: Callable[[ReviewResult], Awaitable[None]] | None = None
    ) -> list[ReviewResult]:
        """Run the code review on all discovered files."""
        files = self.scanner.scan()
        logger.info("Found %d files to review", len(files))

        if self.config.review.dry_run:
            logger.info("Dry run mode - files that would be reviewed:")
            for f in files:
                try:
                    rel = f.relative_to(self.config.project_root)
                except ValueError:
                    rel = f
                logger.info("  - %s", rel)
            return []

        max_workers = self.config.review.max_workers
        if max_workers <= 0:
            logger.warning(
                "max_workers must be positive, defaulting to 1. Got: %d", max_workers
            )
            max_workers = 1
        self._semaphore = asyncio.Semaphore(max_workers)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task_id = progress.add_task("Reviewing files...", total=len(files))

            # PERFORMANCE: Use a worker pattern with a queue to avoid creating all tasks at once
            queue = asyncio.Queue()
            for f in files:
                await queue.put(f)

            async def worker(worker_id: int):
                while not queue.empty():
                    file_path = await queue.get()
                    try:
                        result = await self._review_single_file(
                            file_path, progress, task_id
                        )
                        self.results.append(result)
                        if on_result:
                            await on_result(result)
                    except Exception as exc:
                        logger.error(
                            "Worker %d failed processing %s: %s",
                            worker_id,
                            file_path,
                            exc,
                        )
                    finally:
                        queue.task_done()

            self.results = []
            worker_tasks = [asyncio.create_task(worker(i)) for i in range(max_workers)]
            await queue.join()

            for wt in worker_tasks:
                wt.cancel()
            await asyncio.gather(*worker_tasks, return_exceptions=True)

        return self.results

    def get_summary(self) -> dict:
        """Get a summary of review results."""
        successful = [r for r in self.results if r.is_success]
        skipped = [r for r in self.results if r.skipped]
        failed = [r for r in self.results if r.error]
        with_issues = [r for r in successful if r.has_issues]

        total_issues = sum(len(r.issues) for r in successful if r.issues)

        return {
            "total_files": len(self.results),
            "reviewed": len(successful),
            "skipped": len(skipped),
            "failed": len(failed),
            "files_with_issues": len(with_issues),
            "total_issues": total_issues,
        }
