"""
Search Service Module

Handles all search-related business logic, abstracting it from the GUI layer.
"""

import logging
from pathlib import Path
from typing import Any

from emailops.feature_search_draft import (
    SearchFilters,
    _parse_iso_date,
    _search,
)

logger = logging.getLogger(__name__)


class SearchService:
    """Service for handling email search operations."""

    def __init__(self, export_root: str, index_dirname: str = ".email_index"):
        """
        Initialize the search service.

        Args:
            export_root: Root directory for email exports
            index_dirname: Name of the index directory
        """
        self.export_root = Path(export_root)
        self.index_dirname = index_dirname
        self.ix_dir = self.export_root / index_dirname

    def validate_index(self) -> tuple[bool, str]:
        """
        Validate that the index directory exists and is ready.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.ix_dir.exists():
            return (
                False,
                f"Index directory not found at {self.ix_dir}. Please build index first.",
            )

        mapping_path = self.ix_dir / "mapping.json"
        if not mapping_path.exists():
            return False, "Index mapping.json not found. Please rebuild index."

        return True, ""

    def build_search_filters(
        self,
        from_email: str = "",
        to_email: str = "",
        cc_email: str = "",
        subject_contains: str = "",
        file_types: str = "",
        has_attachment: str = "any",
        date_from: str = "",
        date_to: str = "",
    ) -> SearchFilters | None:
        """
        Build SearchFilters object from individual filter parameters.

        Args:
            from_email: Filter by sender email
            to_email: Filter by recipient email
            cc_email: Filter by CC email
            subject_contains: Filter by subject content
            file_types: Comma-separated file types (e.g., "pdf,docx")
            has_attachment: "yes", "no", or "any"
            date_from: ISO date string for start date
            date_to: ISO date string for end date

        Returns:
            SearchFilters object or None if no filters specified
        """
        # Return None if no filters are specified
        if not any(
            [
                from_email,
                to_email,
                cc_email,
                subject_contains,
                file_types,
                has_attachment != "any",
                date_from,
                date_to,
            ]
        ):
            return None

        filters = SearchFilters()

        if from_email:
            filters.from_emails = {from_email.lower().strip()}

        if to_email:
            filters.to_emails = {to_email.lower().strip()}

        if cc_email:
            filters.cc_emails = {cc_email.lower().strip()}

        if subject_contains:
            filters.subject_contains = [subject_contains.lower().strip()]

        if file_types:
            filters.types = {
                t.strip().lower().lstrip(".")
                for t in file_types.split(",")
                if t.strip()
            }

        if has_attachment == "yes":
            filters.has_attachment = True
        elif has_attachment == "no":
            filters.has_attachment = False

        if date_from:
            try:
                filters.date_from = _parse_iso_date(date_from)
            except Exception as e:
                logger.warning(f"Invalid date_from format: {date_from}. Error: {e}")

        if date_to:
            try:
                filters.date_to = _parse_iso_date(date_to)
            except Exception as e:
                logger.warning(f"Invalid date_to format: {date_to}. Error: {e}")

        return filters

    def perform_search(
        self,
        query: str,
        k: int = 10,
        provider: str = "vertex",
        mmr_lambda: float = 0.7,
        rerank_alpha: float = 0.35,
        filters: SearchFilters | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform a search operation.

        Args:
            query: Search query string
            k: Number of results to return
            provider: Embedding provider
            mmr_lambda: MMR lambda parameter for relevance vs diversity
            rerank_alpha: Reranking alpha parameter
            filters: Optional SearchFilters object

        Returns:
            List of search results

        Raises:
            ValueError: If query is empty or index is invalid
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        is_valid, error_msg = self.validate_index()
        if not is_valid:
            raise ValueError(error_msg)

        try:
            results = _search(
                ix_dir=self.ix_dir,
                query=query.strip(),
                k=k,
                provider=provider,
                mmr_lambda=mmr_lambda,
                rerank_alpha=rerank_alpha,
                filters=filters,
            )

            logger.info(
                f"Search completed: found {len(results)} results for query: {query}"
            )
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise RuntimeError(f"Search operation failed: {e}") from e

    def format_search_results(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Format search results for display.

        Args:
            results: Raw search results

        Returns:
            Formatted results with display-ready fields
        """
        formatted = []
        for result in results:
            formatted.append(
                {
                    "score": float(result.get("score", 0.0)),
                    "subject": result.get("subject", "No Subject"),
                    "id": result.get("id", "Unknown"),
                    "conv_id": result.get("conv_id", ""),
                    "type": result.get("type", ""),
                    "date": result.get("date", ""),
                    "text": result.get("text", "No snippet available"),
                    "from": result.get("from", ""),
                    "to": result.get("to", ""),
                }
            )
        return formatted

    def export_search_results_csv(
        self, results: list[dict[str, Any]], output_path: Path
    ) -> None:
        """
        Export search results to CSV file.

        Args:
            results: Search results to export
            output_path: Path to output CSV file
        """
        import csv

        if not results:
            raise ValueError("No results to export")

        try:
            with output_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Score",
                        "Document ID",
                        "Subject",
                        "Conv ID",
                        "Type",
                        "Date",
                        "From",
                        "To",
                        "Text Preview",
                    ]
                )

                for result in results:
                    writer.writerow(
                        [
                            f"{result.get('score', 0):.4f}",
                            result.get("id", ""),
                            result.get("subject", ""),
                            result.get("conv_id", ""),
                            result.get("type", ""),
                            result.get("date", ""),
                            result.get("from", ""),
                            result.get("to", ""),
                            (
                                (result.get("text", "")[:200] + "...")
                                if len(result.get("text", "")) > 200
                                else result.get("text", "")
                            ),
                        ]
                    )

            logger.info(f"Exported {len(results)} search results to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export search results: {e}", exc_info=True)
            raise RuntimeError(f"Failed to export search results: {e}") from e
