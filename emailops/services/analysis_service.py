"""
Analysis Service Module

Handles email thread analysis and summarization operations.
"""

import asyncio
import csv
import logging
from pathlib import Path
from typing import Any

from emailops import feature_summarize as summarizer

from .base_service import BaseService

logger = logging.getLogger(__name__)


class AnalysisService(BaseService):
    """Service for handling email thread analysis and summarization."""

    def __init__(self, export_root: str):
        """
        Initialize the analysis service.

        Args:
            export_root: Root directory for email exports
        """
        super().__init__(export_root)

    def analyze_conversation(
        self,
        thread_dir: Path,
        temperature: float = 0.7,
        merge_manifest: bool = True,
        output_format: str = "json",
    ) -> dict[str, Any]:
        """
        Analyze a conversation thread.

        Args:
            thread_dir: Path to conversation directory
            temperature: LLM temperature for analysis
            merge_manifest: Whether to merge manifest data
            output_format: Output format ("json", "markdown", or "both")

        Returns:
            Dictionary containing analysis results

        Raises:
            ValueError: If thread_dir is invalid
            RuntimeError: If analysis fails
        """
        # Validate conversation directory
        if not thread_dir.exists():
            raise ValueError(f"Conversation directory does not exist: {thread_dir}")

        conv_file = thread_dir / "Conversation.txt"
        if not conv_file.exists():
            raise ValueError(
                f"Invalid conversation directory - missing Conversation.txt: {thread_dir}"
            )

        try:
            logger.info(f"Analyzing conversation: {thread_dir.name}")

            # Run async analysis
            analysis = asyncio.run(
                summarizer.analyze_conversation_dir(
                    thread_dir=thread_dir,
                    temperature=temperature,
                    merge_manifest=merge_manifest,
                )
            )

            # Process output format
            result = {
                "conv_id": thread_dir.name,
                "analysis": analysis,
                "output_format": output_format,
                "files_created": [],
            }

            # Save output files based on format
            if output_format in ["json", "both"]:
                json_path = thread_dir / "summary.json"
                if self.file_service.save_json(analysis, json_path):
                    result["files_created"].append(str(json_path))

            if output_format in ["markdown", "both"]:
                md_content = summarizer.format_analysis_as_markdown(analysis)
                md_path = thread_dir / "summary.md"
                if self.file_service.save_text_file(md_content, md_path):
                    result["files_created"].append(str(md_path))

            logger.info(f"Analysis completed for {thread_dir.name}")
            return result

        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Validation error during analysis: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to analyze conversation: {e}", exc_info=True)
            raise RuntimeError(f"Analysis failed: {e}") from e

    def export_todos_csv(self, thread_dir: Path, todos: list[dict[str, Any]]) -> bool:
        """
        Export action items to CSV file.

        Args:
            thread_dir: Conversation directory
            todos: List of action items

        Returns:
            True if successful, False otherwise
        """
        if not todos:
            logger.info("No todos to export")
            return True

        try:
            summarizer.append_todos_to_csv(thread_dir.parent, thread_dir.name, todos)
            logger.info(f"Exported {len(todos)} todos to CSV")
            return True

        except (OSError, csv.Error) as e:
            logger.error(f"Failed to export todos to CSV: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during todo export: {e}", exc_info=True)
            return False

    def format_as_markdown(self, analysis: dict[str, Any]) -> str:
        """
        Format analysis results as markdown.

        Args:
            analysis: Analysis dictionary

        Returns:
            Markdown formatted string
        """
        try:
            return summarizer.format_analysis_as_markdown(analysis)
        except Exception as e:
            logger.error(f"Failed to format as markdown: {e}", exc_info=True)
            return f"# Error\n\nFailed to format analysis: {e}"

    def get_analysis_statistics(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """
        Extract statistics from analysis results.

        Args:
            analysis: Analysis dictionary

        Returns:
            Dictionary containing statistics
        """
        stats = {
            "has_summary": bool(analysis.get("brief_summary")),
            "has_detailed_summary": bool(analysis.get("detailed_summary")),
            "num_next_actions": len(analysis.get("next_actions", [])),
            "num_recipients": len(analysis.get("recipients", [])),
            "num_attachments": len(analysis.get("attachments", [])),
            "has_decision_points": bool(analysis.get("decision_points")),
            "has_issues": bool(analysis.get("issues_raised")),
            "email_count": analysis.get("email_count", 0),
        }

        # Extract topic if available
        if "detailed_summary" in analysis:
            detailed = analysis["detailed_summary"]
            if isinstance(detailed, dict):
                stats["topic"] = detailed.get("topic", "")
                stats["sentiment"] = detailed.get("sentiment", "")

        return stats
