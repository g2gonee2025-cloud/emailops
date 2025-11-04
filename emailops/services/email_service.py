"""
Email Service Module

Handles all email-related business logic including drafting replies,
creating fresh emails, and managing email operations.
"""

import logging
from pathlib import Path
from typing import Any

from emailops.feature_search_draft import (
    draft_email_reply_eml,
    draft_fresh_email_eml,
)

logger = logging.getLogger(__name__)


class EmailService:
    """Service for handling email operations."""

    def __init__(self, export_root: str):
        """
        Initialize the email service.

        Args:
            export_root: Root directory for email exports
        """
        self.export_root = Path(export_root)

    def draft_reply(
        self,
        conv_id: str,
        query: str = "",
        provider: str = "vertex",
        sim_threshold: float = 0.5,
        target_tokens: int = 20000,
        temperature: float = 0.7,
        reply_policy: str = "reply_all",
        include_attachments: bool = True,
    ) -> dict[str, Any]:
        """
        Generate a draft reply for a conversation.

        Args:
            conv_id: Conversation ID
            query: Optional query for context
            provider: LLM provider
            sim_threshold: Similarity threshold
            target_tokens: Maximum tokens for response
            temperature: LLM temperature
            reply_policy: "reply_all", "smart", or "sender_only"
            include_attachments: Whether to include attachments

        Returns:
            Dictionary containing eml_bytes and metadata

        Raises:
            ValueError: If conv_id is empty or invalid
            RuntimeError: If draft generation fails
        """
        if not conv_id or not conv_id.strip():
            raise ValueError("Conversation ID cannot be empty")

        # Validate conversation path
        conv_path = self._validate_conversation_path(conv_id)
        if not conv_path:
            raise ValueError(f"Invalid conversation ID: {conv_id}")

        try:
            result = draft_email_reply_eml(
                export_root=self.export_root,
                conv_id=conv_id.strip(),
                query=query.strip(),
                provider=provider,
                sim_threshold=sim_threshold,
                target_tokens=target_tokens,
                temperature=temperature,
                reply_policy=reply_policy,
                include_attachments=include_attachments,
            )

            logger.info(f"Generated reply draft for conversation: {conv_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to generate reply draft: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate reply: {e}") from e

    def draft_fresh_email(
        self,
        to_list: list[str],
        cc_list: list[str],
        subject: str,
        query: str,
        provider: str = "vertex",
        target_tokens: int = 10000,
        temperature: float = 0.7,
        include_attachments: bool = True,
    ) -> dict[str, Any]:
        """
        Generate a fresh email draft.

        Args:
            to_list: List of recipient email addresses
            cc_list: List of CC email addresses
            subject: Email subject
            query: Intent/instructions for email content
            provider: LLM provider
            target_tokens: Maximum tokens for response
            temperature: LLM temperature
            include_attachments: Whether to include attachments

        Returns:
            Dictionary containing eml_bytes and metadata

        Raises:
            ValueError: If query is empty
            RuntimeError: If draft generation fails
        """
        if not query or not query.strip():
            raise ValueError("Email intent/instructions cannot be empty")

        try:
            result = draft_fresh_email_eml(
                export_root=self.export_root,
                provider=provider,
                to_list=[t.strip() for t in to_list if t.strip()],
                cc_list=[c.strip() for c in cc_list if c.strip()],
                subject=subject.strip(),
                query=query.strip(),
                target_tokens=target_tokens,
                temperature=temperature,
                include_attachments=include_attachments,
            )

            logger.info(f"Generated fresh email draft with subject: {subject}")
            return result

        except Exception as e:
            logger.error(f"Failed to generate fresh email: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate fresh email: {e}") from e

    def save_draft_eml(self, eml_bytes: bytes, output_path: Path) -> None:
        """
        Save email draft to .eml file.

        Args:
            eml_bytes: Email content in bytes
            output_path: Path to save the .eml file

        Raises:
            ValueError: If eml_bytes is empty
            RuntimeError: If save operation fails
        """
        if not eml_bytes:
            raise ValueError("No email content to save")

        try:
            output_path.write_bytes(eml_bytes)
            logger.info(f"Saved email draft to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save email draft: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save email draft: {e}") from e

    def extract_email_preview(self, result: dict[str, Any]) -> str:
        """
        Extract preview text from email generation result.

        Args:
            result: Email generation result dictionary

        Returns:
            Preview text or default message
        """
        return result.get("body_preview", "No preview available")

    def validate_email_addresses(
        self, addresses: list[str]
    ) -> tuple[list[str], list[str]]:
        """
        Validate email addresses.

        Args:
            addresses: List of email addresses to validate

        Returns:
            Tuple of (valid_addresses, invalid_addresses)
        """
        import re

        email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

        valid = []
        invalid = []

        for addr in addresses:
            addr = addr.strip()
            if addr and email_pattern.match(addr):
                valid.append(addr)
            elif addr:
                invalid.append(addr)

        return valid, invalid

    def _validate_conversation_path(self, conv_id: str) -> Path | None:
        """
        Validate and resolve conversation path.

        Args:
            conv_id: Conversation ID

        Returns:
            Path object if valid, None otherwise
        """
        try:
            root = self.export_root.resolve()
            candidate = (root / conv_id).resolve()

            # Check if path is within export root
            try:
                candidate.relative_to(root)
            except ValueError:
                return None

            if candidate.exists() and candidate.is_dir():
                # Check for required conversation files
                conv_file = candidate / "Conversation.txt"
                if conv_file.exists():
                    return candidate

            return None

        except Exception as e:
            logger.error(f"Failed to validate conversation path: {e}")
            return None

    def get_reply_policy_options(self) -> list[str]:
        """
        Get available reply policy options.

        Returns:
            List of reply policy options
        """
        return ["reply_all", "smart", "sender_only"]

    def get_default_tokens(self, operation: str) -> int:
        """
        Get default token limit for an operation.

        Args:
            operation: "reply" or "fresh"

        Returns:
            Default token limit
        """
        defaults = {
            "reply": 20000,
            "fresh": 10000,
        }
        return defaults.get(operation, 10000)
