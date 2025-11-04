"""
Chat Service Module

Handles chat operations with context from email search.
"""

import logging
from pathlib import Path
from typing import Any

from emailops.feature_search_draft import (
    ChatSession,
    _search,
    chat_with_context,
)

logger = logging.getLogger(__name__)


class ChatService:
    """Service for handling chat operations."""

    def __init__(self, export_root: str, index_dirname: str = ".email_index"):
        """
        Initialize the chat service.

        Args:
            export_root: Root directory for email exports
            index_dirname: Name of the index directory
        """
        self.export_root = Path(export_root)
        self.index_dirname = index_dirname
        self.index_dir = self.export_root / index_dirname
        self.sessions: dict[str, ChatSession] = {}

    def get_or_create_session(
        self, session_id: str = "default", max_history: int = 5
    ) -> ChatSession:
        """
        Get an existing session or create a new one.

        Args:
            session_id: Session identifier
            max_history: Maximum history entries to maintain

        Returns:
            ChatSession instance
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(
                base_dir=self.index_dir, session_id=session_id, max_history=max_history
            )
            # Load existing session if it exists
            self.sessions[session_id].load()

        return self.sessions[session_id]

    def chat_with_query(
        self,
        query: str,
        session_id: str = "default",
        k: int = 10,
        provider: str = "vertex",
        temperature: float = 0.7,
        max_history: int = 5,
    ) -> dict[str, Any]:
        """
        Process a chat query with context from email search.

        Args:
            query: User's question
            session_id: Chat session identifier
            k: Number of context snippets to retrieve
            provider: LLM provider
            temperature: LLM temperature
            max_history: Maximum chat history to maintain

        Returns:
            Dictionary containing answer and citations

        Raises:
            ValueError: If query is empty
            RuntimeError: If chat operation fails
        """
        if not query or not query.strip():
            raise ValueError("Chat query cannot be empty")

        if not self.index_dir.exists():
            raise ValueError(f"Index directory not found at {self.index_dir}")

        try:
            # Get or create session
            session = self.get_or_create_session(session_id, max_history)

            # Search for context
            logger.info(f"Searching for context with query: {query}")
            context_snippets = _search(
                ix_dir=self.index_dir, query=query.strip(), k=k, provider=provider
            )

            # Get recent chat history
            chat_history = session.recent()

            # Generate response with context
            logger.info("Generating chat response with context")
            result = chat_with_context(
                query=query.strip(),
                context_snippets=context_snippets,
                chat_history=chat_history,
                temperature=temperature,
            )

            # Update session history
            session.add_message("user", query.strip())
            session.add_message("assistant", result.get("answer", ""))
            session.save()

            # Add metadata to result
            result["session_id"] = session_id
            result["context_count"] = len(context_snippets)
            result["history_count"] = len(chat_history)

            logger.info(f"Chat completed with {len(context_snippets)} context snippets")
            return result

        except Exception as e:
            logger.error(f"Chat operation failed: {e}", exc_info=True)
            raise RuntimeError(f"Chat operation failed: {e}") from e

    def get_session_history(self, session_id: str = "default") -> list[dict[str, str]]:
        """
        Get chat history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        session = self.sessions.get(session_id)
        if not session:
            # Try to load from disk
            session = ChatSession(
                base_dir=self.index_dir, session_id=session_id, max_history=5
            )
            session.load()
            self.sessions[session_id] = session

        return [{"role": msg.role, "content": msg.content} for msg in session.messages]

    def clear_session(self, session_id: str = "default") -> bool:
        """
        Clear a chat session.

        Args:
            session_id: Session identifier

        Returns:
            True if successful
        """
        try:
            session = self.get_or_create_session(session_id)
            session.reset()
            session.save()
            logger.info(f"Cleared chat session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            return False

    def save_session(self, session_id: str = "default") -> bool:
        """
        Save a chat session to disk.

        Args:
            session_id: Session identifier

        Returns:
            True if successful
        """
        try:
            session = self.sessions.get(session_id)
            if session:
                session.save()
                logger.info(f"Saved chat session: {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            return False

    def load_session(self, session_id: str = "default", max_history: int = 5) -> bool:
        """
        Load a chat session from disk.

        Args:
            session_id: Session identifier
            max_history: Maximum history entries

        Returns:
            True if successful
        """
        try:
            session = ChatSession(
                base_dir=self.index_dir, session_id=session_id, max_history=max_history
            )
            session.load()
            self.sessions[session_id] = session
            logger.info(f"Loaded chat session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return False

    def list_sessions(self) -> list[str]:
        """
        List available chat sessions.

        Returns:
            List of session IDs
        """
        session_dir = self.index_dir / "chat_sessions"
        if not session_dir.exists():
            return []

        sessions = []
        for session_file in session_dir.glob("*.json"):
            session_id = session_file.stem
            sessions.append(session_id)

        return sorted(sessions)

    def get_session_statistics(self, session_id: str = "default") -> dict[str, Any]:
        """
        Get statistics about a chat session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary containing session statistics
        """
        session = self.sessions.get(session_id)
        if not session:
            # Try to load from disk
            session = ChatSession(
                base_dir=self.index_dir, session_id=session_id, max_history=5
            )
            try:
                session.load()
            except Exception:
                # Session doesn't exist
                return {
                    "exists": False,
                    "session_id": session_id,
                    "message_count": 0,
                    "user_messages": 0,
                    "assistant_messages": 0,
                }

        user_count = sum(1 for msg in session.messages if msg.role == "user")
        assistant_count = sum(1 for msg in session.messages if msg.role == "assistant")

        return {
            "exists": True,
            "session_id": session_id,
            "message_count": len(session.messages),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "max_history": session.max_history,
        }

    def export_session_history(self, session_id: str, output_path: Path) -> bool:
        """
        Export session history to a text file.

        Args:
            session_id: Session identifier
            output_path: Path to save the history

        Returns:
            True if successful
        """
        try:
            history = self.get_session_history(session_id)

            with output_path.open("w", encoding="utf-8") as f:
                f.write(f"Chat Session: {session_id}\n")
                f.write("=" * 80 + "\n\n")

                for msg in history:
                    role = msg["role"].capitalize()
                    content = msg["content"]
                    f.write(f"[{role}]\n{content}\n\n")
                    f.write("-" * 40 + "\n\n")

            logger.info(f"Exported session {session_id} to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export session: {e}", exc_info=True)
            return False

    def format_citations(self, citations: list[dict[str, Any]]) -> str:
        """
        Format citations for display.

        Args:
            citations: List of citation dictionaries

        Returns:
            Formatted string
        """
        if not citations:
            return "No citations available."

        lines = [f"Citations ({len(citations)}):"]
        for i, cite in enumerate(citations[:10], 1):  # Limit to 10
            doc_id = cite.get("document_id", "Unknown")
            score = cite.get("relevance_score", 0.0)
            snippet = cite.get("text", "")[:100]

            lines.append(f"{i}. [{score:.3f}] {doc_id}")
            if snippet:
                lines.append(f"   {snippet}...")

        return "\n".join(lines)
