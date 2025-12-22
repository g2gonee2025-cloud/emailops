"""
Conversation Summarizer.

Generates high-level summaries of conversation threads to enable
"topic-aware" retrieval.
"""

from __future__ import annotations

import logging

from cortex.llm.runtime import LLMRuntime

logger = logging.getLogger(__name__)


class ConversationSummarizer:
    """
    Summarizes text content using the LLM.
    Pure intelligence service: Input Text -> Output Summary.
    """

    MAX_CONTEXT_TOKENS = 8192  # Safety limit for context window

    def __init__(self):
        self.llm = LLMRuntime()

    def generate_summary(self, text: str) -> str:
        """
        Generate a summary for the given text.
        Returns empty string if input is empty or generation fails.
        """
        if not text or not text.strip():
            return ""

        # Truncate naive token estimation (4 chars/token approx)
        max_chars = self.MAX_CONTEXT_TOKENS * 4
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...(truncated)..."

        prompt = (
            "You are an expert AI assistant analyzing an email conversation thread.\n"
            "Generate a comprehensive summary of the following conversation.\n"
            "Focus on:\n"
            "1. The main topic or request.\n"
            "2. Key decisions made.\n"
            "3. Action items or next steps.\n"
            "4. Important entities (people, dates, project names).\n\n"
            "Conversation:\n"
            f"{text}\n\n"
            "Summary:"
        )

        try:
            # Using a deterministic temperature for summaries
            return self.llm.complete_text(prompt, temperature=0.2)
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return ""

    def embed_summary(self, summary: str) -> list[float]:
        """Embed the summary text."""
        if not summary:
            return []
        try:
            embeddings = self.llm.embed_documents([summary])
            if len(embeddings) > 0:
                return embeddings[0].tolist()
        except Exception as e:
            logger.error(f"Summary embedding failed: {e}")
        return []
