"""
Verify the Thread Summarization functionality.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import patch

# Ensure backend/src is added to sys.path securely relative to this file
_repo_root = Path(__file__).resolve().parents[2]
_backend_src = (_repo_root / "backend" / "src").resolve()
if _backend_src.is_dir():
    _backend_src_str = str(_backend_src)
    if _backend_src_str not in sys.path:
        sys.path.insert(0, _backend_src_str)
else:
    raise RuntimeError(f"Expected backend/src at {_backend_src} but it was not found")

from cortex.db.models import Chunk  # noqa: E402
from cortex.intelligence.summarizer import ConversationSummarizer  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_conversation_summarizer():
    """Test the summarizer logic in isolation."""
    # Mock LLM Runtime
    with patch("cortex.intelligence.summarizer.LLMRuntime") as MockLLM:
        import numpy as np

        mock_instance = MockLLM.return_value
        mock_instance.complete_text.return_value = (
            "This is a summary of the project discussion."
        )
        mock_instance.embed_documents.return_value = np.array([[0.1, 0.2, 0.3]])

        summarizer = ConversationSummarizer()

        # Test generation
        text = "Message 1: We need to launch Project X.\nMessage 2: Agreed, let's do it on Friday."
        summary = summarizer.generate_summary(text)

        assert summary == "This is a summary of the project discussion."
        mock_instance.complete_text.assert_called_once()

        # Test embedding
        emb = summarizer.embed_summary(summary)
        assert emb == [0.1, 0.2, 0.3]

        logger.info("ConversationSummarizer unit test passed!")


def verify_db_schema():
    """Verify that the DB schema has the new column."""
    # We can inspect the model class directly
    assert hasattr(Chunk, "is_summary"), "Chunk model missing is_summary column"
    assert hasattr(Chunk, "chunk_type"), "Chunk model missing chunk_type column"
    logger.info("DB Schema verification passed!")


if __name__ == "__main__":
    try:
        verify_db_schema()
        test_conversation_summarizer()
        print("VERIFICATION SUCCESSFUL")
    except Exception as e:
        print(f"VERIFICATION FAILED: {e}")
        sys.exit(1)
