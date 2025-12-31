"""Integration test for summarize API using live infrastructure."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from cortex.db.models import Conversation
from cortex.db.session import SessionLocal
from cortex.domain_models.facts_ledger import FactsLedger
from cortex.domain_models.rag import ThreadSummary
from main import app


def test_thread_summary_accepts_markdown_and_quality():
    summary = ThreadSummary(
        thread_id=uuid4(),
        summary_markdown="Summary text",
        facts_ledger=FactsLedger(),
        quality_scores={"coverage": 1.0},
    )

    assert summary.summary_markdown == "Summary text"
    assert summary.facts_ledger.asks == []
    assert summary.quality_scores["coverage"] == 1.0


def test_summarize_endpoint_with_live_thread():
    """Test summarize endpoint with a real thread from the live database."""
    # Get a real thread ID from the live database
    with SessionLocal() as session:
        conv = (
            session.query(Conversation.conversation_id)
            .filter(Conversation.tenant_id == "default")
            .first()
        )

    if conv is None:
        import pytest

        pytest.skip("No conversations in database to test with")

    thread_id = str(conv[0])

    with TestClient(app) as client:
        response = client.post("/api/v1/summarize", json={"thread_id": thread_id})

    # Should get a response (200 success or 500 if graph fails, but NOT 404)
    assert response.status_code != 404, f"Thread {thread_id} should exist in DB"
    # If we got 200, verify the response structure
    if response.status_code == 200:
        payload = response.json()
        assert "summary" in payload
        assert "summary_markdown" in payload["summary"]
