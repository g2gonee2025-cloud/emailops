from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from main import app  # noqa: E402
from cortex.rag_api import routes_summarize

def test_summarize_endpoint_returns_service_summary():
    """
    Verify the endpoint correctly calls the summarize service and returns its result.
    """
    thread_id = str(uuid4())
    expected_summary = {
        "type": "thread_summary",
        "thread_id": thread_id,
        "summary_markdown": "Summarized!",
        "facts_ledger": {
            "asks": [],
            "commitments": [],
            "key_dates": [],
            "key_decisions": [],
            "open_questions": [],
            "risks_concerns": [],
            "participants": [],
        },
        "quality_scores": {"coherence": 0.9},
        "key_points": [],
        "action_items": [],
        "participants": [],
    }

    # Patch the service layer, which is now the dependency of the route
    with patch(
        "cortex.rag_api.routes_summarize.summarize_thread_service",
        new_callable=AsyncMock,
        return_value=expected_summary,
    ) as mock_summarize_service:
        with TestClient(app) as client:
            # The dependency injection needs to be overridden for the test
            # to bypass the actual authentication. We provide a dummy user.
            app.dependency_overrides[
                routes_summarize.get_current_user
            ] = lambda: "test_user"

            response = client.post("/api/v1/summarize", json={"thread_id": thread_id})

            # Clean up the override after the test
            app.dependency_overrides = {}

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == expected_summary
    mock_summarize_service.assert_awaited_once()

# Keep other tests if they are still relevant
from cortex.domain_models.facts_ledger import FactsLedger
from cortex.domain_models.rag import ThreadSummary

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
