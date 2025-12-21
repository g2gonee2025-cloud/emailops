from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from cortex.domain_models.facts_ledger import FactsLedger  # noqa: E402
from cortex.domain_models.rag import ThreadSummary  # noqa: E402
from cortex.rag_api import routes_summarize  # noqa: E402
from main import app  # noqa: E402


class DummyGraph:
    async def ainvoke(self, state):
        return {
            "summary": ThreadSummary(
                thread_id=state.get("thread_id"),
                summary_markdown="Summarized!",
                facts_ledger=FactsLedger(),
                quality_scores={"coherence": 0.9},
            )
        }


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


def test_summarize_endpoint_returns_graph_summary(monkeypatch):
    # P0 Fix: Ensure we mock the graph in app.state if it exists, as that takes precedence
    dummy_graph = DummyGraph()
    monkeypatch.setattr(routes_summarize, "_summarize_graph", dummy_graph)

    thread_id = str(uuid4())

    with TestClient(app) as client:
        # Inject into app.state to bypass lifespan-loaded real graph
        if not hasattr(app.state, "graphs"):
            app.state.graphs = {}
        app.state.graphs["summarize"] = dummy_graph

        response = client.post("/api/v1/summarize", json={"thread_id": thread_id})

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["summary_markdown"] == "Summarized!"
    assert payload["summary"]["facts_ledger"]["asks"] == []
