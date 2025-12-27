import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

# Ensure backend/src is on the path for importing middleware
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from main import CorrelationIdMiddleware


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/echo-correlation")
    async def echo(request: Request):
        return {"correlation_id": getattr(request.state, "correlation_id", None)}

    return app


def test_correlation_id_added_to_request_state_and_headers():
    client = TestClient(_build_test_app())

    response = client.get("/echo-correlation")

    assert response.status_code == 200
    correlation_id = response.json()["correlation_id"]
    assert correlation_id  # middleware should inject a value
    assert response.headers["X-Correlation-ID"] == correlation_id


def test_correlation_id_respects_incoming_header():
    client = TestClient(_build_test_app())
    incoming = "test-correlation-id-123"

    response = client.get("/echo-correlation", headers={"X-Correlation-ID": incoming})

    assert response.status_code == 200
    assert response.json()["correlation_id"] == incoming
    assert response.headers["X-Correlation-ID"] == incoming
