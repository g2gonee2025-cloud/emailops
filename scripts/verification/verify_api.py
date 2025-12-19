import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir / "backend" / "src"))
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(str(Path("backend/src").resolve()))

from cortex.rag_api.routes_ingest import router as ingest_router
from cortex.rag_api.routes_search import router as search_router

app = FastAPI()
app.include_router(ingest_router)
app.include_router(search_router)

client = TestClient(app)


def verify_api_definitions():
    print("Verifying API Routes...")

    # Test Ingest Routes exists
    response = client.get("/ingest/s3/folders", params={"dry_run": "true"})
    # Since we don't have AWS creds mocked, this might 500 or 403, but getting a response
    # other than 404 means the route exists.
    print(f"GET /ingest/s3/folders status: {response.status_code}")
    assert response.status_code != 404, "Ingest route invalid"

    # Test Search Route existence
    # We send an invalid body to check that validation runs
    response = client.post("/search", json={})
    print(f"POST /search status (expected 422): {response.status_code}")
    assert response.status_code == 422, "Search route should validate input"


if __name__ == "__main__":
    verify_api_definitions()
