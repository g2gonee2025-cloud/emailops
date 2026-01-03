import sys
from pathlib import Path


def find_project_root(marker: str = "pyproject.toml") -> Path:
    """Locate the project root by searching for a marker file."""
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent:
        if (current_dir / marker).exists():
            return current_dir
        current_dir = current_dir.parent
    raise FileNotFoundError(f"Project root marker '{marker}' not found.")


# Add the backend/src directory to the Python path
try:
    project_root = find_project_root()
    src_path = project_root / "backend" / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
except FileNotFoundError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)


from fastapi import FastAPI
from fastapi.testclient import TestClient


from cortex.rag_api.routes_ingest import router as ingest_router
from cortex.rag_api.routes_search import router as search_router

app = FastAPI()
app.include_router(ingest_router)
app.include_router(search_router)


def verify_api_definitions():
    print("Verifying API Routes...")
    with TestClient(app, raise_server_exceptions=False) as client:
        # Test Ingest Routes exists
        response = client.get("/ingest/s3/folders", params={"dry_run": "true"})
        # Since we don't have AWS creds mocked, this might 500 or 403, but getting a response
        # other than 404 means the route exists.
        print(f"GET /ingest/s3/folders status: {response.status_code}")
        if response.status_code == 404:
            print("Error: Ingest route '/ingest/s3/folders' not found (404).", file=sys.stderr)
            sys.exit(1)

        # Test Search Route existence
        # We send an invalid body to check that validation runs
        response = client.post("/search", json={})
        print(f"POST /search status (expected 422): {response.status_code}")
        if response.status_code != 422:
            print(f"Error: Search route '/search' returned unexpected status {response.status_code}. Expected 422.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    verify_api_definitions()
