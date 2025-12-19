import sys
from pathlib import Path

import pytest
from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "backend" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import main  # noqa: E402


@pytest.mark.asyncio
async def test_extract_identity_supports_sync_decoder(monkeypatch: pytest.MonkeyPatch):
    def sync_decoder(token: str) -> dict[str, str]:
        assert token == "token123"
        return {"tenant_id": "acme", "sub": "user@example.com"}

    monkeypatch.setattr(main, "_jwt_decoder", sync_decoder, raising=False)

    scope = {
        "type": "http",
        "headers": [(b"authorization", b"Bearer token123")],
    }
    request = Request(scope)

    tenant_id, user_id, claims = await main._extract_identity(request)

    assert tenant_id == "acme"
    assert user_id == "user@example.com"
    assert claims["sub"] == "user@example.com"
