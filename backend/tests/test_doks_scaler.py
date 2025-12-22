from unittest.mock import MagicMock, patch

import pytest
from cortex.common.exceptions import ConfigurationError
from cortex.llm.doks_scaler import DOApiClient


class TestDOApiClient:
    def test_init_missing_token(self):
        with pytest.raises(ConfigurationError):
            DOApiClient(token="")

    def test_init_dry_run(self):
        client = DOApiClient(token="", dry_run=True)
        assert client.dry_run is True

    def test_request_success(self):
        client = DOApiClient(token="test-token")
        with patch("requests.Session.request") as mock_req:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"foo": "bar"}
            mock_req.return_value = mock_resp

            data = client.request("GET", "/account")
            assert data == {"foo": "bar"}
            # Check URL construction uses base_url (lowercase)
            mock_req.assert_called_with(
                "GET", "https://api.digitalocean.com/v2/account", timeout=30
            )

    def test_pagination(self):
        client = DOApiClient(token="test-token")
        with patch.object(client, "request") as mock_request:
            mock_request.side_effect = [
                {
                    "items": [{"id": 1}],
                    "links": {
                        "pages": {"next": "https://api.digitalocean.com/v2/next"}
                    },
                },
                {"items": [{"id": 2}], "links": {}},
            ]

            items = client._paginate("/items", "items")
            assert len(items) == 2
            assert items[0]["id"] == 1
            assert items[1]["id"] == 2

            # Verify correct calls including stripped URL logic check
            assert mock_request.call_count == 2
            mock_request.assert_any_call("GET", "/items")
            mock_request.assert_any_call("GET", "/next")
