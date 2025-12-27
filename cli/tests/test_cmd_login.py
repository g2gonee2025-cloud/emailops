import argparse
from unittest.mock import MagicMock, patch

import httpx
import pytest
from cortex_cli.cmd_login import _run_login


@patch("httpx.Client")
def test_run_login_success(mock_client):
    """Test successful login."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"access_token": "test_token"}
    mock_response.raise_for_status.return_value = None

    mock_post = MagicMock(return_value=mock_response)
    mock_client.return_value.__enter__.return_value.post = mock_post

    args = argparse.Namespace(
        username="admin",
        password="admin",
        host="http://localhost:8000",
    )

    with patch("builtins.print") as mock_print:
        _run_login(args)
        mock_print.assert_any_call("Login successful!")
        mock_print.assert_any_call("Access Token: test_token")


@patch("httpx.Client")
def test_run_login_failure(mock_client):
    """Test failed login."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Invalid credentials"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Invalid credentials", request=MagicMock(), response=mock_response
    )

    mock_post = MagicMock(return_value=mock_response)
    mock_client.return_value.__enter__.return_value.post = mock_post

    args = argparse.Namespace(
        username="admin",
        password="wrong_password",
        host="http://localhost:8000",
    )

    with patch("builtins.print") as mock_print:
        _run_login(args)
        mock_print.assert_any_call("Login failed: 401 Invalid credentials")


@patch("httpx.Client")
def test_run_login_request_error(mock_client):
    """Test request error during login."""
    mock_request = MagicMock()
    mock_request.url = "http://localhost:8000/auth/login"
    mock_post = MagicMock(
        side_effect=httpx.RequestError("Connection error", request=mock_request)
    )
    mock_client.return_value.__enter__.return_value.post = mock_post

    args = argparse.Namespace(
        username="admin",
        password="admin",
        host="http://localhost:8000",
    )

    with patch("builtins.print") as mock_print:
        _run_login(args)
        mock_print.assert_any_call(
            "An error occurred while requesting 'http://localhost:8000/auth/login'."
        )
