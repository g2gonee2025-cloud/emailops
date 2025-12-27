import json
from unittest.mock import ANY, MagicMock, patch

import httpx
import pytest
from cortex.health import DoctorCheckResult
from cortex_cli.cmd_doctor import main


@pytest.fixture
def mock_httpx_client():
    """Provides a mock httpx.Client."""
    with patch("httpx.Client") as mock_client_class:
        mock_client = mock_client_class.return_value.__enter__.return_value
        yield mock_client


@patch.dict("os.environ", {"OUTLOOKCORTEX_DB_URL": "sqlite:///:memory:"})
@patch("cortex.routes_admin.asyncio.gather")
def test_doctor_healthy_human_output(mock_gather, mock_httpx_client, capsys):
    """Test the doctor command with a healthy API response."""
    mock_gather.return_value = [
        DoctorCheckResult(name="PostgreSQL", status="pass", message="Connected"),
        DoctorCheckResult(name="Redis", status="pass", message="Connected"),
        DoctorCheckResult(
            name="Embeddings API", status="pass", message="Dimension: 384"
        ),
        DoctorCheckResult(name="Reranker API", status="pass", message="Connected"),
    ]

    report = {
        "overall_status": "healthy",
        "checks": [check.model_dump() for check in mock_gather.return_value],
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = report
    mock_httpx_client.post.return_value = mock_response

    with (
        patch("sys.argv", ["cortex_cli/cmd_doctor.py"]),
        pytest.raises(SystemExit) as e,
    ):
        main()

    assert e.value.code == 0
    captured = capsys.readouterr()
    assert "Overall Status: HEALTHY" in captured.out
    assert "✓ PostgreSQL: Connected" in captured.out
    assert "✓ Redis: Connected" in captured.out
    assert "✓ Embeddings API: Dimension: 384" in captured.out
    assert "✓ Reranker API: Connected" in captured.out


@patch.dict("os.environ", {"OUTLOOKCORTEX_DB_URL": "sqlite:///:memory:"})
def test_doctor_degraded_human_output(mock_httpx_client, capsys):
    """Test the doctor command with a degraded API response."""
    report = {
        "overall_status": "degraded",
        "checks": [
            {"name": "PostgreSQL", "status": "pass", "message": "Connected"},
            {
                "name": "Reranker API",
                "status": "warn",
                "message": "Connection failed",
            },
        ],
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = report
    mock_httpx_client.post.return_value = mock_response

    with (
        patch("sys.argv", ["cortex_cli/cmd_doctor.py"]),
        pytest.raises(SystemExit) as e,
    ):
        main()

    assert e.value.code == 1
    captured = capsys.readouterr()
    assert "Overall Status: DEGRADED" in captured.out
    assert "⚠ Reranker API: Connection failed" in captured.out


@patch.dict("os.environ", {"OUTLOOKCORTEX_DB_URL": "sqlite:///:memory:"})
def test_doctor_unhealthy_json_output(mock_httpx_client, capsys):
    """Test the doctor command with an unhealthy response and JSON flag."""
    report = {
        "overall_status": "unhealthy",
        "checks": [
            {"name": "PostgreSQL", "status": "fail", "message": "Connection refused"}
        ],
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = report
    mock_httpx_client.post.return_value = mock_response

    with (
        patch("sys.argv", ["cortex_cli/cmd_doctor.py", "--json"]),
        pytest.raises(SystemExit) as e,
    ):
        main()

    assert e.value.code == 2
    captured = capsys.readouterr()
    # The output should be the raw JSON report
    parsed_output = json.loads(captured.out)
    assert parsed_output == report


@patch.dict("os.environ", {"OUTLOOKCORTEX_DB_URL": "sqlite:///:memory:"})
def test_doctor_connection_error(mock_httpx_client, capsys):
    """Test the doctor command when it fails to connect to the API."""
    mock_httpx_client.post.side_effect = httpx.RequestError("Connection failed")

    with (
        patch("sys.argv", ["cortex_cli/cmd_doctor.py"]),
        pytest.raises(SystemExit) as e,
    ):
        main()

    assert e.value.code == 2
    captured = capsys.readouterr()
    assert "Could not connect to the Cortex backend" in captured.out
