import argparse

# Make imports work as if running from the project root
import sys
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest

# Add backend and CLI source paths
backend_src = str(Path(__file__).resolve().parents[2] / "backend/src")
cli_src = str(Path(__file__).resolve().parents[2] / "cli/src")
if backend_src not in sys.path:
    sys.path.insert(0, backend_src)
if cli_src not in sys.path:
    sys.path.insert(0, cli_src)

from cortex.safety.grounding import ClaimAnalysis, GroundingCheck
from cortex_cli.cmd_grounding import run_grounding_check

# --- Mocks and Fixtures ---


@pytest.fixture
def mock_tool_check_grounding():
    """Fixture to mock the core grounding logic."""
    with patch("cortex_cli.cmd_grounding.tool_check_grounding") as mock_tool:
        yield mock_tool


@pytest.fixture
def mock_console_and_table():
    """Fixture to mock both the Rich Console and Table."""
    with (
        patch("cortex_cli.cmd_grounding.Console") as mock_console_class,
        patch("cortex_cli.cmd_grounding.Table") as mock_table_class,
    ):
        mock_console_instance = MagicMock()
        mock_table_instance = MagicMock()
        mock_console_class.return_value = mock_console_instance
        mock_table_class.return_value = mock_table_instance
        yield mock_console_instance, mock_table_instance


# --- Test Cases ---


def test_run_grounding_check_fully_grounded(
    mock_tool_check_grounding, mock_console_and_table
):
    """Verify the CLI output for a fully grounded answer."""
    mock_console, _mock_table = mock_console_and_table
    # Arrange: Configure the mock to return a "grounded" result
    mock_result = GroundingCheck(
        answer_candidate="The sky is blue.",
        is_grounded=True,
        confidence=0.95,
        grounding_ratio=1.0,
        claim_analyses=[
            ClaimAnalysis(
                claim="The sky is blue.",
                is_supported=True,
                confidence=0.95,
                supporting_fact="The sky appears blue due to Rayleigh scattering.",
            )
        ],
        unsupported_claims=[],
    )
    mock_tool_check_grounding.return_value = mock_result

    args = argparse.Namespace(
        answer="The sky is blue.",
        facts=["The sky appears blue due to Rayleigh scattering."],
        use_llm=False,
    )

    # Act
    run_grounding_check(args)

    # Assert
    # Check that the core tool was called correctly
    mock_tool_check_grounding.assert_called_once()
    call_args = mock_tool_check_grounding.call_args[0][0]
    assert call_args.answer_candidate == args.answer
    assert call_args.facts == args.facts
    assert not call_args.use_llm

    # Check the console output
    mock_console.print.assert_any_call(
        "Overall Status: [bold green]GROUNDED[/bold green]"
    )
    mock_console.print.assert_any_call("Confidence: 0.95")
    # Check that a table was created for the claim analysis
    mock_console.print.assert_any_call(ANY)  # This would be the table object


def test_run_grounding_check_not_grounded(
    mock_tool_check_grounding, mock_console_and_table
):
    """Verify the CLI output for a non-grounded answer."""
    mock_console, mock_table = mock_console_and_table
    # Arrange
    mock_result = GroundingCheck(
        answer_candidate="The moon is made of cheese.",
        is_grounded=False,
        confidence=0.88,
        grounding_ratio=0.0,
        unsupported_claims=["The moon is made of cheese."],
        claim_analyses=[
            ClaimAnalysis(
                claim="The moon is made of cheese.",
                is_supported=False,
                confidence=0.1,
                supporting_fact=None,
            )
        ],
    )
    mock_tool_check_grounding.return_value = mock_result

    args = argparse.Namespace(
        answer="The moon is made of cheese.",
        facts=["The moon is a rocky satellite."],
        use_llm=True,
    )

    # Act
    run_grounding_check(args)

    # Assert
    mock_tool_check_grounding.assert_called_once()
    call_args = mock_tool_check_grounding.call_args[0][0]
    assert call_args.use_llm

    mock_console.print.assert_any_call(
        "Overall Status: [bold red]NOT GROUNDED[/bold red]"
    )
    mock_console.print.assert_any_call("Confidence: 0.88")

    # Check that the table's row contains the unsupported icon '❌'
    mock_table.add_row.assert_called_with(
        "The moon is made of cheese.",
        "❌",
        "0.10",
        "N/A",
    )
    mock_console.print.assert_any_call(mock_table)


def test_run_grounding_check_no_claims(
    mock_tool_check_grounding, mock_console_and_table
):
    """Verify output when no verifiable claims are found."""
    mock_console, _ = mock_console_and_table
    # Arrange
    mock_result = GroundingCheck(
        answer_candidate="Hmm, I'm not sure.",
        is_grounded=False,  # Based on the corrected logic
        confidence=1.0,
        grounding_ratio=0.0,
        claim_analyses=[],
        unsupported_claims=[],
    )
    mock_tool_check_grounding.return_value = mock_result

    args = argparse.Namespace(
        answer="Hmm, I'm not sure.", facts=["Some fact."], use_llm=False
    )

    # Act
    run_grounding_check(args)

    # Assert
    mock_console.print.assert_any_call(
        "Overall Status: [bold red]NOT GROUNDED[/bold red]"
    )
    mock_console.print.assert_any_call(
        "\n[yellow]No verifiable claims were extracted from the answer.[/yellow]"
    )


def test_run_grounding_check_handles_exception(
    mock_tool_check_grounding, mock_console_and_table
):
    """Verify graceful error handling on unexpected failure."""
    mock_console, _ = mock_console_and_table
    # Arrange
    error_message = "Embedding model not found"
    mock_tool_check_grounding.side_effect = Exception(error_message)

    args = argparse.Namespace(
        answer="This will fail.", facts=["Any fact."], use_llm=False
    )

    # Act & Assert
    with pytest.raises(SystemExit) as e:
        run_grounding_check(args)

    assert e.value.code == 1
    mock_console.print.assert_called_with(
        f"[bold red]An unexpected error occurred[/bold red]: {error_message}"
    )
