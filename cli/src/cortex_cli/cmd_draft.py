"""Draft command for Cortex CLI."""

from __future__ import annotations

import argparse
import os
import uuid
from typing import Any

import httpx
from rich.console import Console
from rich.markup import escape
from rich.syntax import Syntax


def setup_draft_parser(parser: Any) -> None:
    """
    Setup the parser for the 'draft' command.
    """
    draft_parser = parser.add_parser("draft", help="Draft an email")
    draft_parser.add_argument("instruction", help="Drafting instruction")
    draft_parser.add_argument("--thread-id", help="Thread context")
    draft_parser.add_argument("--reply-to-message-id", help="Message to reply to")
    draft_parser.add_argument("--tone", default="professional", help="Email tone")
    draft_parser.set_defaults(func=_run_draft_cli)


def _run_draft_cli(args: argparse.Namespace) -> None:
    run_draft_command(args, exit_on_error=True)


def run_draft_command(args: argparse.Namespace, exit_on_error: bool = False) -> bool:
    """
    Run the 'draft' command.
    """
    console = Console()

    api_url = os.getenv("CORTEX_API_URL") or "http://localhost:8000"
    instruction = getattr(args, "instruction", None)
    if not instruction:
        console.print("[red]Draft instruction is required.[/red]")
        if exit_on_error:
            raise SystemExit(1)
        return False

    try:
        data = {
            "instruction": instruction,
            "thread_id": getattr(args, "thread_id", None),
            "reply_to_message_id": getattr(args, "reply_to_message_id", None),
            "tone": getattr(args, "tone", "professional"),
        }

        with httpx.Client(base_url=api_url, timeout=60.0) as client:
            request_id = str(uuid.uuid4())
            headers = {"X-Request-ID": request_id}
            console.print(f"Sending request (ID: {request_id})...")

            res = client.post("/api/v1/draft-email", json=data, headers=headers)
            res.raise_for_status()

            try:
                response_data = res.json()
            except ValueError:
                console.print("[red]Invalid JSON response from server.[/red]")
                if exit_on_error:
                    raise SystemExit(1)
                return False
            if not isinstance(response_data, dict):
                console.print("[red]Unexpected response format from server.[/red]")
                if exit_on_error:
                    raise SystemExit(1)
                return False

            draft_content = "No draft generated."
            draft_payload = response_data.get("draft")
            if isinstance(draft_payload, dict):
                draft_content = (
                    draft_payload.get("body_markdown")
                    or draft_payload.get("draft")
                    or draft_content
                )
            elif isinstance(draft_payload, str):
                draft_content = draft_payload

            console.print("\n[bold]Drafted Email:[/bold]")
            draft_syntax = Syntax(
                draft_content, "markdown", theme="default", word_wrap=True
            )
            console.print(draft_syntax)

            console.print(
                f"\n[dim]Correlation ID: {response_data.get('correlation_id')}[/dim]"
            )
            console.print(
                f"[dim]Iterations: {response_data.get('iterations', 0)}[/dim]"
            )

    except httpx.HTTPStatusError as e:
        message = escape(e.response.text) if e.response.text else "Request failed."
        console.print(f"[red]Error: {e.response.status_code} - {message}[/red]")
        if exit_on_error:
            raise SystemExit(1)
        return False
    except httpx.RequestError as e:
        console.print(f"[red]Request failed: {e}[/red]")
        if exit_on_error:
            raise SystemExit(1)
        return False
    except Exception as e:
        console.print(f"[red]An unexpected error occurred: {e}[/red]")
        if exit_on_error:
            raise SystemExit(1)
        return False

    return True
