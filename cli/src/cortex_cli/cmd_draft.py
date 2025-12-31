"""
from __future__ import annotations

Draft command for Cortex CLI.
"""

import argparse
import os
import uuid

import httpx
from rich.console import Console
from rich.syntax import Syntax


def setup_draft_parser(
    parser: argparse._SubParsersAction,
) -> None:
    """
    Setup the parser for the 'draft' command.
    """
    draft_parser = parser.add_parser("draft", help="Draft an email")
    draft_parser.add_argument("instruction", help="Drafting instruction")
    draft_parser.add_argument("--thread-id", help="Thread context")
    draft_parser.add_argument("--reply-to-message-id", help="Message to reply to")
    draft_parser.add_argument("--tone", default="professional", help="Email tone")
    draft_parser.set_defaults(func=run_draft_command)


def run_draft_command(args: argparse.Namespace) -> None:
    """
    Run the 'draft' command.
    """
    console = Console()

    api_url = os.getenv("CORTEX_API_URL", "http://localhost:8000")
    if not api_url:
        console.print("[red]CORTEX_API_URL environment variable not set.[/red]")
        return

    try:
        data = {
            "instruction": args.instruction,
            "thread_id": args.thread_id,
            "reply_to_message_id": args.reply_to_message_id,
            "tone": args.tone,
        }

        with httpx.Client(base_url=api_url, timeout=60.0) as client:
            request_id = str(uuid.uuid4())
            headers = {"X-Request-ID": request_id}
            console.print(f"Sending request (ID: {request_id})...")

            res = client.post("/api/v1/draft-email", json=data, headers=headers)
            res.raise_for_status()

            response_data = res.json()
            draft_content = response_data.get("draft", {}).get(
                "draft", "No draft generated."
            )

            console.print("\n[bold]Drafted Email:[/bold]")
            syntax = Syntax(draft_content, "markdown", theme="default", word_wrap=True)
            console.print(syntax)

            console.print(
                f"\n[dim]Correlation ID: {response_data.get('correlation_id')}[/dim]"
            )
            console.print(
                f"[dim]Iterations: {response_data.get('iterations', 0)}[/dim]"
            )

    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error: {e.response.status_code} - {e.response.text}[/red]")
    except httpx.RequestError as e:
        console.print(f"[red]Request failed: {e}[/red]")
    except Exception as e:
        console.print(f"[red]An unexpected error occurred: {e}[/red]")
