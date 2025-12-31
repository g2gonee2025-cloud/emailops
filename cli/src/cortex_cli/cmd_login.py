"""
Cortex Login Command

Authenticates with the Cortex API and stores the JWT locally.
"""

import argparse
import getpass
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
from cortex_cli.api_client import get_default_token_path
from cortex_cli.style import colorize


def setup_login_parser(
    subparsers: Any,
) -> None:
    """Setup parser for the `login` command."""
    login_parser = subparsers.add_parser(
        "login",
        help="Authenticate with the Cortex API",
        description="""
        Authenticates with the Cortex API and stores the JWT locally for subsequent commands.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    login_parser.add_argument(
        "--username",
        "-u",
        required=True,
        help="Username for authentication.",
    )
    login_parser.add_argument(
        "--password-stdin",
        action="store_true",
        help="Read the password from stdin instead of prompting.",
    )
    login_parser.add_argument(
        "--host",
        default="https://localhost:8000",
        help="The host of the Cortex API.",
    )
    login_parser.add_argument(
        "--allow-http",
        action="store_true",
        help="Allow HTTP for non-local hosts (not recommended).",
    )
    login_parser.add_argument(
        "--token-path",
        default=None,
        help="Path to store the access token (default: ~/.config/cortex_cli/token).",
    )
    login_parser.add_argument(
        "--show-token",
        action="store_true",
        help="Print the access token to stdout.",
    )
    login_parser.set_defaults(func=_run_login)


def _run_login(args: argparse.Namespace) -> None:
    """Execute the login command."""
    username = (args.username or "").strip()
    if not username:
        print(colorize("Username is required.", "red"))
        sys.exit(1)

    if args.password_stdin:
        password = sys.stdin.read().strip()
    else:
        password = getpass.getpass("Password: ")

    if not password:
        print(colorize("Password is required.", "red"))
        sys.exit(1)

    host = (args.host or "").strip()
    if "://" not in host:
        host = f"https://{host}"

    try:
        base_url = httpx.URL(host)
    except Exception:
        print(colorize("Invalid host URL.", "red"))
        sys.exit(1)

    if base_url.scheme != "https":
        is_local = base_url.host in {"localhost", "127.0.0.1", "::1"}
        if not is_local and not args.allow_http:
            print(colorize("Refusing to send credentials over insecure HTTP.", "red"))
            sys.exit(1)

    token_path = (
        Path(args.token_path).expanduser()
        if args.token_path
        else get_default_token_path()
    )
    token_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with httpx.Client() as client:
            response = client.post(
                urljoin(str(base_url).rstrip("/") + "/", "auth/login"),
                json={"username": username, "password": password},
            )
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError as exc:
                raise RuntimeError("Login response was not valid JSON.") from exc

            if not isinstance(data, dict):
                raise RuntimeError("Login response had unexpected structure.")

            token = data.get("access_token")
            if not token:
                raise RuntimeError("Login response did not include an access token.")

            token_path.write_text(f"{token}\n", encoding="utf-8")
            try:
                os.chmod(token_path, 0o600)
            except OSError:
                pass

            print(colorize("Login successful!", "green"))
            print(
                f"Access token saved to {token_path} "
                "(export CORTEX_API_TOKEN to override)."
            )
            if args.show_token:
                print(f"Access Token: {token}")
    except httpx.HTTPStatusError as e:
        detail = None
        try:
            payload = e.response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            detail = payload.get("detail")
        message = f"Login failed: {e.response.status_code} {e.response.reason_phrase}"
        if detail:
            message = f"{message} ({detail})"
        print(colorize(message, "red"))
        sys.exit(1)
    except httpx.RequestError as e:
        print(colorize(f"An error occurred while requesting {e.request.url!r}.", "red"))
        sys.exit(1)
    except Exception as e:
        print(colorize(f"Login failed: {e}", "red"))
        sys.exit(1)
