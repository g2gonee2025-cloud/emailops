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
    login_parser.set_defaults(func=lambda args: _run_login(args, exit_on_error=True))


def _run_login(args: argparse.Namespace, exit_on_error: bool = False) -> bool:
    """Execute the login command."""
    username = (getattr(args, "username", "") or "").strip()
    if not username:
        print(colorize("Username is required.", "red"))
        if exit_on_error:
            sys.exit(1)
        return False

    password = getattr(args, "password", None)
    if password:
        password = str(password).strip()
    elif bool(getattr(args, "password_stdin", False)):
        password = sys.stdin.read().strip()
    else:
        password = getpass.getpass("Password: ")

    if not password:
        print(colorize("Password is required.", "red"))
        if exit_on_error:
            sys.exit(1)
        return False

    host = (getattr(args, "host", "") or "").strip()
    if "://" not in host:
        host = f"https://{host}"

    try:
        base_url = httpx.URL(host)
    except Exception:
        print(colorize("Invalid host URL.", "red"))
        if exit_on_error:
            sys.exit(1)
        return False

    if base_url.scheme != "https":
        is_local = base_url.host in {"localhost", "127.0.0.1", "::1"}
        if not is_local and not bool(getattr(args, "allow_http", False)):
            print(colorize("Refusing to send credentials over insecure HTTP.", "red"))
            if exit_on_error:
                sys.exit(1)
            return False

    token_path = (
        Path(getattr(args, "token_path", "")).expanduser()
        if getattr(args, "token_path", None)
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

            token_saved = True
            try:
                token_path.write_text(f"{token}\n", encoding="utf-8")
                try:
                    os.chmod(token_path, 0o600)
                except OSError:
                    pass
            except OSError as exc:
                token_saved = False
                print(
                    colorize(
                        f"Warning: failed to save token to {token_path}: {exc}",
                        "yellow",
                    )
                )

            print(colorize("Login successful!", "green"))
            if token_saved:
                print(
                    f"Access token saved to {token_path} "
                    "(export CORTEX_API_TOKEN to override)."
                )
            else:
                print("Access token not saved; export CORTEX_API_TOKEN to override.")
            show_token = bool(getattr(args, "show_token", True))
            if show_token:
                print(f"Access Token: {token}")
            return True
    except httpx.HTTPStatusError as e:
        response = e.response
        detail = None
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            detail = payload.get("detail")
        error_text = response.text or response.reason_phrase or "Request failed"
        message = f"Login failed: {response.status_code} {error_text}"
        if detail and detail not in error_text:
            message = f"{message} ({detail})"
        print(colorize(message, "red"))
        if exit_on_error:
            sys.exit(1)
        return False
    except httpx.RequestError as e:
        url = str(e.request.url) if e.request else "unknown"
        print(colorize(f"An error occurred while requesting '{url}'.", "red"))
        if exit_on_error:
            sys.exit(1)
        return False
    except Exception as e:
        print(colorize(f"Login failed: {e}", "red"))
        if exit_on_error:
            sys.exit(1)
        return False
