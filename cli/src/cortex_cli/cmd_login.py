"""
Cortex Login Command

Authenticates with the Cortex API and stores the JWT locally.
"""

import argparse
import httpx
from cortex_cli.style import colorize

def setup_login_parser(subparsers: argparse._SubParsersAction) -> None:
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
        "--password",
        "-p",
        required=True,
        help="Password for authentication.",
    )
    login_parser.add_argument(
        "--host",
        default="http://localhost:8000",
        help="The host of the Cortex API.",
    )
    login_parser.set_defaults(func=_run_login)

def _run_login(args: argparse.Namespace) -> None:
    """Execute the login command."""
    try:
        with httpx.Client() as client:
            response = client.post(
                f"{args.host}/auth/login",
                json={"username": args.username, "password": args.password},
            )
            response.raise_for_status()
            data = response.json()
            print(colorize("Login successful!", "green"))
            print(f"Access Token: {data['access_token']}")
    except httpx.HTTPStatusError as e:
        print(colorize(f"Login failed: {e.response.status_code} {e.response.text}", "red"))
    except httpx.RequestError as e:
        print(colorize(f"An error occurred while requesting {e.request.url!r}.", "red"))
