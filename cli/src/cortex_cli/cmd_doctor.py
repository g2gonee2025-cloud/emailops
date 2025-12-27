#!/usr/bin/env python3
"""
Cortex Doctor - System Diagnostics Tool.

This command connects to the Cortex backend's API to run a series of
health checks on the system's components.

Exit Codes:
  0 - All checks passed (healthy)
  1 - Warnings detected (degraded)
  2 - Failures detected (unhealthy)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List

import httpx

from cortex.config.loader import get_config

# Library-safe logger
logger = logging.getLogger("cortex.doctor")

# ANSI color codes
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "cyan": "\033[36m",
}


def _c(text: str, color: str) -> str:
    """Apply ANSI color to text."""
    if not sys.stdout.isatty():
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def _get_api_url(config: Any) -> str:
    """Constructs the API URL from configuration."""
    api_host = "127.0.0.1"
    api_port = "8000"
    if hasattr(config, "api") and config.api.host:
        api_host = config.api.host
    if hasattr(config, "api") and config.api.port:
        api_port = config.api.port
    return f"http://{api_host}:{api_port}"


def _print_report_human(report: Dict[str, Any]) -> None:
    """Prints the doctor report in a human-readable format."""
    status_colors = {
        "healthy": "green",
        "degraded": "yellow",
        "unhealthy": "red",
    }
    status = report.get("overall_status", "unknown")
    color = status_colors.get(status, "reset")

    print()
    print(f"{_c('Cortex System Health Report', 'bold')}")
    print(f"{_c('═' * 40, 'cyan')}")
    print(f"Overall Status: {_c(status.upper(), color)}")
    print()

    for check in report.get("checks", []):
        name = check.get("name")
        status = check.get("status")
        message = check.get("message")

        if status == "pass":
            symbol = _c("✓", "green")
            color = "green"
        elif status == "warn":
            symbol = _c("⚠", "yellow")
            color = "yellow"
        else:
            symbol = _c("✗", "red")
            color = "red"

        print(f"{symbol} {_c(name, 'bold')}: {_c(message, color)}")
        details = check.get("details")
        if details:
            for key, value in details.items():
                print(f"    {_c(key, 'dim')}: {value}")


def _get_exit_code(status: str) -> int:
    """Maps the overall status to a process exit code."""
    if status == "unhealthy":
        return 2
    if status == "degraded":
        return 1
    return 0


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cortex Doctor - System Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON only"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose logging (DEBUG)"
    )
    args = parser.parse_args()
    _configure_logging(args.verbose)

    config = get_config()
    api_url = _get_api_url(config)
    doctor_endpoint = f"{api_url}/admin/doctor"

    if not args.json:
        print(f"Contacting Cortex backend at {_c(api_url, 'cyan')}...")

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(doctor_endpoint)
            response.raise_for_status()
            report = response.json()

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_report_human(report)

        exit_code = _get_exit_code(report.get("overall_status", "unknown"))
        sys.exit(exit_code)

    except httpx.RequestError as e:
        logger.error(f"API request failed: {e}")
        if not args.json:
            print(
                f"\n{_c('Error:', 'red')} Could not connect to the Cortex backend at {_c(doctor_endpoint, 'bold')}."
            )
            print("Please ensure the backend service is running.")
        sys.exit(2)
    except httpx.HTTPStatusError as e:
        logger.error(f"API returned an error: {e.response.status_code} {e.response.text}")
        if not args.json:
            print(f"\n{_c('Error:', 'red')} The API returned a status code of {e.response.status_code}.")
            print("Response:", e.response.text)
        sys.exit(2)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        if not args.json:
            print(f"\n{_c('An unexpected error occurred.', 'red')}")
        sys.exit(2)


if __name__ == "__main__":
    main()
