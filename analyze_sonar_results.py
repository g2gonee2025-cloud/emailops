import argparse
import asyncio
import ipaddress
import os
import re
import socket
import sys
from typing import Any, List, Union
from urllib.parse import urlparse

import httpx


def _redact_pii(data: str | dict | list) -> Any:
    """Recursively scans and redacts PII from strings."""
    if isinstance(data, dict):
        return {k: _redact_pii(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_redact_pii(item) for item in data]
    if isinstance(data, str):
        data = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED_EMAIL]", data)
        data = re.sub(r"https?://\S+", "[REDACTED_URL]", data)
        data = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "[REDACTED_IP]", data)
    return data


async def validate_url_is_public(url: str) -> None:
    """Validates that the URL does not resolve to a private or loopback IP to prevent SSRF."""
    try:
        hostname = urlparse(url).hostname
        if not hostname:
            raise ValueError("Hostname could not be parsed from URL.")

        loop = asyncio.get_running_loop()
        ip_addr = await loop.run_in_executor(None, socket.gethostbyname, hostname)
        ip = ipaddress.ip_address(ip_addr)

        if ip.is_private or ip.is_loopback or ip.is_unspecified:
            raise ValueError(f"URL resolves to a non-public IP address: {ip_addr}")

    except socket.gaierror:
        print(f"Warning: Could not resolve hostname for {url}. Proceeding...", file=sys.stderr)
    except ValueError as e:
        print(f"Error: Invalid SonarQube host URL. {e}", file=sys.stderr)
        sys.exit(1)


class SonarAnalyzer:
    def __init__(self, project_key: str, sonar_host_url: str, sonar_token: str | None):
        self.project_key = project_key
        self.sonar_host_url = sonar_host_url.rstrip("/")
        self.auth = (sonar_token or "", "")
        self.timeout = 10.0
        self.page_size = 100

    async def _get_paginated(
        self, client: httpx.AsyncClient, endpoint: str, params: dict, results_key: str
    ) -> list[Any]:
        all_results = []
        page = 1
        total = 1

        base_params = params.copy()
        base_params["ps"] = self.page_size

        while page * self.page_size <= total + self.page_size:
            base_params["p"] = page
            url = f"{self.sonar_host_url}{endpoint}"
            try:
                r = await client.get(url, auth=self.auth, params=base_params, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()

                results = data.get(results_key, [])
                all_results.extend(results)

                if "paging" in data:
                    total = data["paging"]["total"]
                elif page == 1:
                    total = data.get("total", len(results))

                if not results or len(results) < self.page_size:
                    break
                page += 1

            except httpx.HTTPStatusError as e:
                print(f"HTTP error occurred: {e}", file=sys.stderr)
                raise
            except httpx.RequestError as e:
                print(f"An error occurred while requesting {e.request.url!r}.", file=sys.stderr)
                raise
        return _redact_pii(all_results)

    async def get_hotspots(self, client: httpx.AsyncClient) -> list[Any]:
        params = {"projectKey": self.project_key, "status": "TO_REVIEW"}
        return await self._get_paginated(client, "/api/hotspots/search", params, "hotspots")

    async def get_coverage(self, client: httpx.AsyncClient) -> list[Any]:
        params = {
            "component": self.project_key,
            "metricKeys": "new_coverage,new_lines_to_cover,new_uncovered_lines",
            "qualifiers": "FIL",
        }
        return await self._get_paginated(client, "/api/measures/component_tree", params, "components")

    async def get_issues(self, client: httpx.AsyncClient) -> list[Any]:
        params = {"componentKeys": self.project_key, "resolved": "false"}
        return await self._get_paginated(client, "/api/issues/search", params, "issues")

    async def print_analysis(self) -> None:
        async with httpx.AsyncClient() as client:
            issues_task = self.get_issues(client)
            hotspots_task = self.get_hotspots(client)
            coverage_task = self.get_coverage(client)

            issues, hotspots, coverage_components = await asyncio.gather(
                issues_task, hotspots_task, coverage_task
            )

            print("=== VIOLATIONS (New Code) ===")
            for issue in issues:
                print(
                    f"[{issue.get('type', 'N/A')}] {issue.get('component', 'N/A')}: {issue.get('message', 'N/A')} (Line {issue.get('line', '?')})"
                )

            print("\n=== SECURITY HOTSPOTS ===")
            for h in hotspots:
                print(
                    f"[{h.get('securityCategory', 'N/A')}] {h.get('component', 'N/A')}: {h.get('message', 'N/A')} (Line {h.get('line', '?')})"
                )
                print(f"  - Rule: {h.get('ruleKey', 'N/A')}")
                print(f"  - Key: {h.get('key', 'N/A')}")

            print("\n=== LOW COVERAGE FILES (New Code) ===")
            for comp in coverage_components:
                measures = {
                    m["metric"]: m.get("period", {}).get("value", m.get("value"))
                    for m in comp.get("measures", [])
                }
                new_lines = float(measures.get("new_lines_to_cover", 0))
                if new_lines > 0:
                    cov = float(measures.get("new_coverage", 0))
                    if cov < 80.0:
                        print(f"{comp.get('path', 'N/A')}: {cov}% (Lines to cover: {int(new_lines)})")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SonarQube results.")
    parser.add_argument("--project-key", default="emailops-vertex-ai", help="SonarQube project key.")
    parser.add_argument("--sonar-host-url", default="http://localhost:9000", help="SonarQube host URL.")
    args = parser.parse_args()

    await validate_url_is_public(args.sonar_host_url)

    SONAR_TOKEN = os.environ.get("SONAR_TOKEN")

    analyzer = SonarAnalyzer(args.project_key, args.sonar_host_url, SONAR_TOKEN)
    await analyzer.print_analysis()


if __name__ == "__main__":
    asyncio.run(main())
