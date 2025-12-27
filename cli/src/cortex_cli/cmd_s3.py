"""
S3/Spaces subcommands for Cortex CLI.

Provides:
- `cortex s3 list` - List S3 prefixes/folders
- `cortex s3 ingest` - Ingest from S3
"""

import argparse
import sys
from typing import Any

from cortex_cli.style import colorize as _colorize

try:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


def cmd_s3_list(args: argparse.Namespace) -> None:
    """List S3/Spaces prefixes (conversation folders)."""
    try:
        from cortex.config.loader import get_config
        from cortex.ingestion.s3_source import S3SourceHandler

        config = get_config()
        handler = S3SourceHandler()

        print(f"\n{_colorize('S3/SPACES BROWSER', 'bold')}\n")
        print(f"  Endpoint: {_colorize(config.storage.endpoint_url, 'cyan')}")
        print(f"  Bucket:   {_colorize(config.storage.bucket_raw, 'cyan')}")
        if args.prefix:
            print(f"  Prefix:   {_colorize(args.prefix, 'dim')}")
        print()

        folders = list(handler.list_conversation_folders(prefix=args.prefix or ""))

        if not folders:
            print(f"  {_colorize('○', 'yellow')} No conversation folders found.")
            return

        if args.json:
            import json

            # Serialize folder objects to dicts
            folder_data = [{"name": f.name, "prefix": f.prefix} for f in folders]
            print(json.dumps({"folders": folder_data}, indent=2))
        elif RICH_AVAILABLE and console:
            table = Table(title=f"Folders ({len(folders)})", box=box.SIMPLE)
            table.add_column("#", style="dim")
            table.add_column("Prefix", style="cyan")
            for i, folder in enumerate(folders[: args.limit or 50], 1):
                table.add_row(str(i), folder.name)
            if len(folders) > (args.limit or 50):
                table.add_row("...", f"({len(folders) - (args.limit or 50)} more)")
            console.print(table)
        else:
            for i, folder in enumerate(folders[: args.limit or 50], 1):
                print(f"  {i}. {folder.name}")
            if len(folders) > (args.limit or 50):
                print(f"  ... and {len(folders) - (args.limit or 50)} more")

    except Exception as e:
        print(f"{_colorize('ERROR:', 'red')} {e}")
        sys.exit(1)


def cmd_s3_ingest(args: argparse.Namespace) -> None:
    """Ingest conversations from S3/Spaces."""
    import json
    import requests

    try:
        from cortex.config.loader import get_config

        config = get_config()
        api_base = config.core.api_base_url.rstrip("/")
        url = f"{api_base}/ingest/s3/start"

        print(f"\n{_colorize('S3/SPACES INGESTION', 'bold')}\n")
        print(f"  API Target: {_colorize(url, 'cyan')}")
        print(f"  Prefix:     {_colorize(args.prefix, 'cyan')}")
        print(f"  Tenant:     {_colorize(args.tenant, 'dim')}")

        headers = {"X-Tenant-ID": args.tenant, "Content-Type": "application/json"}
        payload = {"prefix": args.prefix, "dry_run": args.dry_run}

        print(f"\n  {_colorize('⏳', 'yellow')} Sending request to start ingestion job...\n")

        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "dry_run":
                print(f"  {_colorize('DRY RUN', 'yellow')}: {result.get('message')}")
                folders_to_process = result.get("folders_to_process", [])
                print(f"  Folders to process: {len(folders_to_process)}")
                for folder in folders_to_process[:10]:
                    print(f"    - {folder}")
                if len(folders_to_process) > 10:
                    print(f"    ... and {len(folders_to_process) - 10} more.")
            else:
                print(f"  {_colorize('✔', 'green')} Job started successfully!")
                print(f"  Job ID: {_colorize(result.get('job_id'), 'cyan')}")
                job_id = result.get("job_id")
                if job_id:
                    print(
                        "  To check status, run: "
                        f"{_colorize(f'cortex s3 status {job_id}', 'bold')}"
                    )
        else:
            print(
                f"  {_colorize('ERROR:', 'red')} "
                f"Failed to start ingestion job (HTTP {response.status_code})"
            )
            try:
                error_detail = response.json()
                print(json.dumps(error_detail, indent=2))
            except json.JSONDecodeError:
                print(response.text)
            sys.exit(1)

    except requests.exceptions.RequestException as e:
        print(f"{_colorize('ERROR:', 'red')} API request failed: {e}")
        print("  Please ensure the Cortex API server is running and accessible.")
        sys.exit(1)
    except Exception as e:
        print(f"{_colorize('ERROR:', 'red')} {e}")
        sys.exit(1)


def cmd_s3_validate(args: argparse.Namespace) -> None:
    """Validate and enrich S3 manifests with scan_and_refresh.

    Downloads conversation folders from S3, runs B1 validation to add:
    - Real SHA256 content hash
    - participants list
    - last_from / last_to fields

    Optionally uploads enriched manifests back to S3.
    """
    import json
    import shutil
    import tempfile
    from pathlib import Path

    try:
        from cortex.ingestion.conv_manifest.validation import scan_and_refresh
        from cortex.ingestion.s3_source import S3SourceHandler

        handler = S3SourceHandler()

        if not args.json:
            print(f"\n{_colorize('S3/SPACES MANIFEST VALIDATION', 'bold')}\n")
            print(f"  Endpoint: {_colorize(handler.endpoint_url, 'cyan')}")
            print(f"  Bucket:   {_colorize(handler.bucket, 'cyan')}")
            print(f"  Prefix:   {_colorize(args.prefix, 'dim')}")
            print(f"  Limit:    {args.limit}")
            upload_str = "YES" if args.upload else "NO"
            upload_color = "yellow" if args.upload else "dim"
            print(f"  Upload:   {_colorize(upload_str, upload_color)}")
            print()

        # List folders
        folders = list(
            handler.list_conversation_folders(prefix=args.prefix, limit=args.limit)
        )

        if not folders:
            if args.json:
                print(json.dumps({"error": "No folders found", "folders_processed": 0}))
            else:
                print(f"  {_colorize('○', 'yellow')} No conversation folders found.")
            return

        if args.dry_run:
            if args.json:
                folder_data = [{"name": f.name, "prefix": f.prefix} for f in folders]
                print(json.dumps({"dry_run": True, "folders": folder_data}))
            else:
                print(
                    f"  {_colorize('DRY RUN:', 'yellow')} "
                    f"Would validate {len(folders)} folder(s):"
                )
                for f in folders[:10]:
                    print(f"    • {f.name}")
                if len(folders) > 10:
                    print(f"    ... and {len(folders) - 10} more")
            return

        # Process folders
        temp_root = Path(tempfile.mkdtemp(prefix="cortex_s3_validate_"))
        results: list[dict[str, Any]] = []

        try:
            if not args.json:
                print(
                    f"  {_colorize('⏳', 'yellow')} "
                    f"Validating {len(folders)} folder(s)...\n"
                )

            for i, folder in enumerate(folders, 1):
                if not args.json:
                    print(
                        f"  [{i}/{len(folders)}] {folder.name[:50]}...",
                        end=" ",
                        flush=True,
                    )

                try:
                    # Download folder
                    local_path = handler.download_conversation_folder(folder, temp_root)

                    # Get manifest before
                    manifest_path = local_path / "manifest.json"
                    before: dict[str, Any] = {}
                    if manifest_path.exists():
                        before = json.loads(manifest_path.read_text(encoding="utf-8"))

                    # Run validation on the parent (scan_and_refresh expects root)
                    scan_and_refresh(temp_root)

                    # Get manifest after
                    after: dict[str, Any] = {}
                    if manifest_path.exists():
                        after = json.loads(manifest_path.read_text(encoding="utf-8"))

                    # Check what changed
                    added_fields = []
                    if "participants" in after and "participants" not in before:
                        added_fields.append("participants")
                    if "last_from" in after and "last_from" not in before:
                        added_fields.append("last_from")
                    if "last_to" in after and "last_to" not in before:
                        added_fields.append("last_to")
                    sha_before = before.get("sha256_conversation", "")
                    sha_after = after.get("sha256_conversation", "")
                    if sha_before.startswith("00000000") and not sha_after.startswith(
                        "00000000"
                    ):
                        added_fields.append("sha256_fixed")

                    result: dict[str, Any] = {
                        "folder": folder.name,
                        "prefix": folder.prefix,
                        "added_fields": added_fields,
                        "uploaded": False,
                    }

                    # Upload if requested
                    if args.upload and manifest_path.exists():
                        manifest_key = f"{folder.prefix}manifest.json"
                        handler.upload_file(manifest_path, manifest_key)
                        result["uploaded"] = True

                    results.append(result)

                    if not args.json:
                        status = _colorize("OK", "green")
                        if added_fields:
                            status += f" (+{', '.join(added_fields)})"
                        if result["uploaded"]:
                            status += f" {_colorize('[uploaded]', 'cyan')}"
                        print(status)

                except Exception as e:
                    results.append(
                        {
                            "folder": folder.name,
                            "prefix": folder.prefix,
                            "error": str(e),
                        }
                    )
                    if not args.json:
                        print(f"{_colorize('ERROR', 'red')}: {e}")

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_root, ignore_errors=True)

        # Summary
        success = sum(1 for r in results if "error" not in r)
        failed = len(results) - success
        uploaded = sum(1 for r in results if r.get("uploaded"))

        if args.json:
            print(
                json.dumps(
                    {
                        "folders_processed": len(results),
                        "success": success,
                        "failed": failed,
                        "uploaded": uploaded,
                        "results": results,
                    },
                    indent=2,
                )
            )
        else:
            print(f"\n{_colorize('═' * 40, 'cyan')}")
            print(f"  Processed: {success}")
            failed_color = "red" if failed > 0 else "dim"
            print(f"  Failed:    {_colorize(str(failed), failed_color)}")
            if args.upload:
                print(f"  Uploaded:  {_colorize(str(uploaded), 'cyan')}")

    except Exception as e:
        if args.json:
            import json

            print(json.dumps({"error": str(e)}))
        else:
            print(f"{_colorize('ERROR:', 'red')} {e}")
        sys.exit(1)


def cmd_s3_status(args: argparse.Namespace) -> None:
    """Check the status of an S3 ingestion job."""
    import json
    import requests

    try:
        from cortex.config.loader import get_config

        config = get_config()
        api_base = config.core.api_base_url.rstrip("/")
        url = f"{api_base}/ingest/status/{args.job_id}"

        print(f"\n{_colorize('S3 INGESTION STATUS', 'bold')}\n")
        print(f"  API Target: {_colorize(url, 'cyan')}")
        print(f"  Job ID:     {_colorize(args.job_id, 'cyan')}")

        headers = {"X-Tenant-ID": args.tenant}

        print(f"\n  {_colorize('⏳', 'yellow')} Fetching job status...\n")

        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code == 200:
            status = response.json()
            if args.json:
                print(json.dumps(status, indent=2))
                return

            print(f"  Status:     {_colorize(status.get('status', 'N/A').upper(), 'bold')}")
            print(f"  Message:    {status.get('message', 'N/A')}")
            print("-" * 30)
            print(f"  Folders Processed: {status.get('folders_processed', 0)}")
            print(f"  Chunks Created:    {status.get('chunks_created', 0)}")
            print(f"  Embeddings:        {status.get('embeddings_generated', 0)}")
            print(f"  Errors:            {_colorize(str(status.get('errors', 0)), 'red' if status.get('errors') else 'dim')}")

        elif response.status_code == 404:
            print(f"  {_colorize('ERROR: Job not found', 'red')}")
            sys.exit(1)
        else:
            print(
                f"  {_colorize('ERROR:', 'red')} "
                f"Failed to get job status (HTTP {response.status_code})"
            )
            try:
                error_detail = response.json()
                print(json.dumps(error_detail, indent=2))
            except json.JSONDecodeError:
                print(response.text)
            sys.exit(1)

    except requests.exceptions.RequestException as e:
        print(f"{_colorize('ERROR:', 'red')} API request failed: {e}")
        print("  Please ensure the Cortex API server is running and accessible.")
        sys.exit(1)
    except Exception as e:
        print(f"{_colorize('ERROR:', 'red')} {e}")
        sys.exit(1)


def setup_s3_parser(subparsers: Any) -> None:
    """Add s3 subcommands to the CLI parser."""
    s3_parser = subparsers.add_parser(
        "s3",
        help="S3/Spaces storage commands",
        description="Browse and ingest from S3/DigitalOcean Spaces.",
    )

    s3_subparsers = s3_parser.add_subparsers(dest="s3_command", title="S3 Commands")

    # s3 list
    list_parser = s3_subparsers.add_parser(
        "list",
        help="List S3 prefixes/folders",
        description="Browse conversation folders in S3/Spaces.",
    )
    list_parser.add_argument("--prefix", "-p", default="", help="S3 prefix to filter")
    list_parser.add_argument(
        "--limit", "-l", type=int, default=50, help="Max folders to show"
    )
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")
    list_parser.set_defaults(func=cmd_s3_list)

    # s3 ingest
    ingest_parser = s3_subparsers.add_parser(
        "ingest",
        help="Ingest from S3/Spaces",
        description="Ingest conversation folders from S3/Spaces.",
    )
    ingest_parser.add_argument("prefix", metavar="PREFIX", help="S3 prefix to ingest")
    ingest_parser.add_argument("--tenant", "-t", default="default", help="Tenant ID")
    ingest_parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Show what would be done"
    )
    ingest_parser.set_defaults(func=cmd_s3_ingest)

    # s3 status
    status_parser = s3_subparsers.add_parser(
        "status",
        help="Check status of an S3 ingestion job",
        description="Check the status of a background S3 ingestion job.",
    )
    status_parser.add_argument("job_id", metavar="JOB_ID", help="Ingestion job ID to check")
    status_parser.add_argument("--tenant", "-t", default="default", help="Tenant ID")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")
    status_parser.set_defaults(func=cmd_s3_status)


    # s3 validate
    validate_parser = s3_subparsers.add_parser(
        "validate",
        help="Validate and enrich S3 manifests",
        description="Download folders, run B1 validation, optionally upload enriched manifests.",
    )
    validate_parser.add_argument(
        "--prefix",
        "-p",
        default="Outlook/",
        help="S3 prefix to filter (default: Outlook/)",
    )
    validate_parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=10,
        help="Max folders to process (default: 10)",
    )
    validate_parser.add_argument(
        "--upload",
        "-u",
        action="store_true",
        help="Upload enriched manifests back to S3",
    )
    validate_parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Show what would be done"
    )
    validate_parser.add_argument("--json", action="store_true", help="Output as JSON")
    validate_parser.set_defaults(func=cmd_s3_validate)

    # s3 check-structure
    check_structure_parser = s3_subparsers.add_parser(
        "check-structure",
        help="Check S3 folder structure for correctness",
        description="Scans S3 folders and reports missing files or structural issues.",
    )
    check_structure_parser.add_argument(
        "--prefix",
        "-p",
        default="raw/outlook/",
        help="S3 prefix to check (default: raw/outlook/)",
    )
    check_structure_parser.add_argument(
        "--sample",
        "-s",
        type=int,
        default=20,
        help="Number of folders to sample (default: 20)",
    )
    check_structure_parser.set_defaults(func=cmd_s3_check_structure)

    # Default: show list when no subcommand given
    def _default_s3_handler(args: argparse.Namespace) -> None:
        if not args.s3_command:
            args.prefix = getattr(args, "prefix", "")
            args.limit = getattr(args, "limit", 20)
            args.json = getattr(args, "json", False)
            cmd_s3_list(args)

    s3_parser.set_defaults(func=_default_s3_handler)


def cmd_s3_check_structure(args: argparse.Namespace) -> None:
    """Check S3 folder structure for correctness."""
    import json
    from cortex_cli.s3_check import check_s3_structure

    try:
        results = check_s3_structure(
            prefix=args.prefix,
            sample_size=args.sample,
        )
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
