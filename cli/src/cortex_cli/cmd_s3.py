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
    import uuid

    try:
        from cortex.config.loader import get_config
        from cortex.ingestion.mailroom import process_job
        from cortex.ingestion.models import IngestJobRequest
        from cortex.ingestion.s3_source import S3SourceHandler

        config = get_config()
        handler = S3SourceHandler()

        print(f"\n{_colorize('S3/SPACES INGESTION', 'bold')}\n")
        print(f"  Prefix:  {_colorize(args.prefix, 'cyan')}")
        print(f"  Tenant:  {_colorize(args.tenant, 'dim')}")

        if args.dry_run:
            # List what would be ingested
            folders = list(handler.list_conversation_folders(prefix=args.prefix))
            print(
                f"\n  {_colorize('DRY RUN:', 'yellow')} Would ingest {len(folders)} folder(s):"
            )
            for f in folders[:10]:
                print(f"    • {f.name}")
            if len(folders) > 10:
                print(f"    ... and {len(folders) - 10} more")
            return

        # Ingest
        folders = list(handler.list_conversation_folders(prefix=args.prefix))
        if not folders:
            print(
                f"\n  {_colorize('⚠', 'yellow')} No folders found with prefix: {args.prefix}"
            )
            return

        print(f"\n  {_colorize('⏳', 'yellow')} Ingesting {len(folders)} folder(s)...\n")

        success = 0
        failed = 0

        for i, folder in enumerate(folders, 1):
            print(f"  [{i}/{len(folders)}] {folder.name}...", end=" ", flush=True)

            job = IngestJobRequest(
                job_id=uuid.uuid4(),
                tenant_id=args.tenant,
                source_type="s3",
                source_uri=f"s3://{config.storage.bucket_raw}/{folder.prefix}",
            )

            try:
                summary = process_job(job)
                if summary.aborted_reason:
                    print(f"{_colorize('FAILED', 'red')}")
                    failed += 1
                else:
                    print(
                        f"{_colorize('OK', 'green')} ({summary.messages_ingested} msg)"
                    )
                    success += 1
            except Exception as e:
                print(f"{_colorize('ERROR', 'red')}: {e}")
                failed += 1

        print(f"\n{_colorize('═' * 40, 'cyan')}")
        print(f"  Succeeded: {_colorize(str(success), 'green')}")
        print(f"  Failed:    {_colorize(str(failed), 'red' if failed > 0 else 'dim')}")

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
