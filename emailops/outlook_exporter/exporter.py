"""
Outlook Exporter V3 - Optimized with proper COM object handling
Stores only metadata in indexes, re-fetches emails when needed
"""

import gc
import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import pythoncom
    import pywintypes
    import win32com.client
except ImportError as e:
    raise ImportError(
        "pywin32 is required but not installed. "
        "Install it with: pip install pywin32"
    ) from e

from .attachments import save_attachments_for_items
from .conversation import get_conversation_key
from .manifest_builder import build_conversation_text, generate_manifest
from .smtp_resolver import get_recipients_array, get_sender_smtp
from .state import ExportState
from .utils import (
    OL_CC,
    OL_FULLITEM,
    OL_MAIL,
    OL_TO,
    ensure_dir,
    outlook_restrict_datetime,
    sanitize_filename,
    short_hex,
)

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class EmailMetadata:
    """Lightweight metadata for an email without holding COM object"""
    def __init__(self, entry_id: str, conv_id: str | None, conv_key: str,
                 sort_date: datetime, folder_path: str, subject: str = "",
                 store_id: str | None = None):
        self.entry_id = entry_id
        self.conv_id = conv_id  # Can be None if email doesn't have ConversationID
        self.conv_key = conv_key
        self.sort_date = sort_date
        self.folder_path = folder_path
        self.subject = subject
        self.store_id = store_id  # Store ID for reliable re-fetching


class OutlookExporter:
    """Optimized Outlook exporter with proper COM object handling"""

    def __init__(
        self,
        output_root: Path,
        outlook_profile: str | None = None,
        enforce_offline: bool = True,
        progress_callback: ProgressCallback | None = None,
        user_email: str | None = None,
    ):
        self.output_root = output_root
        ensure_dir(self.output_root)
        self.enforce_offline = enforce_offline
        self.progress_callback = progress_callback

        # User email for "from me" detection
        self.user_email = user_email

        # Initialize COM properly
        pythoncom.CoInitialize()

        # Initialize Outlook
        self.app = win32com.client.Dispatch("Outlook.Application")
        if outlook_profile:
            self.app.Session.Logon(outlook_profile)
        self.ns = self.app.GetNamespace("MAPI")

        # Auto-detect user email if not provided
        if not self.user_email:
            try:
                current_user = self.ns.CurrentUser
                if current_user:
                    addr_entry = current_user.AddressEntry
                    if addr_entry:
                        self.user_email = getattr(addr_entry, "Address", None)
                        if not self.user_email:
                            # Try to get SMTP address
                            try:
                                smtp_addr = addr_entry.GetExchangeUser()
                                if smtp_addr:
                                    self.user_email = smtp_addr.PrimarySmtpAddress
                            except Exception:  # Outlook COM may raise opaque errors
                                pass
                        if self.user_email:
                            log.info("Auto-detected user email: %s", self.user_email)
            except Exception as e:
                log.warning("Failed to auto-detect user email: %s", e)

        # State management
        self.state = ExportState(self.output_root / "_state.json")

        # Track processed conversations (just IDs)
        self.processed_conversations: set[str] = set()

        # Index for efficient conversation traversal - stores only metadata
        # ConversationID -> List of EmailMetadata
        self.conversation_index: dict[str, list[EmailMetadata]] = {}

        # Fallback index for emails without ConversationID
        # conversation_key -> List of EmailMetadata
        self.fallback_index: dict[str, list[EmailMetadata]] = {}

        # Cache folder references to avoid repeated lookups
        self.folder_cache: dict[str, Any] = {}

        # Track Store IDs for reliable re-fetching
        self.store_ids: dict[str, str] = {}

    def export_folders(
        self,
        folder_paths: list[str],
        full_export: bool = False,
        since_utc: datetime | None = None,
    ) -> None:
        """Export emails using optimized conversation-based processing"""

        if not folder_paths:
            raise ValueError("No folder paths provided")

        # Load state for incremental updates
        if not full_export and since_utc is None:
            since_utc = self.state.last_sync_utc
        self.state.set_folders(folder_paths)

        # Step 1: Pool and index all emails (metadata only)
        log.info("Pooling and indexing email metadata from %d folders...", len(folder_paths))
        sorted_emails = self._pool_and_index_emails(folder_paths, since_utc)

        if not sorted_emails:
            log.info("No emails to export")
            return

        total_emails = len(sorted_emails)
        log.info("Found %d emails to process (indexed %d unique conversations)",
                total_emails, len(self.conversation_index) + len(self.fallback_index))

        # Step 2: Process emails by conversation in batches
        processed_count = 0
        skipped_count = 0
        exported_conversations = 0
        max_success_dt = None
        error_count = 0
        batch_size = 100

        log.info("Starting batch processing of %d emails (batch size: %d)", total_emails, batch_size)

        # Process in batches for better memory management
        for batch_start in range(0, total_emails, batch_size):
            batch_end = min(batch_start + batch_size, total_emails)
            batch_emails = sorted_emails[batch_start:batch_end]

            log.info("="*60)
            log.info("Processing batch %d-%d of %d total emails",
                    batch_start + 1, batch_end, total_emails)
            log.info("="*60)

            batch_exported = 0
            batch_skipped = 0
            batch_errors = 0

            for email_meta in batch_emails:
                idx = processed_count + 1

                try:
                    log.debug("Processing email %d/%d - EntryID: %s",
                             idx, total_emails, email_meta.entry_id[:20])

                    # Progress callback every 10 emails
                    if self.progress_callback and idx % 10 == 0:
                        self.progress_callback(
                            idx, total_emails,
                            f"Batch {batch_start//batch_size + 1}: {idx}/{total_emails} emails ({exported_conversations} conversations, {skipped_count} skipped, {error_count} errors)"
                        )

                    # Use ConversationID if available, otherwise fall back to conv_key
                    conversation_identifier = email_meta.conv_id if email_meta.conv_id else email_meta.conv_key

                    # Check if conversation already processed
                    if conversation_identifier in self.processed_conversations:
                        log.debug("Skipping email %s - conversation %s already processed",
                                 email_meta.entry_id[:20], conversation_identifier[:20])
                        skipped_count += 1
                        batch_skipped += 1
                        processed_count += 1
                        continue

                    # Get all emails in this conversation
                    log.info("Processing conversation %s from email %s (subject: %s)",
                            conversation_identifier[:20], email_meta.entry_id[:20],
                            email_meta.subject[:50] if email_meta.subject else "no subject")

                    conversation_emails = self._get_conversation_emails(email_meta)

                    if not conversation_emails:
                        log.warning("No emails found for conversation %s", conversation_identifier[:20])
                        error_count += 1
                        batch_errors += 1
                        processed_count += 1
                        continue

                    log.debug("Found %d emails in conversation %s",
                             len(conversation_emails), conversation_identifier[:20])

                    # For incremental updates, check if conversation needs updating
                    if not full_export:
                        needs_update, cutoff_date = self._check_incremental_update(
                            conversation_identifier, conversation_emails
                        )
                        if not needs_update:
                            log.info("Skipping conversation %s - no updates needed since %s",
                                    conversation_identifier[:20], cutoff_date)
                            self.processed_conversations.add(conversation_identifier)
                            skipped_count += 1
                            batch_skipped += 1
                            processed_count += 1
                            continue

                    # Export the conversation
                    log.info("Exporting conversation %s with %d emails...",
                            conversation_identifier[:20], len(conversation_emails))

                    export_dt = self._export_conversation(conversation_identifier, conversation_emails)

                    if export_dt:
                        # Mark conversation as processed
                        self.processed_conversations.add(conversation_identifier)
                        exported_conversations += 1
                        batch_exported += 1

                        # Track max date for state update
                        if max_success_dt is None or export_dt > max_success_dt:
                            max_success_dt = export_dt

                        log.info("SUCCESS: Exported conversation %s with %d emails (newest: %s)",
                                conversation_identifier[:20], len(conversation_emails),
                                export_dt.strftime("%Y-%m-%d %H:%M:%S") if export_dt else "unknown")
                    else:
                        log.error("FAILED: Could not export conversation %s", conversation_identifier[:20])
                        error_count += 1
                        batch_errors += 1

                    processed_count += 1

                except Exception as e:
                    log.exception("ERROR: Failed to process email %s: %s",
                                 email_meta.entry_id[:20] if email_meta else "unknown", e)
                    error_count += 1
                    batch_errors += 1
                    processed_count += 1
                    continue

            # Batch summary
            log.info("-"*60)
            log.info("Batch %d-%d Summary:", batch_start + 1, batch_end)
            log.info("  - Exported: %d conversations", batch_exported)
            log.info("  - Skipped: %d emails", batch_skipped)
            log.info("  - Errors: %d", batch_errors)
            log.info("  - Total Progress: %d/%d emails processed", processed_count, total_emails)
            log.info("-"*60)

            # Gentle memory cleanup - don't clear indexes!
            log.info("Performing batch cleanup and garbage collection...")
            self._cleanup_memory_safe()

        # Update state
        if max_success_dt:
            self.state.last_sync_utc = max_success_dt
        self.state.save()

        # Final cleanup
        self._cleanup_memory_safe()

        # Final summary
        log.info("Export complete: %d/%d emails processed, %d conversations exported, %d skipped, %d errors",
                processed_count, total_emails, exported_conversations, skipped_count, error_count)

    def _pool_and_index_emails(self, folder_paths: list[str], since_utc: datetime | None) -> list[EmailMetadata]:
        """Pool all emails using date-chunked processing to avoid COM corruption"""

        email_pool: list[EmailMetadata] = []
        seen_entry_ids: set[str] = set()

        log.info("Building conversation indexes with chunked processing...")

        for folder_path in folder_paths:
            try:
                folder = self._get_folder(folder_path)

                # Get store ID for this folder
                try:
                    store_id = folder.StoreID
                except Exception:
                    store_id = None

                # Process in monthly chunks to avoid COM corruption
                folder_emails = self._process_folder_in_chunks(
                    folder, folder_path, store_id, since_utc, seen_entry_ids
                )

                # Add to pool and index
                for email_meta in folder_emails:
                    email_pool.append(email_meta)

                    # Store the store ID mapping
                    if store_id:
                        self.store_ids[email_meta.entry_id] = store_id

                    # Index by ConversationID if available
                    if email_meta.conv_id:
                        if email_meta.conv_id not in self.conversation_index:
                            self.conversation_index[email_meta.conv_id] = []
                        self.conversation_index[email_meta.conv_id].append(email_meta)

                    # Always index by conversation key as fallback
                    if email_meta.conv_key not in self.fallback_index:
                        self.fallback_index[email_meta.conv_key] = []
                    self.fallback_index[email_meta.conv_key].append(email_meta)

                log.info("Indexed %d emails from '%s'", len(folder_emails), folder_path)

            except Exception as e:
                log.error("Failed to index emails from '%s': %s", folder_path, e)
                continue

        # Sort emails newest to oldest
        sorted_pool = sorted(email_pool, key=lambda x: x.sort_date, reverse=True)

        log.info("Indexed %d total emails across %d ConversationIDs and %d conversation keys",
                len(sorted_pool), len(self.conversation_index), len(self.fallback_index))

        return sorted_pool

    def _process_folder_in_chunks(
        self,
        folder: Any,
        folder_path: str,
        store_id: str | None,
        since_utc: datetime | None,
        seen_entry_ids: set[str]
    ) -> list[EmailMetadata]:
        """Process a folder in date-based chunks to avoid COM corruption"""

        from datetime import timedelta

        # Determine date range
        end_date = datetime.now()
        # Default to last 5 years if no since date
        start_date = since_utc or end_date - timedelta(days=5 * 365)

        log.info("Processing '%s' in chunks from %s to %s",
                folder_path, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        folder_emails: list[EmailMetadata] = []
        current_date = start_date
        chunk_size_days = 60  # 2-month chunks to keep Items collections small

        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=chunk_size_days), end_date)

            try:
                # Create date-restricted Items collection for this chunk
                start_filter = outlook_restrict_datetime(current_date)
                end_filter = outlook_restrict_datetime(chunk_end)
                restrict_filter = f"[ReceivedTime] >= '{start_filter}' AND [ReceivedTime] < '{end_filter}'"

                items = folder.Items
                items = items.Restrict(restrict_filter)
                items.Sort("[ReceivedTime]", False)  # Oldest first within chunk

                try:
                    chunk_count = items.Count
                    log.info(
                        "Chunk %s to %s: %d items",
                        current_date.strftime("%Y-%m-%d"),
                        chunk_end.strftime("%Y-%m-%d"),
                        chunk_count,
                    )
                except Exception:
                    chunk_count = 0

                # Process this chunk with limited size
                chunk_emails = self._process_items_collection(
                    items, chunk_count, folder_path, store_id, seen_entry_ids
                )

                folder_emails.extend(chunk_emails)

                # Clean up COM references
                del items
                gc.collect()

            except Exception as e:
                log.warning("Failed to process chunk %s to %s in '%s': %s",
                           current_date.strftime("%Y-%m-%d"),
                           chunk_end.strftime("%Y-%m-%d"),
                           folder_path, e)

            # Move to next chunk
            current_date = chunk_end

        return folder_emails

    def _process_items_collection(
        self,
        items: Any,
        item_count: int,
        folder_path: str,
        store_id: str | None,
        seen_entry_ids: set[str]
    ) -> list[EmailMetadata]:
        """Process a single Items collection with corruption detection"""

        chunk_emails: list[EmailMetadata] = []
        processed = 0

        # Track consecutive duplicate EntryIDs to detect corruption
        last_entry_id = None
        consecutive_duplicates = 0
        corruption_threshold = 10  # If same ID appears 10 times in a row, corruption detected

        # Limit to 2000 items per chunk as safety measure
        max_items = min(item_count, 2000)

        for i in range(1, max_items + 1):
            try:
                item = items.Item(i)

                # Only process mail items
                if getattr(item, "Class", None) != OL_MAIL:
                    continue

                # Get entry ID
                entry_id = getattr(item, "EntryID", None)
                if not entry_id:
                    continue

                # Detect COM corruption: same EntryID repeating
                if entry_id == last_entry_id:
                    consecutive_duplicates += 1
                    if consecutive_duplicates >= corruption_threshold:
                        log.error("COM corruption detected in '%s' at item %d - same EntryID %s repeated %d times. Stopping chunk processing.",
                                folder_path, i, entry_id[:20], consecutive_duplicates)
                        break
                else:
                    consecutive_duplicates = 0
                    last_entry_id = entry_id

                # Skip already seen entries
                if entry_id in seen_entry_ids:
                    log.debug("Duplicate EntryID detected: %s", entry_id[:20])
                    continue

                seen_entry_ids.add(entry_id)

                # Get conversation identifiers
                conv_id = getattr(item, "ConversationID", None)
                conv_key, _is_fallback = get_conversation_key(item)

                # Get date for sorting
                sender_email = get_sender_smtp(item)
                is_from_me = (
                    sender_email and self.user_email and
                    self.user_email.lower() in sender_email.lower()
                )

                if is_from_me:
                    sort_date = getattr(item, "SentOn", None) or getattr(item, "ReceivedTime", None)
                else:
                    sort_date = getattr(item, "ReceivedTime", None) or getattr(item, "SentOn", None)

                if not sort_date:
                    continue

                # Get subject
                subject = getattr(item, "Subject", "") or ""

                # Create metadata object
                email_meta = EmailMetadata(
                    entry_id=entry_id,
                    conv_id=conv_id,
                    conv_key=conv_key,
                    sort_date=sort_date,
                    folder_path=folder_path,
                    subject=subject,
                    store_id=store_id
                )
                chunk_emails.append(email_meta)
                processed += 1

                # Release COM reference
                del item

            except Exception as e:
                log.debug("Failed to process item %d: %s", i, e)
                continue

        log.info("Processed %d unique emails from chunk in '%s'", len(chunk_emails), folder_path)
        return chunk_emails

    def _get_conversation_emails(self, email_meta: EmailMetadata) -> list[Any]:
        """Get all emails in a conversation by re-fetching from Outlook"""

        conversation_metadata: list[EmailMetadata] = []
        seen_entry_ids: set[str] = set()

        # First get metadata from indexes
        if email_meta.conv_id and email_meta.conv_id in self.conversation_index:
            log.debug("Using ConversationID index for: %s", email_meta.conv_id[:20])
            for meta in self.conversation_index[email_meta.conv_id]:
                if meta.entry_id not in seen_entry_ids:
                    seen_entry_ids.add(meta.entry_id)
                    conversation_metadata.append(meta)

        # Fallback to conversation key index
        if not conversation_metadata and email_meta.conv_key in self.fallback_index:
            log.debug("Using conversation key index for: %s", email_meta.conv_key[:20])
            for meta in self.fallback_index[email_meta.conv_key]:
                if meta.entry_id not in seen_entry_ids:
                    seen_entry_ids.add(meta.entry_id)
                    conversation_metadata.append(meta)

        # Sort metadata chronologically
        conversation_metadata.sort(key=lambda m: m.sort_date)

        # Now re-fetch the actual emails from Outlook
        conversation_emails: list[Any] = []
        for meta in conversation_metadata:
            try:
                # Re-fetch email using improved method
                email = self._get_email_by_id_fixed(meta.entry_id, meta.store_id, meta.folder_path)
                if email:
                    conversation_emails.append(email)
                else:
                    log.warning("Could not re-fetch email %s", meta.entry_id[:20])
            except Exception as e:
                log.warning("Failed to re-fetch email %s: %s", meta.entry_id[:20], e)
                continue

        return conversation_emails

    def _get_email_by_id_fixed(self, entry_id: str, store_id: str | None, folder_path: str) -> Any:
        """Fixed method to reliably re-fetch an email from Outlook"""

        try:
            # Try GetItemFromID with StoreID (most reliable)
            if store_id:
                try:
                    return self.ns.GetItemFromID(entry_id, store_id)
                except pywintypes.com_error as e:
                    log.debug("GetItemFromID with StoreID failed: %s", e)

            # Try GetItemFromID without StoreID
            try:
                return self.ns.GetItemFromID(entry_id)
            except pywintypes.com_error as e:
                log.debug("GetItemFromID without StoreID failed: %s", e)

            # Fallback: iterate through the folder (slower but more reliable)
            folder = self._get_folder(folder_path)
            items = folder.Items

            # Try to find by iterating (last resort)
            for i in range(1, min(items.Count + 1, 5000)):  # Limit search
                try:
                    item = items.Item(i)
                    if getattr(item, "EntryID", None) == entry_id:
                        return item
                except Exception:
                    continue

            log.warning("Could not find email with EntryID %s in any method", entry_id[:20])
            return None

        except Exception as e:
            log.error("Failed to get email by ID %s: %s", entry_id[:20], e)
            return None

    def _check_incremental_update(self, conversation_id: str, emails: list[Any]) -> tuple[bool, datetime | None]:
        """Check if conversation needs incremental update"""

        # Check if conversation already exists
        existing_dir = self.state.get_conversation_dir(conversation_id)
        if not existing_dir:
            # New conversation
            return True, None

        # Load existing manifest
        manifest_path = self.output_root / existing_dir / "manifest.json"
        if not manifest_path.exists():
            # Manifest missing, re-export
            return True, None

        try:
            with manifest_path.open(encoding="utf-8") as f:
                manifest = json.load(f)

            # Get last email date from manifest
            timeline = manifest.get("timeline", {})
            end_str = timeline.get("end")
            if not end_str:
                # No timeline info, re-export
                return True, None

            # Parse the date
            last_date = datetime.fromisoformat(end_str.replace('Z', '+00:00'))

            # Check if we have newer emails
            newest_email_date = None
            for email in emails:
                try:
                    email_date = getattr(email, "ReceivedTime", None) or getattr(email, "SentOn", None)
                    if email_date and (newest_email_date is None or email_date > newest_email_date):
                        newest_email_date = email_date
                except Exception:
                    # Skip if we can't get date
                    continue

            if newest_email_date and newest_email_date > last_date:
                # We have newer emails
                log.debug("Conversation %s has new emails since %s", conversation_id[:20], end_str)
                return True, last_date
            else:
                # No new emails
                return False, None

        except Exception as e:
            log.warning("Failed to check incremental update for %s: %s", conversation_id[:20], e)
            # On error, re-export to be safe
            return True, None

    def _export_conversation(self, conv_id: str, emails: list[Any]) -> datetime | None:
        """Export a single conversation with incremental update support"""

        if not emails:
            return None

        try:
            # Check if conversation already exists (for incremental)
            existing_dir = self.state.get_conversation_dir(conv_id)
            previous_manifest = None

            if existing_dir:
                # Load existing manifest to check for incremental update
                manifest_path = self.output_root / existing_dir / "manifest.json"
                if manifest_path.exists():
                    try:
                        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except Exception as e:
                        log.warning("Failed to load previous manifest: %s", e)

            # Create conversation directory
            first_subject = getattr(emails[0], "Subject", "") or "no subject"
            dir_name = existing_dir or self._safe_folder_name(conv_id, first_subject)
            conv_dir = self.output_root / dir_name
            ensure_dir(conv_dir)

            # Extract bodies and SMTP data
            cached_bodies: list[str] = []
            cached_smtp_data: list[
                tuple[str, list[dict[str, str]], list[dict[str, str]]]
            ] = []
            conv_max_dt = None

            # Check for headers-only emails
            full_item_count = 0

            for email in emails:
                try:
                    # Check download state
                    download_state = getattr(email, "DownloadState", None)
                    if download_state == OL_FULLITEM:
                        full_item_count += 1

                    # Extract body (handle both text and HTML)
                    body = ""
                    try:
                        # Try to get plain text body first
                        body = getattr(email, "Body", "") or ""

                        # If no plain text, try HTML
                        if not body:
                            html_body = getattr(email, "HTMLBody", "")
                            if html_body:
                                # Import HTML conversion utilities
                                from .manifest_builder import (
                                    html_to_text,
                                    looks_like_html,
                                )
                                if looks_like_html(html_body):
                                    body = html_to_text(html_body)
                                else:
                                    body = html_body
                    except Exception as e:
                        log.debug("Failed to extract body: %s", e)

                    cached_bodies.append(body)

                    # Extract SMTP data
                    sender_smtp = get_sender_smtp(email)
                    to_list = get_recipients_array(email, OL_TO)
                    cc_list = get_recipients_array(email, OL_CC)
                    cached_smtp_data.append((sender_smtp, to_list, cc_list))

                    # Track max date
                    rcv = getattr(email, "ReceivedTime", None) or getattr(email, "SentOn", None)
                    if rcv and (conv_max_dt is None or rcv > conv_max_dt):
                        conv_max_dt = rcv

                except Exception as e:
                    log.warning("Failed to extract email data: %s", e)
                    cached_bodies.append("")
                    cached_smtp_data.append(("", [], []))

            # Warn if most emails are headers-only
            if full_item_count < len(emails) / 2:
                log.warning("Only %d/%d emails have full content for conversation %s",
                           full_item_count, len(emails), conv_id[:20])

            # Generate manifest (with previous manifest for incremental)
            manifest = generate_manifest(
                emails, conv_id, conv_dir, cached_bodies, cached_smtp_data,
                previous_manifest=previous_manifest
            )

            # Build conversation text
            conv_text = build_conversation_text(emails, cached_bodies, cached_smtp_data)
            (conv_dir / "Conversation.txt").write_text(conv_text, encoding="utf-8")

            # Save attachments (with deduplication for incremental)
            _, att_meta = save_attachments_for_items(
                emails, conv_dir,
                previous_manifest=previous_manifest
            )
            manifest["has_attachments"] = bool(att_meta)
            if att_meta:
                manifest["attachments"] = att_meta

            # Save manifest
            (conv_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            # Update state
            self.state.mark_conversation_exported(conv_id, dir_name)

            log.info("Exported conversation %s with %d emails to %s",
                    conv_id[:20], len(emails), dir_name)

            return conv_max_dt

        except Exception as e:
            log.error("Failed to export conversation %s: %s", conv_id[:20], e)
            return None

    def _cleanup_memory_safe(self):
        """Safe memory cleanup that doesn't corrupt COM references"""

        # Don't clear the indexes during processing!
        # Just do garbage collection
        collected = gc.collect()
        log.debug("GC collected %d objects", collected)

        # Get memory stats if available
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            log.info("Memory usage: %.2f MB", mem_info.rss / 1024 / 1024)
        except ImportError:
            pass

    def _get_folder(self, path: str) -> Any:
        """Get Outlook folder by path (with caching)"""

        # Check cache first
        if path in self.folder_cache:
            return self.folder_cache[path]

        if not path.startswith("\\"):
            raise ValueError(f"Folder path must start with \\: {path}")
        parts = [p for p in path.split("\\") if p]
        if not parts:
            raise ValueError(f"Invalid folder path: {path}")

        root_name = parts[0]
        folder = None

        try:
            # Try to get the folder directly
            folder = self.ns.Folders.Item(root_name)
        except Exception as e:
            # If that fails, try to find it in the default account
            log.debug("Failed to get root folder '%s' directly: %s", root_name, e)
            # Try iterating through all folders to find the right one
            for f in self.ns.Folders:
                try:
                    if f.Name == root_name or str(f) == root_name:
                        folder = f
                        break
                except Exception:
                    continue

            if folder is None:
                raise RuntimeError(f"Root folder '{root_name}' not found: {e}") from e

        # Navigate to subfolders
        for name in parts[1:]:
            try:
                folder = folder.Folders.Item(name)
            except Exception as e:
                msg = f"Subfolder '{name}' under '{root_name}' not found: {e}"
                raise RuntimeError(msg) from e

        # Cache the folder reference
        self.folder_cache[path] = folder

        return folder

    def _safe_folder_name(self, conv_key: str, subject: str) -> str:
        """Generate safe folder name for conversation"""
        prefix = f"C_{short_hex(conv_key, 6)}"
        subject_part = sanitize_filename(subject).replace(" ", "_") if subject else ""
        combined = f"{prefix}_{subject_part}" if subject_part else prefix
        return combined[:100]
