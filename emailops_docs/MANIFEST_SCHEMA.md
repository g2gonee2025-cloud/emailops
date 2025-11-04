# EmailOps manifest.json Schema Documentation

This is the **single source of truth** for the manifest.json schema used by EmailOps.

## Complete Schema

```json
{
  "subject": "RE: Insurance - details of all furniture transfer",
  
  "messages": [
    {
      "from": {
        "name": "Patrick Chalhoub",
        "smtp": "patrick@company.com"
      },
      "to": [
        {"name": "Insurance Dept", "smtp": "insurance@company.com"},
        {"name": "Admin Team", "smtp": "admin@company.com"}
      ],
      "cc": [
        {"name": "Manager", "smtp": "manager@company.com"}
      ],
      "date": "2025-05-08T15:02:24Z",
      "subject": "RE: Insurance - details...",
      "text": "Full email body content here..."
    }
  ]
}
```

**Note:** `time_span` object removed - dates come from `messages[].date` field.

## Required Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `subject` | string | YES | Email subject line |
| `messages` | array | YES | Array of message objects (at least 1) |
| `messages[].from` | object | YES | Sender info |
| `messages[].from.name` | string | YES | Sender display name |
| `messages[].from.smtp` | string | YES | Sender email address |
| `messages[].to` | array | YES | Recipients (can be empty) |
| `messages[].cc` | array | NO | CC recipients (can be empty) |
| `messages[].date` | string | YES | ISO 8601 with T separator |
| `messages[].text` | string | NO | Email body (full text recommended) |

**Dates:** Extracted from `messages[]` array:
- First message date = conversation start
- Last message date = conversation end

## Date Format

**Required format:** ISO 8601 with `T` separator

```
✅ CORRECT: "2025-05-08T15:02:24"
❌ WRONG:   "2025-05-08 15:02:24"  (space separator not supported)
```

## Filter Support

### `from:` Filter (OR Logic)
Matches if sender email is in the from list.
```
from:john@company.com               // Match John as sender
from:john@company.com,jane@company.com  // Match John OR Jane
```

### `to:` Filter (OR Logic)
Matches if ANY recipient email matches.
```
to:jane@company.com                 // Match if Jane in To
to:jane@company.com,bob@company.com     // Match Jane OR Bob
```

### `has:attachment` Filter
Checks if chunk has attachment metadata.
```
has:attachment      // Only chunks from attachments
has:noattachment    // Only conversation text chunks
```

Detected by presence of `attachment_name` field in chunk.

## Outlook Export Requirements

1. **Create `Conversation.txt`** (can be empty) - required for discovery
2. **Create `manifest.json`** with schema above
3. **Dates in messages must use T separator** - `"2025-05-08T15:02:24Z"`
4. **Include full email body** in `messages[].text` - no truncation needed

## Processing Flow

```
manifest.json
  ↓
Load body from messages[].text or Conversation.txt
  ↓
Clean text
  ↓
Chunk text
  ↓
Extract metadata (from/to emails, dates from messages[])
  ↓
Add metadata to EACH chunk
  ↓
Save to _chunks/{conv_id}.json (metadata complete)
  ↓
Indexing loads chunks (no manifest parsing needed)
```

Metadata extracted AFTER successful chunking for efficiency.

## Implementation

All schema parsing handled by [`core_manifest.py`](emailops/core_manifest.py) - no other module should duplicate this logic.
