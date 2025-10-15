## JSON String Scrubbing Utility

### `scrub_json_string(raw: str) -> str`

Removes control characters from a raw JSON string before parsing. Use this before `json.loads` on any user-supplied or external JSON to prevent parse errors and ensure hygiene.

**Example:**

```python
from emailops.utils import scrub_json_string
import json
safe_json = scrub_json_string(raw_json)
data = json.loads(safe_json)
```
# `utils.py` — Text Extraction, Cleaning, and Conversation Loading

> **Purpose:** Foundation utilities: robust file text extraction, email‑aware cleaning, and conversation/attachment loading used by the indexer and downstream tools.

---

## 1) Extraction

### `extract_text(path: Path, max_chars: int | None = None) -> str`
Dispatches by extension; returns sanitized text (never raises).

| Type | Extensions | Primary libs | Fallback |
|---|---|---|---|
| Plain | .txt .md .log .json .yaml/.yml .csv .xml .html/.htm | stdlib | multi‑encoding read; HTML → text |
| PDF | .pdf | `pypdf` | per‑page best‑effort; skip encrypted if cannot decrypt with empty pass |
| Word | .docx | `python-docx` | — |
| Word (legacy) | .doc | `win32com` (Windows) | `textract` if available |
| Excel | .xlsx/.xls | `pandas` + engine | auto‑engine; rows capped by `EXCEL_MAX_CELLS` |
| PowerPoint | .pptx/.ppt | `python-pptx` | — |
| Email | .eml | `email` stdlib | HTML → text fallback |
| Outlook | .msg | `extract-msg` | — |
| RTF | .rtf | `striprtf` | — |

**Sanitization**
- Multi‑encoding fallback: `utf-8-sig` → `utf-8` → `utf-16` → `latin-1` (ignore).  
- Control chars stripped; CR/LF normalized.

---

## 2) Email Cleaning & Parsing

### `clean_email_text(text: str) -> str`
- Removes common headers (From/To/Subject/etc.).  
- Strips simple signatures/footers from the tail portion.  
- Removes forwarding separators and quoted `>` lines.  
- Redacts emails → `[email@domain]` and URLs → `[URL]`.  
- Collapses repeated punctuation/whitespace/newlines.

### `extract_email_metadata(text: str) -> dict`
- Best‑effort regex extraction of `From/To/Cc/Date/Subject` into a simple dict.

### `split_email_thread(text: str) -> list[str]`
- Splits on “Original/Forwarded Message” markers and “On … wrote:” lines.  
- Sorts chronologically when multiple `Date:` headers are present.

---

## 3) Conversations

### `find_conversation_dirs(root: Path) -> list[Path]`
- A conversation is any directory that contains `Conversation.txt` (recursive search).

### `load_conversation(convo_dir: Path, include_attachment_text=False, max_total_attachment_text=10000, *, max_attachment_text_chars=env, skip_if_attachment_over_mb=env) -> dict`
- Reads conversation text and parses `manifest.json` / `summary.json` (BOM‑safe).  
- Collects candidate attachments from `Attachments/` and the conversation folder (excluding known system files).  
- Deduplicates attachment paths; extracts text (size‑capped and truncated).  
- Optionally appends small snippets of attachment text into the conversation body.

**Environment knobs**
- `MAX_ATTACHMENT_TEXT_CHARS` (default **500000**)  
- `SKIP_ATTACHMENT_OVER_MB` (default **0**, disabled)  
- `EXCEL_MAX_CELLS` (default **200000**)

---

## 4) Misc

- `ensure_dir(Path)` idempotently creates directories.  
- `Person` class exposes an `age` property derived from an ISO birthdate string.

