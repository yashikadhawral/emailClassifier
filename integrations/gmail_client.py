"""
gmail_client.py

Read-only Gmail access: list recent inbox messages and turn one into plain
text the pipeline can analyze.

Two steps on purpose, so the caller can skip messages it has already
analyzed without downloading them again:
    ids = list_message_ids()        # cheap: ids only
    msg = fetch_message(ids[0])     # full message, parsed
"""

import base64
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.header import decode_header, make_header
from email.utils import parseaddr, parsedate_to_datetime
from html.parser import HTMLParser

from googleapiclient.discovery import build

from .google_auth import get_credentials

DEFAULT_QUERY = "in:inbox newer_than:3d"


@dataclass
class GmailMessage:
    id: str
    sender: str        # display name if present, else the address
    sender_email: str
    subject: str
    date: datetime     # timezone-aware
    body: str          # plain text, quoted replies removed
    link: str


def _service():
    return build("gmail", "v1", credentials=get_credentials(), cache_discovery=False)


def list_message_ids(max_results: int = 10, query: str = DEFAULT_QUERY) -> list:
    resp = _service().users().messages().list(
        userId="me", q=query, maxResults=max_results
    ).execute()
    return [m["id"] for m in resp.get("messages", [])]


def fetch_message(message_id: str) -> GmailMessage:
    raw = _service().users().messages().get(
        userId="me", id=message_id, format="full"
    ).execute()
    return parse_message(raw)


# ---- parsing (pure functions, no network) ----

def parse_message(raw: dict) -> GmailMessage:
    payload = raw.get("payload", {})
    headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}

    name, addr = parseaddr(_decode_header(headers.get("from", "")))
    subject = _decode_header(headers.get("subject", "")).strip() or "(no subject)"

    return GmailMessage(
        id=raw["id"],
        sender=name or addr or "Unknown sender",
        sender_email=addr,
        subject=subject,
        date=_parse_date(headers.get("date"), raw.get("internalDate")),
        body=strip_quoted(_extract_body(payload)),
        link=f"https://mail.google.com/mail/u/0/#all/{raw['id']}",
    )


def _decode_header(value: str) -> str:
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def _parse_date(date_header, internal_date_ms) -> datetime:
    if date_header:
        try:
            dt = parsedate_to_datetime(date_header)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (TypeError, ValueError):
            pass
    if internal_date_ms:
        return datetime.fromtimestamp(int(internal_date_ms) / 1000, tz=timezone.utc)
    return datetime.now(timezone.utc)


def _b64(data: str) -> str:
    data += "=" * (-len(data) % 4)  # Gmail strips padding
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")


def _walk(part: dict):
    yield part
    for sub in part.get("parts", []) or []:
        yield from _walk(sub)


def _extract_body(payload: dict) -> str:
    """Prefer text/plain; fall back to text/html with tags stripped.
    Attachments (parts with a filename) are ignored."""
    plain, html = [], []
    for part in _walk(payload):
        if part.get("filename"):
            continue
        data = part.get("body", {}).get("data")
        if not data:
            continue
        mime = part.get("mimeType", "")
        if mime == "text/plain":
            plain.append(_b64(data))
        elif mime == "text/html":
            html.append(_b64(data))
    if plain:
        return "\n".join(plain).strip()
    if html:
        return html_to_text("\n".join(html))
    return ""


class _TextExtractor(HTMLParser):
    BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "table"}

    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _TextExtractor()
    p.feed(html)
    text = "".join(p.parts)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


_REPLY_HEADER = re.compile(r"^(On .{5,200} wrote:|-{2,} ?Original Message ?-{2,}|From: .+)$", re.I)


def strip_quoted(text: str) -> str:
    """Drop quoted reply history so old dates/deadlines from earlier messages
    in the thread don't leak into extraction."""
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if _REPLY_HEADER.match(stripped) and kept:
            break
        if stripped.startswith(">"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()