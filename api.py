# api.py
import json
import os
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline import EmailAssistantPipeline
from key_info_extractor import extract_key_info
from category_classifier import CategoryClassifier
from template_reply import build_reply
from integrations import calendar_client, gmail_client
from integrations.google_auth import NotAuthorized

ROOT = Path(__file__).resolve().parent

app = FastAPI()

# The inbox endpoint returns real email, so don't allow every website to call
# it (the old allow_origins=["*"] would let any page you have open read your
# mail). The dashboard is served by this same server at /app/, so it's
# same-origin and doesn't need CORS at all; these entries are just a safety net.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

pipeline = EmailAssistantPipeline()
category_clf = CategoryClassifier()

PRIORITY_MAP = {
    "High": "urgent",
    "Medium": "normal",
    "Low": "low",
}

MAX_EMAIL_CHARS = 4000          # long newsletters/threads: the models truncate anyway
INBOX_FETCH_LIMIT = 25
CACHE_KEEP = 100
CACHE_PATH = ROOT / "data" / "inbox_cache.json"   # data/ is gitignored


class EmailIn(BaseModel):
    text: str

class TranslateIn(BaseModel):
    text: str
    lang: str  # "hi" or "mr"

class EventIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    start: str                       # "YYYY-MM-DDTHH:MM" (IST) or full ISO
    duration_min: int = Field(default=30, ge=5, le=480)
    description: str = Field(default="", max_length=2000)
    source_id: Optional[str] = Field(default=None, max_length=200)


def parse_headers(raw: str):
    from_match = re.search(r"^from:\s*(.+)$", raw, re.IGNORECASE | re.MULTILINE)
    subject_match = re.search(r"^subject:\s*(.+)$", raw, re.IGNORECASE | re.MULTILINE)

    from_ = from_match.group(1).strip() if from_match else "Unknown sender"

    if subject_match:
        subject = subject_match.group(1).strip()
    else:
        # no Subject: line -> fall back to the first non-empty line of the body
        first_line = next((l.strip() for l in raw.splitlines() if l.strip()), "")
        subject = (first_line[:70] + "…") if len(first_line) > 70 else (first_line or "No subject")

    return from_, subject


def analyze_text(raw: str, from_: Optional[str] = None, subject: Optional[str] = None,
                 received: Optional[datetime] = None, time_label: str = "Just now") -> dict:
    """
    The analysis the old /api/analyze did inline, now shared by the paste box
    and the Gmail sync. from_/subject/received are passed in for Gmail messages
    (known exactly); for pasted text they're parsed out of the text as before.
    """
    parsed_from, parsed_subject = parse_headers(raw)
    from_ = from_ or parsed_from
    subject = subject or parsed_subject

    extracted = extract_key_info(raw)

    # no "From:" header found -> fall back to the last person name mentioned,
    # since that's usually the sign-off/signature at the bottom of the email
    if from_ == "Unknown sender":
        people_field = next((f for f in extracted if f["k"] == "People"), None)
        if people_field:
            names = [n.strip() for n in people_field["v"].split(",")]
            from_ = names[-1]

    result = pipeline.process(raw, sender_name=from_)
    frontend_priority = PRIORITY_MAP.get(result["priority"], "normal")

    # key info + category run on the English text, so Hindi/Marathi emails still
    # yield deadlines the date parser can read (spaCy en model is English-only)
    english = result["translated_input"] or raw
    if result["translated_input"]:
        extracted = extract_key_info(english)
    category = category_clf.predict(english)

    # grounded, rule-based reply instead of flan-t5's free-generation --
    # every claim in it traces back to something extracted, never invented
    reply = build_reply(extracted, priority=frontend_priority, sender_name=from_)
    if result["detected_language"] != "en":
        reply = pipeline.translator.translate(reply, target_lang=result["detected_language"])

    key_info = [
        {"k": "Detected language", "v": result["detected_language"]},
    ]
    key_info.extend(extracted)
    key_info.append({"k": "Suggested reply", "v": reply})

    # prefilled calendar event, only when a usable date/time was found
    calendar = None
    if category != "promotions":
        calendar = calendar_client.suggest_event(subject, extracted, base=received)

    return {
        "id": int(time.time() * 1000),
        "from": from_,
        "subject": subject,
        "time": time_label,
        "category": category,
        "priority": frontend_priority,
        "summary": result["summary"],
        "keyInfo": key_info,
        "actions": [],  # TODO: no action-item extractor yet
        "calendar": calendar,
        "raw": raw,
    }


@app.post("/api/analyze")
def analyze(payload: EmailIn):
    return analyze_text(payload.text)


@app.post("/api/translate")
def translate_text(payload: TranslateIn):
    translated = pipeline.translator.translate(payload.text, target_lang=payload.lang)
    return {"translated": translated}


# ---------------- Gmail sync ----------------

_inbox_lock = threading.Lock()
_failed_ids = set()   # messages whose analysis crashed; skipped until restart so we don't retry every minute


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict) -> None:
    newest = sorted(cache.values(), key=lambda e: e["ts"], reverse=True)[:CACHE_KEEP]
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({e["id"]: e for e in newest}), encoding="utf-8")
    os.replace(tmp, CACHE_PATH)   # atomic: a crash can't leave a half-written cache


def _time_label(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, calendar_client.TZ)
    today = datetime.now(calendar_client.TZ).date()
    if dt.date() == today:
        return dt.strftime("%H:%M")
    if dt.date() == today - timedelta(days=1):
        return "Yesterday"
    return dt.strftime("%d %b")


@app.get("/api/inbox")
def inbox(max_results: int = 10):
    """
    Lists recent inbox mail, analyzes only messages not seen before (the models
    are slow; results are cached in data/inbox_cache.json), and returns them
    newest-first. The first sync after a fresh start can take a while.
    """
    max_results = max(1, min(max_results, INBOX_FETCH_LIMIT))
    try:
        ids = gmail_client.list_message_ids(max_results=max_results)
    except NotAuthorized as e:
        raise HTTPException(status_code=401, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gmail request failed: {e}")

    with _inbox_lock:
        cache = _load_cache()
        new_ids = [i for i in ids if i not in cache and i not in _failed_ids]
        for mid in new_ids:
            try:
                msg = gmail_client.fetch_message(mid)
                body_text = f"{msg.subject}\n\n{msg.body}"[:MAX_EMAIL_CHARS]
                item = analyze_text(body_text, from_=msg.sender, subject=msg.subject,
                                    received=msg.date)
                item["id"] = mid
                item["link"] = msg.link
                item["ts"] = msg.date.timestamp()
                cache[mid] = item
            except NotAuthorized as e:
                raise HTTPException(status_code=401, detail=str(e))
            except Exception as e:
                print(f"[inbox] skipping message {mid}: {e}")
                _failed_ids.add(mid)
        if new_ids:
            _save_cache(cache)

        emails = sorted(cache.values(), key=lambda e: e["ts"], reverse=True)
        emails = [dict(e, time=_time_label(e["ts"])) for e in emails[:50]]

    return {"emails": emails, "new_ids": [i for i in new_ids if i in cache]}


# ---------------- Google Calendar ----------------

@app.post("/api/calendar/add")
def add_to_calendar(payload: EventIn):
    try:
        start = calendar_client.parse_start(payload.start)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid start time")
    try:
        return calendar_client.create_event(
            title=payload.title,
            start=start,
            duration_min=payload.duration_min,
            description=payload.description,
            source_id=payload.source_id,
        )
    except NotAuthorized as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Calendar request failed: {e}")


# Serve the dashboard from this server: open http://localhost:8000/app/frontend.html
# Keep this mount LAST so it never shadows the /api routes.
app.mount("/app", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")