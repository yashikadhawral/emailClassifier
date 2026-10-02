"""
calendar_client.py

Two jobs:
  1. suggest_event() -- turn the "Deadline" text that key_info_extractor found
     ("Thursday, 11:00 AM", "Fri, EOD", "tomorrow at 5pm", "15 Sep 2026")
     into a real datetime. Pure parsing, no Google login needed.
  2. create_event()  -- insert that event into the user's primary Google
     Calendar with phone-notification reminders. Only runs when the user
     clicks "Add to calendar".

Dates are resolved relative to when the email was RECEIVED (so "Thursday" in
an email from Monday means that week's Thursday), in TIMEZONE below.

Deliberately conservative:
  - a day but no time  -> default to 09:00 and flag it (time_defaulted)
  - a time but no day  -> assume the day the email was received and flag it (day_assumed)
  - only a month or year, no day ("October 2026") -> no suggestion; we won't guess
  - resolved time already in the past -> no suggestion
The UI turns the flags into a "check this before adding" note.
"""

import re
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import dateparser

TIMEZONE = "Asia/Kolkata"
TZ = ZoneInfo(TIMEZONE)

DEFAULT_HOUR = 9     # used when the email names a day but no time
EOD_HOUR = 17        # "EOD" / "end of day" / "COB"
TONIGHT_HOUR = 20
REMINDER_MINUTES = (60, 10)   # popup reminders -> phone notifications

_WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}
_WEEKDAY_RE = re.compile(
    r"\b(?:(next|this|coming)\s+)?(" + "|".join(sorted(_WEEKDAYS, key=len, reverse=True)) + r")\b",
    re.I,
)
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
          r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_EXPLICIT_DATE_RE = re.compile(
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}\b"      # 15 September
    rf"|\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b"              # September 15
    r"|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",                    # 15/09/2026
    re.I,
)
_AMPM_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?(?!\w)", re.I)
_24H_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_EOD_RE = re.compile(r"\b(?:eod|cob|end of (?:the )?day|close of business)\b", re.I)
_NOON_RE = re.compile(r"\bnoon\b", re.I)


def _extract_time(text: str):
    """Returns (hour, minute, text_with_time_removed) or (None, None, text)."""
    m = _AMPM_RE.search(text)
    if m:
        hour, minute = int(m.group(1)) % 12, int(m.group(2) or 0)
        if m.group(3).lower() == "p":
            hour += 12
        return hour, minute, text[:m.start()] + " " + text[m.end():]
    m = _24H_RE.search(text)
    if m:
        return int(m.group(1)), int(m.group(2)), text[:m.start()] + " " + text[m.end():]
    m = _EOD_RE.search(text)
    if m:
        return EOD_HOUR, 0, text[:m.start()] + " " + text[m.end():]
    m = _NOON_RE.search(text)
    if m:
        return 12, 0, text[:m.start()] + " " + text[m.end():]
    return None, None, text


def _extract_date(text: str, base: datetime):
    """Returns a date or None."""
    lowered = text.lower()
    if re.search(r"\btoday\b|\btonight\b", lowered):
        return base.date()
    if re.search(r"\btomorrow\b", lowered):
        return (base + timedelta(days=1)).date()

    m = _WEEKDAY_RE.search(text)
    if m:
        qualifier, day = (m.group(1) or "").lower(), m.group(2).lower()
        delta = (_WEEKDAYS[day] - base.weekday()) % 7
        if qualifier == "next" and delta == 0:
            delta = 7
        return (base + timedelta(days=delta)).date()

    if _EXPLICIT_DATE_RE.search(text):
        parsed = dateparser.parse(text, settings={
            "DATE_ORDER": "DMY",
            "PREFER_DATES_FROM": "current_period",
            "RELATIVE_BASE": base,
            "TIMEZONE": TIMEZONE,
            "RETURN_AS_TIMEZONE_AWARE": True,
        })
        if parsed:
            return parsed.date()
    return None


def parse_deadline(text: str, base: Optional[datetime] = None, now: Optional[datetime] = None) -> Optional[dict]:
    """
    text: deadline string from the extractor, e.g. "Thursday, 11:00 AM".
    base: when the email was received (relative words resolve against this).
    now:  current time, for the "already in the past" check.
    Returns {"start": aware datetime, "time_defaulted": bool, "day_assumed": bool} or None.
    """
    if not text:
        return None
    now = (now or datetime.now(TZ)).astimezone(TZ)
    base = (base or now).astimezone(TZ)

    hour, minute, remainder = _extract_time(text)
    day = _extract_date(remainder, base)

    if day is None and hour is None:
        return None            # nothing usable ("two days", "this year", "Q3" ...)
    day_assumed = day is None
    if day_assumed:
        if re.search(rf"\b{_MONTH}\b|\b(?:19|20)\d{{2}}\b", remainder, re.I):
            return None        # a month or year with no day: too vague to guess
        day = base.date()      # time only, e.g. "by 6 PM" -> the day the email arrived

    time_defaulted = hour is None
    if time_defaulted:
        hour, minute = (TONIGHT_HOUR, 0) if re.search(r"\btonight\b", text, re.I) else (DEFAULT_HOUR, 0)

    start = datetime(day.year, day.month, day.day, hour, minute, tzinfo=TZ)
    if start < now - timedelta(minutes=5):
        return None
    return {"start": start, "time_defaulted": time_defaulted, "day_assumed": day_assumed}


def suggest_event(subject: str, key_info: list, base: Optional[datetime] = None,
                  now: Optional[datetime] = None) -> Optional[dict]:
    """Builds the prefilled event the UI offers, or None if there's no usable deadline."""
    deadline = next((f["v"] for f in key_info if f["k"] == "Deadline"), None)
    parsed = parse_deadline(deadline, base=base, now=now)
    if not parsed:
        return None
    title = re.sub(r"^((re|fwd?|fw):\s*)+", "", subject or "", flags=re.I).strip() or "Email deadline"
    return {
        "title": title[:200],
        "start": parsed["start"].strftime("%Y-%m-%dT%H:%M"),   # local, for <input type=datetime-local>
        "time_defaulted": parsed["time_defaulted"],
        "day_assumed": parsed["day_assumed"],
        "deadline_text": deadline,
    }


def parse_start(value: str) -> datetime:
    """Accepts 'YYYY-MM-DDTHH:MM' (assumed TIMEZONE) or a full ISO string. Raises ValueError."""
    dt = datetime.fromisoformat(value)
    return dt.replace(tzinfo=TZ) if dt.tzinfo is None else dt


def create_event(title: str, start: datetime, duration_min: int = 30,
                 description: str = "", source_id: Optional[str] = None) -> dict:
    """
    Inserts the event into the primary calendar. source_id (the Gmail message id)
    is stored on the event, so clicking twice never creates a duplicate.
    Returns {"id", "link", "existing"}.
    """
    from googleapiclient.discovery import build   # lazy: parsing above works without Google libs
    from .google_auth import get_credentials

    service = build("calendar", "v3", credentials=get_credentials(), cache_discovery=False)

    if source_id:
        found = service.events().list(
            calendarId="primary",
            privateExtendedProperty=f"sourceId={source_id}",
            maxResults=5,
        ).execute().get("items", [])
        found = [e for e in found if e.get("status") != "cancelled"]
        if found:
            return {"id": found[0]["id"], "link": found[0].get("htmlLink"), "existing": True}

    end = start + timedelta(minutes=duration_min)
    body = {
        "summary": title,
        "description": description[:2000],
        "start": {"dateTime": start.isoformat(), "timeZone": TIMEZONE},
        "end": {"dateTime": end.isoformat(), "timeZone": TIMEZONE},
        "reminders": {
            "useDefault": False,
            "overrides": [{"method": "popup", "minutes": m} for m in REMINDER_MINUTES],
        },
    }
    if source_id:
        body["extendedProperties"] = {"private": {"sourceId": source_id}}

    event = service.events().insert(calendarId="primary", body=body).execute()
    return {"id": event["id"], "link": event.get("htmlLink"), "existing": False}