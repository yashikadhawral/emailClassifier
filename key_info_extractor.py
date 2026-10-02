"""
key_info_extractor.py

Lightweight, no-training-required extraction of "key info" fields for the
frontend's keyInfo cards: Deadline, People, Referenced doc, Sentiment.

Uses spaCy's small English model for NER (PERSON entities, DATE spans)
plus regex for filenames, and a keyword-based sentiment heuristic (no
model download needed). Swap the sentiment heuristic for a real
classifier later if you want something more robust.

Setup:
    pip install spacy --break-system-packages
    python -m spacy download en_core_web_sm
"""

import re
import spacy

_NLP = None

FILENAME_PATTERN = re.compile(
    r"\b[\w,\s-]+\.(pptx?|docx?|pdf|xlsx?|csv|zip|png|jpg|jpeg)\b", re.IGNORECASE
)

URGENT_WORDS = ["urgent", "asap", "immediately", "critical", "right away", "deadline"]
POSITIVE_WORDS = ["thanks", "appreciate", "great", "glad", "looking forward", "congrats"]
NEGATIVE_WORDS = ["concerned", "issue", "problem", "failing", "delay", "sorry", "unfortunately"]

# words that, appearing shortly before an urgency keyword, hedge/negate it
# (e.g. "if anything urgent comes up" is NOT actually urgent)
HEDGE_WORDS = {"if", "anything", "nothing", "no", "unless", "without", "never"}
HEDGE_WINDOW = 3


# --- deadline cleanup -------------------------------------------------------
# spaCy's DATE/TIME labels are noisy on real mail: ZIP codes ("CA 94107"), phone
# numbers, bare years, and event names ("Creator Day") all get tagged, while
# explicit dates like "21 Oct" or "15/10/2026" are sometimes missed. So: drop
# the obvious junk, and add a regex pass for explicit dates/times.
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
          r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_DAY_WORDS = r"(?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)[a-z]*|today|tomorrow|tonight|noon|eod|cob"

# a DATE/TIME entity must contain at least one of these to be kept
_DATE_SIGNAL = re.compile(
    rf"\b{_MONTH}\b|\b(?:{_DAY_WORDS})\b|\b(?:next|this|coming|last)\b|\bweek|\bmonth|\bend of\b"
    r"|\b\d{1,2}\s*[ap]\.?m\b|\b\d{1,2}:\d{2}\b|\b\d{1,2}[/-]\d{1,2}\b"
    r"|\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:days?|hours?)\b",
    re.IGNORECASE,
)

_REGEX_DATES = re.compile(
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}\b(?:,?\s+\d{{4}})?"     # 21 Oct, 5th October 2026
    rf"|\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?:,?\s+\d{{4}})?"          # October 21, Oct 21 2026
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"                                        # 15/10/2026
    r"|\b\d{1,2}(?::\d{2})?\s*[ap]\.?m\b"                                       # 3:30 pm, 11 AM
    r"|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow)\b",   # NER sometimes misses these
    re.IGNORECASE,
)


_ID_NUMBERS = re.compile(r"\+?\d[\d\-\s()]{6,}\d|\d{5,}")   # phone numbers, ZIP/PIN codes, invoice ids


def _clean_date_entity(text: str):
    """Strips ID-like numbers out of a spaCy DATE/TIME span ("022-2345-6789 tomorrow"
    -> "tomorrow"); returns None if nothing date-like is left ("CA 94107")."""
    t = re.sub(r"\s+", " ", _ID_NUMBERS.sub(" ", text)).strip(" ,.-")
    return t if t and _DATE_SIGNAL.search(t) else None


def _keyword_hit(lowered_text: str, keywords: list) -> bool:
    words = re.findall(r"[a-z0-9']+", lowered_text)
    for kw in keywords:
        kw_words = kw.split()
        n = len(kw_words)
        for i in range(len(words) - n + 1):
            if words[i:i + n] == kw_words:
                window = words[max(0, i - HEDGE_WINDOW):i]
                if any(hw in window for hw in HEDGE_WORDS):
                    continue
                return True
    return False


def _load_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _extract_people(doc) -> list:
    people = []
    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.text not in people:
            people.append(ent.text)
    return people


def _extract_deadline(doc) -> str:
    found = [c for c in (_clean_date_entity(ent.text) for ent in doc.ents
                         if ent.label_ in ("DATE", "TIME")) if c]

    # regex pass: explicit dates/times spaCy missed or only half-caught. spaCy often
    # splits "3rd october, 2026" into ORDINAL "3rd" + DATE "october, 2026", so when a
    # regex hit is a longer version of an entity we already have, it replaces that entity.
    for m in _REGEX_DATES.finditer(doc.text):
        hit = m.group(0).strip()
        low = hit.lower()
        if any(low in f.lower() for f in found):
            continue                                   # already covered by an equal/longer entity
        found = [f for f in found if f.lower() not in low]   # drop shorter fragments of this hit
        found.append(hit)

    if not found:
        return None
    lowered = doc.text.lower()
    found.sort(key=lambda f: lowered.find(f.lower()))     # reading order, so the date comes before its time
    return ", ".join(dict.fromkeys(found))  # dedupe


def _extract_referenced_doc(text: str) -> str:
    matches = FILENAME_PATTERN.findall(text)
    full_matches = FILENAME_PATTERN.finditer(text)
    filenames = [m.group(0).strip() for m in full_matches]
    return ", ".join(dict.fromkeys(filenames)) if filenames else None


def _extract_sentiment(text: str) -> str:
    lowered = text.lower()
    urgent_hit = _keyword_hit(lowered, URGENT_WORDS)
    neg_hit = any(w in lowered for w in NEGATIVE_WORDS)
    pos_hit = any(w in lowered for w in POSITIVE_WORDS)

    if urgent_hit and neg_hit:
        return "Time-pressured, concerned"
    if urgent_hit:
        return "Time-pressured, direct"
    if neg_hit:
        return "Concerned, direct"
    if pos_hit:
        return "Positive, cordial"
    return "Neutral"


def extract_key_info(text: str) -> list:
    """
    Returns a list of {"k": ..., "v": ...} dicts matching the frontend's
    keyInfo card format. Only includes fields that were actually found.
    """
    nlp = _load_nlp()
    doc = nlp(text)

    fields = []

    deadline = _extract_deadline(doc)
    if deadline:
        fields.append({"k": "Deadline", "v": deadline})

    people = _extract_people(doc)
    if people:
        fields.append({"k": "People", "v": ", ".join(people)})

    referenced_doc = _extract_referenced_doc(text)
    if referenced_doc:
        fields.append({"k": "Referenced doc", "v": referenced_doc})

    fields.append({"k": "Sentiment", "v": _extract_sentiment(text)})

    return fields


if __name__ == "__main__":
    sample = (
        "Hi, URGENT - we need the signed contract back by end of day today. "
        "Please review the attached Q3_eval_slides_v4.pptx and send it to "
        "Priya Nandakumar before Thursday."
    )
    for field in extract_key_info(sample):
        print(field)