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
    dates = [ent.text for ent in doc.ents if ent.label_ in ("DATE", "TIME")]
    if not dates:
        return None
    # prefer dates near urgency keywords if present, else just the first date found
    return ", ".join(dict.fromkeys(dates))  # dedupe, preserve order


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