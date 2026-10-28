# api.py
import re
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import EmailAssistantPipeline
from key_info_extractor import extract_key_info
from category_classifier import CategoryClassifier

ROOT = Path(__file__).resolve().parent

app = FastAPI(title="Email Intelligence Report API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pipeline = EmailAssistantPipeline()
category_clf = CategoryClassifier()

PRIORITY_MAP = {
    "High": "urgent",
    "Medium": "normal",
    "Low": "low",
}

class EmailIn(BaseModel):
    text: str

class TranslateIn(BaseModel):
    text: str
    lang: str  # "hi" or "mr"

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

@app.get("/")
def index():
    """Serves the dashboard directly when navigating to http://localhost:8000/"""
    return FileResponse(ROOT / "frontend" / "frontend.html")

@app.post("/api/analyze")
def analyze(payload: EmailIn):
    raw = payload.text
    from_, subject = parse_headers(raw)

    extracted = extract_key_info(raw)

    # no "From:" header found -> fall back to the last person name mentioned,
    # since that's usually the sign-off/signature at the bottom of the email
    if from_ == "Unknown sender":
        people_field = next((f for f in extracted if f["k"] == "People"), None)
        if people_field:
            names = [n.strip() for n in people_field["v"].split(",")]
            from_ = names[-1]

    result = pipeline.process(raw, sender_name=from_, extracted_info=extracted)
    frontend_priority = PRIORITY_MAP.get(result["priority"], "normal")

    # Grounded rule-based reply from pipeline (never hallucinates, unlike free-form flan-t5)
    reply = result["suggested_response"]

    key_info = [
        {"k": "Detected language", "v": result["detected_language"]},
    ]
    key_info.extend(extracted)
    key_info.append({"k": "Suggested reply", "v": reply})

    return {
        "id": int(time.time() * 1000),
        "from": from_,
        "subject": subject,
        "time": "Just now",
        "category": category_clf.predict(raw),
        "priority": frontend_priority,
        "summary": result["summary"],
        "keyInfo": key_info,
        "actions": [],  # TODO: no action-item extractor yet
        "raw": raw
    }

@app.post("/api/translate")
def translate_text(payload: TranslateIn):
    translated = pipeline.translator.translate(payload.text, target_lang=payload.lang)
    return {"translated": translated}

# Mount static files so assets can also be served under /frontend
app.mount("/frontend", StaticFiles(directory=str(ROOT / "frontend")), name="frontend")
