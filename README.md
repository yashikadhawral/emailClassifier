# Email Intelligence Report

An NLP/Deep Learning system that turns a raw email into a structured, actionable report: category, priority, summary, extracted key info, and a suggested reply — plus on-demand translation.

## What it does

Paste in a raw email, and the system returns:
- **Category** — work / academic / finance / personal / promotions
- **Priority** — urgent / important / normal / low
- **Summary** — a short, headline-style gist of the email
- **Key info** — deadline, people mentioned, referenced documents, sentiment (extracted via NER + heuristics)
- **Suggested reply** — a grounded, template-based draft built from the extracted fields (never invents details not in the email)
- **Translation** — view the summary in English, Hindi, or Marathi

## Architecture

```
frontend.html  →  FastAPI (api.py)  →  pipeline.py  →  individual models
```

| Component | File | Approach |
|---|---|---|
| Priority classification | `priority_classifier.py` | Fine-tuned DistilBERT, with a negation-aware keyword fallback/override for under-confident predictions |
| Category classification | `category_classifier.py` | Zero-shot (`facebook/bart-large-mnli`) — no labeled data required |
| Summarization | `summarizer.py` | T5-small fine-tuned on AESLC (headline-style summaries), falls back to `facebook/bart-large-cnn` if untrained |
| Key info extraction | `key_info_extractor.py` | spaCy NER (people, dates) + regex (filenames) + keyword-based sentiment |
| Reply drafting | `template_reply.py` | Rule-based, built entirely from extracted key info — no LLM hallucination risk |
| Translation | `translator.py` | Wraps a translation model for language detection + EN ↔ Hindi/Marathi |
| Orchestration | `pipeline.py` | `EmailAssistantPipeline` — wires detection → translation → priority → summary → reply together |
| API | `api.py` | FastAPI wrapper exposing `/api/analyze` and `/api/translate` for the frontend |
| Frontend | `frontend/frontend.html` | Single-file HTML/CSS/JS dashboard — inbox list + report panel, no build step |

## Running it

**1. Install dependencies**
```bash
pip install fastapi uvicorn spacy torch transformers datasets rouge_score
python -m spacy download en_core_web_sm
```

**2. Start the API**
```bash
uvicorn api:app --reload --port 8000
```

**3. Open the frontend**
Just open `frontend/frontend.html` in a browser (or serve it via `python -m http.server` from its folder). Click **+ New**, paste an email, hit **Analyze**.

## Training your own models

- `data_prep.py` — builds priority training data from Enron (manual labelling subset) + spam datasets (auto-labelled Low)
- `auto_label_priority.py` / `label_priority.py` — pre-label and spot-check the manual labelling subset
- `train_priority_classifier.py` — fine-tunes DistilBERT on the labelled priority data
- `train_summarizer.py` — fine-tunes T5-small on AESLC for headline-style summaries

The category classifier and reply generator currently don't require training (zero-shot and rule-based respectively) — see the docstrings in those files for the trade-offs and how to swap in a fine-tuned version later.

## Known limitations

- Category classification is zero-shot, not fine-tuned on real email distribution — slower and less sharp than a trained model
- Summaries are headline-style (from AESLC), not multi-sentence paragraphs, by design
- Suggested replies are template-based for reliability, not free-generated — less "smart" phrasing but grounded in extracted facts
- No action-item extraction yet (`actions` is currently always empty)
- Deadline extraction uses generic spaCy DATE/TIME entities, not deadline-specific logic — occasionally over- or under-inclusive

## Project structure

```
emailClassifier/
├── api.py                        # FastAPI server
├── pipeline.py                   # orchestrates the full email → report flow
├── priority_classifier.py
├── category_classifier.py
├── summarizer.py
├── translator.py
├── response_generator.py         # model-based reply drafting (flan-t5), currently unused in favor of template_reply.py
├── template_reply.py             # rule-based reply drafting (currently used)
├── key_info_extractor.py
├── data_prep.py
├── train_priority_classifier.py
├── train_summarizer.py
├── auto_label_priority.py
├── label_priority.py
├── frontend/
│   └── frontend.html
├── saved_models/                 # fine-tuned model checkpoints (gitignored)
├── data/                         # datasets (gitignored)
└── requirements.txt
```