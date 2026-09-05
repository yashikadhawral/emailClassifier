"""
summarizer.py

Inference-only wrapper for email summarization.
Uses your fine-tuned model (train_summarizer.py) if it exists at
`model_dir`; otherwise falls back to pretrained facebook/bart-large-cnn
so the pipeline still runs before you've fine-tuned anything.

NOTE: loads the model/tokenizer directly with AutoModelForSeq2SeqLM
instead of transformers.pipeline("summarization", ...) — newer
transformers versions (v5+) restructured the pipeline task registry
and dropped/renamed some task strings, so this avoids relying on that
lookup entirely.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

FALLBACK_MODEL = "facebook/bart-large-cnn"


class Summarizer:
    def __init__(self, model_dir: str = "saved_models/summarizer"):
        self.model_dir = model_dir
        self._is_trained = os.path.isdir(model_dir) and bool(os.listdir(model_dir))
        self._tokenizer = None
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self):
        if self._model is None:
            model_name = self.model_dir if self._is_trained else FALLBACK_MODEL
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self._device)
            self._model.eval()

    def summarize(self, text: str, max_length: int = 60, min_length: int = 8) -> str:
        self._load()
        prefix = "summarize: " if self._is_trained else ""  # matches T5 training prefix
        inputs = self._tokenizer(
            prefix + text,
            return_tensors="pt",
            truncation=True,
            max_length=384,
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                num_beams=4,
            )

        summary = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summary.strip()


if __name__ == "__main__":
    s = Summarizer()
    print(s.summarize(
        "Hi team, following up on last week's discussion about the Q3 budget. "
        "We need to finalize the marketing spend allocation by Friday, and I'd "
        "like everyone's input on the proposed cuts to the events line item "
        "before then. Please review the attached sheet and reply with comments."
    ))