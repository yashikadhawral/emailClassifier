"""
priority_classifier.py

Inference-only wrapper around the fine-tuned DistilBERT priority model
(see train_priority_classifier.py for training).

If no trained model is found at `model_dir` yet (e.g. you haven't labelled
data and trained it), falls back to a simple keyword heuristic so the rest
of the app/pipeline still runs end-to-end during development. Swap it out
once your fine-tuned model exists -- the fallback is deliberately basic.

On top of the model's prediction, a keyword override bumps anything with
strong urgency signals up to High -- the training data is Medium-heavy
(zero-shot labelling hedged toward Medium on ambiguous cases), so the
model under-predicts High even on blatantly urgent emails. This override
is a stopgap until the training labels themselves get cleaned up.

NOTE: the keyword check is negation-aware -- it ignores matches that are
hedged/conditional ("if anything urgent comes up", "nothing urgent"),
since a bare substring match on "urgent" flags plenty of emails that
mention the word only to say urgency is NOT expected.
"""

import os
import re
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

ID2LABEL = {0: "Low", 1: "Medium", 2: "High"}

HIGH_KEYWORDS = [
    "urgent", "asap", "as soon as possible", "immediately", "deadline",
    "action required", "critical", "eod", "end of day", "time-sensitive",
    "right away", "top priority",
]
LOW_KEYWORDS = ["newsletter", "unsubscribe", "no reply needed", "fyi", "promotion"]

# words that, appearing shortly before a keyword, hedge/negate it
# (e.g. "if anything urgent", "nothing urgent", "no immediate action")
HEDGE_WORDS = {"if", "anything", "nothing", "no", "unless", "without", "never"}
HEDGE_WINDOW = 3  # how many preceding words to check


def _keyword_hit(lowered_text: str, keywords: list) -> bool:
    words = re.findall(r"[a-z0-9']+", lowered_text)
    for kw in keywords:
        kw_words = kw.split()
        n = len(kw_words)
        for i in range(len(words) - n + 1):
            if words[i:i + n] == kw_words:
                window = words[max(0, i - HEDGE_WINDOW):i]
                if any(hw in window for hw in HEDGE_WORDS):
                    continue  # hedged/negated mention -- doesn't count
                return True
    return False


def _fallback_predict(text: str) -> str:
    lowered = text.lower()
    if _keyword_hit(lowered, HIGH_KEYWORDS):
        return "High"
    if _keyword_hit(lowered, LOW_KEYWORDS):
        return "Low"
    return "Medium"


def _has_high_signal(text: str) -> bool:
    return _keyword_hit(text.lower(), HIGH_KEYWORDS)


def _has_low_signal(text: str) -> bool:
    return _keyword_hit(text.lower(), LOW_KEYWORDS)


class PriorityClassifier:
    def __init__(self, model_dir: str = "saved_models/priority_classifier"):
        self.model_dir = model_dir
        self._tokenizer = None
        self._model = None
        self._is_trained = os.path.isdir(model_dir) and os.listdir(model_dir)

    def _load(self):
        if self._model is None and self._is_trained:
            self._tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_dir)
            self._model = DistilBertForSequenceClassification.from_pretrained(self.model_dir)
            self._model.eval()

    def predict(self, text: str) -> str:
        if not self._is_trained:
            return _fallback_predict(text)

        self._load()
        inputs = self._tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**inputs).logits
        pred_id = int(torch.argmax(logits, dim=1))
        pred = ID2LABEL[pred_id]

        # override: model under-predicts High due to Medium-heavy training
        # labels, so trust strong explicit urgency keywords over the model
        # when it says anything less than High.
        if pred != "High" and _has_high_signal(text):
            return "High"
        if pred == "Medium" and _has_low_signal(text):
            return "Low"
        return pred


if __name__ == "__main__":
    clf = PriorityClassifier()
    print(clf.predict("URGENT: please review the attached contract before EOD."))
    print(clf.predict("Here's our monthly newsletter, no action needed."))
    print(clf.predict("I'll be available by phone if anything urgent comes up."))