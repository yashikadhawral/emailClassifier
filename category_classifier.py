"""
category_classifier.py

Zero-shot email category classification -- no labeled training data
required. Uses facebook/bart-large-mnli (an NLI model repurposed for
zero-shot classification): given the email text and a list of candidate
category names, it scores how well each label fits and returns the best
match.

Trade-off vs a fine-tuned classifier (like priority_classifier.py once
trained): this is slower per-request and won't be as sharp as a model
trained on your actual email distribution, but needs zero labeled data
to get started. Swap this out for a fine-tuned DistilBERT later if you
end up labeling category data.
"""

from transformers import pipeline

CATEGORIES = ["work", "academic", "finance", "personal", "promotions"]


class CategoryClassifier:
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self._model_name = model_name
        self._clf = None

    def _load(self):
        if self._clf is None:
            self._clf = pipeline("zero-shot-classification", model=self._model_name)

    def predict(self, text: str) -> str:
        self._load()
        result = self._clf(text, candidate_labels=CATEGORIES)
        return result["labels"][0]  # highest-scoring category


if __name__ == "__main__":
    clf = CategoryClassifier()
    print(clf.predict(
        "Hi, URGENT - we need the signed contract back by end of day today. "
        "Please review the attached PDF and send your signature ASAP."
    ))
    print(clf.predict(
        "Dear Mr. Sharma, I am writing to request leave for two days, "
        "15 September and 16 September 2026, as I need to attend a family "
        "function in my hometown."
    ))