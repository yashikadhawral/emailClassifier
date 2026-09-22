"""
ngram_lm.py

From-scratch N-gram language model with Laplace (add-k) smoothing.
Trained on Enron email bodies. No external LM libraries.
"""
import re
import random
import math
from collections import defaultdict, Counter
from email import message_from_string

import pandas as pd
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize

START, END = "<s>", "</s>"


def extract_body(raw_message: str) -> str:
    """Enron's emails.csv 'message' column holds a full raw RFC-822
    message (headers + blank line + body). Parse out just the body."""
    try:
        msg = message_from_string(str(raw_message))
        payload = msg.get_payload()
        return payload if isinstance(payload, str) else str(raw_message)
    except Exception:
        parts = re.split(r'\n\s*\n', str(raw_message), maxsplit=1)
        return parts[1] if len(parts) > 1 else str(raw_message)


def clean_text(text: str) -> str:
    text = re.sub(r'-{2,}.*?-{2,}', ' ', str(text))  # strip forwarded-by banners
    text = re.sub(r'\S+@\S+', ' ', text)              # strip emails
    text = re.sub(r'[^a-zA-Z0-9.,!? ]', ' ', text)
    return text.lower()


def load_sentences(csv_path: str, text_col: str = "message", sample_size: int = 5000):
    df = pd.read_csv(csv_path, nrows=sample_size)
    col = text_col if text_col in df.columns else df.columns[-1]
    sentences = []
    for raw in df[col].dropna():
        body = extract_body(raw)
        for sent in sent_tokenize(clean_text(body)):
            tokens = word_tokenize(sent)
            if 2 <= len(tokens) <= 40:
                sentences.append(tokens)
    return sentences


class NGramLanguageModel:
    def __init__(self, n: int = 3, k: float = 1.0):
        self.n = n
        self.k = k
        self.ngram_counts = defaultdict(Counter)   # context -> {next_word: count}
        self.context_totals = Counter()
        self.vocab = set()

    def _pad(self, tokens):
        return [START] * (self.n - 1) + tokens + [END]

    def train(self, sentences):
        for tokens in sentences:
            padded = self._pad(tokens)
            self.vocab.update(padded)
            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i:i + self.n - 1])
                next_word = padded[i + self.n - 1]
                self.ngram_counts[context][next_word] += 1
                self.context_totals[context] += 1
        print(f"Trained {self.n}-gram model: {len(self.vocab)} vocab, "
              f"{len(self.ngram_counts)} contexts")

    def prob(self, context, word):
        V = len(self.vocab)
        count = self.ngram_counts[context][word]
        total = self.context_totals[context]
        return (count + self.k) / (total + self.k * V)

    def perplexity(self, sentences):
        log_prob_sum, N = 0.0, 0
        for tokens in sentences:
            padded = self._pad(tokens)
            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i:i + self.n - 1])
                word = padded[i + self.n - 1]
                log_prob_sum += math.log(self.prob(context, word) + 1e-12)
                N += 1
        return math.exp(-log_prob_sum / N) if N else float('inf')

    def generate(self, max_len: int = 20, seed: int = None):
        if seed is not None:
            random.seed(seed)
        context = tuple([START] * (self.n - 1))
        out = []
        for _ in range(max_len):
            candidates = self.ngram_counts.get(context)
            if not candidates:
                break
            words, weights = zip(*candidates.items())
            next_word = random.choices(words, weights=weights)[0]
            if next_word == END:
                break
            out.append(next_word)
            context = tuple(list(context[1:]) + [next_word])
        return " ".join(out)