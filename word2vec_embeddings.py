"""
word2vec_embeddings.py

Trains a Word2Vec model on the Enron corpus (reuses data_prep.data_prep.load_enron_dataset)
and exposes get_embedding_matrix() as the interface lstm_priority_classifier.py consumes.

Training is driven from training/train_word2vec.py (Colab-friendly CLI). This file stays a
plain importable module: pipeline.py / lstm_priority_classifier.py import from here directly,
they never shell out to the CLI script.

NLP Module 4 deliverable.
"""

import os
import re
import time

import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from data_prep.data_prep import load_enron_dataset

_TOKEN_RE = re.compile(r"[a-zA-Z']+")

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 10

DEFAULT_MODEL_PATH = "saved_models/word2vec_enron.model"


class _EpochLogger(CallbackAny2Vec):
    """Prints timing after each epoch so training is never silent/blind."""
    def __init__(self):
        self.epoch = 0
        self.start = None

    def on_epoch_begin(self, model):
        self.start = time.time()

    def on_epoch_end(self, model):
        elapsed = time.time() - self.start
        print(f"  epoch {self.epoch} done in {elapsed:.1f}s")
        self.epoch += 1


def tokenize(text: str) -> list[str]:
    # regex tokenizer: lowercases and strips punctuation, keeping internal
    # apostrophes (don't -> "don't", not "don" + "t"). Plain .split() was
    # tried first but left punctuation glued to words ("deadline.", "urgent,"),
    # fragmenting the vocab -- e.g. "invoice" and "invoice." became two
    # different tokens with separate, half-trained vectors. This fixes that.
    return _TOKEN_RE.findall(text.lower())


def build_corpus(enron_csv: str) -> list[list[str]]:
    """Reuses load_enron_dataset from data_prep -- does NOT re-parse the raw emails."""
    df = load_enron_dataset(enron_csv)
    corpus = (df["subject"].fillna("") + " " + df["body"].fillna("")).apply(tokenize)
    return corpus.tolist()


def train_word2vec(
    enron_csv: str,
    sg: int = 1,
    out_path: str = DEFAULT_MODEL_PATH,
    vector_size: int = VECTOR_SIZE,
    window: int = WINDOW,
    min_count: int = MIN_COUNT,
    epochs: int = EPOCHS,
    workers: int = 3,
) -> Word2Vec:
    """sg=1 -> skip-gram, sg=0 -> CBOW."""
    corpus = build_corpus(enron_csv)
    print(f"training on {len(corpus)} documents, {workers} workers, {epochs} epochs")

    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=workers,
        callbacks=[_EpochLogger()],
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.save(out_path)
    print(f"saved -> {out_path}")
    return model


def get_embedding_matrix(vocab: dict, model_path: str = DEFAULT_MODEL_PATH) -> np.ndarray:
    """
    vocab: {word: index} from the LSTM tokenizer (lstm_priority_classifier.py).
    Returns a numpy array shaped [len(vocab), VECTOR_SIZE].
    Words not found in the trained Word2Vec vocab get a small random init.
    """
    model = Word2Vec.load(model_path)
    dim = model.wv.vector_size
    matrix = np.random.normal(scale=0.1, size=(len(vocab), dim)).astype(np.float32)

    hits = 0
    for word, idx in vocab.items():
        if word in model.wv:
            matrix[idx] = model.wv[word]
            hits += 1

    print(f"embedding coverage: {hits}/{len(vocab)} words found in word2vec vocab")
    return matrix


def nearest_neighbors(word: str, model_path: str = DEFAULT_MODEL_PATH, topn: int = 8):
    """Sanity check for the viva -- proves the model actually learned from Enron."""
    model = Word2Vec.load(model_path)
    return model.wv.most_similar(word, topn=topn)