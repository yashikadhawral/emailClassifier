"""
training/train_word2vec.py

CLI entry point for Phase 1 (NLP Module 4). Thin wrapper around
word2vec_embeddings.py -- run this from the repo root (not from inside
training/) so the data_prep/ and word2vec_embeddings.py imports resolve:

    python training/train_word2vec.py \
        --enron_csv data/raw/enron_emails.csv \
        --sg 1 \
        --out_path saved_models/word2vec_enron.model

On Colab: mount Drive first, point --out_path at the mounted saved_models/
folder so the trained model survives the runtime disconnecting, then copy
it into your local saved_models/ before running the API locally.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from word2vec_embeddings import train_word2vec, nearest_neighbors

CHECK_WORDS = ["invoice", "meeting", "urgent", "deadline"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enron_csv", required=True)
    parser.add_argument("--sg", type=int, default=1, help="1=skip-gram, 0=CBOW")
    parser.add_argument("--out_path", default="saved_models/word2vec_enron.model")
    parser.add_argument("--vector_size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    train_word2vec(
        args.enron_csv, sg=args.sg, out_path=args.out_path,
        vector_size=args.vector_size, window=args.window,
        min_count=args.min_count, epochs=args.epochs,
    )

    print("\n--- nearest-neighbor sanity check ---")
    for w in CHECK_WORDS:
        try:
            print(f"\n{w}:", nearest_neighbors(w, args.out_path))
        except KeyError:
            print(f"\n{w}: not in vocab (try lowering --min_count)")