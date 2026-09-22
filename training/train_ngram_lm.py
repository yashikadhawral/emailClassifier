"""
training/train_ngram_lm.py

CLI entry point for the N-gram LM module (NLP coursework module).
Trains a smoothed N-gram model on Enron email text, reports perplexity
on a held-out split, generates sample sentences.

    python training/train_ngram_lm.py \
        --enron_csv "{path}/emails.csv" \
        --n 3 --k 1.0 --sample_size 5000
"""
import argparse
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ngram_lm import NGramLanguageModel, load_sentences

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enron_csv", required=True)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--k", type=float, default=1.0, help="Laplace smoothing constant")
    parser.add_argument("--sample_size", type=int, default=5000)
    args = parser.parse_args()

    sentences = load_sentences(args.enron_csv, sample_size=args.sample_size)
    random.seed(42)
    random.shuffle(sentences)
    split = int(0.9 * len(sentences))
    train_sents, test_sents = sentences[:split], sentences[split:]

    model = NGramLanguageModel(n=args.n, k=args.k)
    model.train(train_sents)

    ppl = model.perplexity(test_sents)
    print(f"\nHeld-out perplexity ({len(test_sents)} sentences): {ppl:.2f}")

    print("\n--- sample generations ---")
    for i in range(5):
        print(f"{i+1}. {model.generate(seed=i)}")