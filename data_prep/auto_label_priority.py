"""
auto_label_priority.py

Auto-labels the Enron subset for priority using a ZERO-SHOT classifier
(no training needed) instead of you reading and judging all ~400 emails
from scratch. Produces predicted_priority + a confidence score per row,
sorted so the LOWEST-confidence (most likely wrong / most worth checking)
rows come first.

You then run label_priority.py to spot-check the low-confidence ones —
much faster than manual labelling from zero, since you're just
accepting/overriding a suggestion instead of deciding cold each time.

Note: bart-large-mnli is a big model. This is fast on a Colab GPU
(a few minutes for ~400 rows), noticeably slower on CPU. Be patient
or move this step to Colab if it's dragging on your laptop.

Usage:
    python auto_label_priority.py --csv priority_labelling_subset.csv
"""

import argparse
import pandas as pd
import torch
from transformers import pipeline

# phrase the classifier sees -> the short label we actually store
CANDIDATE_LABELS = {
    "urgent, needs immediate action": "High",
    "needs a reply but not urgent": "Medium",
    "informational, no action needed": "Low",
}


def auto_label(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading zero-shot classifier (device={'GPU' if device == 0 else 'CPU'})...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

    labels_list = list(CANDIDATE_LABELS.keys())
    predicted_priority, confidence = [], []

    print(f"Classifying {len(df)} emails...")
    for i, row in df.iterrows():
        text = f"{row['subject']}. {row['body']}"[:800]  # truncate for speed
        result = classifier(text, candidate_labels=labels_list)
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        predicted_priority.append(CANDIDATE_LABELS[top_label])
        confidence.append(round(top_score, 4))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(df)} done")

    df["predicted_priority"] = predicted_priority
    df["confidence"] = confidence
    if "priority" not in df.columns:
        df["priority"] = ""

    # lowest confidence first -> these are the ones worth your attention
    df = df.sort_values("confidence", ascending=True).reset_index(drop=True)
    df.to_csv(out_path, index=False)

    print("\nPredicted class distribution:")
    print(df["predicted_priority"].value_counts())
    print(f"\nSaved -> {out_path}")
    print("Next: python label_priority.py  (spot-checks the lowest-confidence rows first)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="priority_labelling_subset.csv")
    parser.add_argument("--out", default="priority_auto_labelled.csv")
    args = parser.parse_args()
    auto_label(args.csv, args.out)