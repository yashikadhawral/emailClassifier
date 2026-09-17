"""
label_priority.py

Spot-check tool for the auto-labelled priority subset (run
auto_label_priority.py first). You review a limited number of the
LOWEST-confidence predictions - accept with Enter, override with h/m/l,
skip, or quit anytime. Everything else keeps the model's prediction.

This is much faster than labelling from scratch: you're confirming or
correcting a suggestion, not deciding cold on every single email.

Usage:
    python label_priority.py --review_n 100        # interactive review
    python label_priority.py --finalize             # accept all remaining
                                                      # predictions as-is,
                                                      # no more review
"""

import argparse
import pandas as pd

LABEL_MAP = {"h": "High", "m": "Medium", "l": "Low"}


def review_loop(csv_path: str, review_n: int):
    df = pd.read_csv(csv_path)
    for col in ["predicted_priority", "confidence", "priority"]:
        if col not in df.columns:
            raise ValueError(
                f"'{col}' column missing - run auto_label_priority.py first."
            )
    df["priority"] = df["priority"].fillna("")

    unfinalized = df.index[df["priority"] == ""].tolist()
    # lowest confidence first = most worth reviewing
    unfinalized = sorted(unfinalized, key=lambda i: df.at[i, "confidence"])

    total = len(df)
    print(f"{total - len(unfinalized)}/{total} already finalized. {len(unfinalized)} left.")
    print(f"Reviewing up to {review_n} of the lowest-confidence ones this run.\n")

    reviewed = 0
    for i in unfinalized:
        if reviewed >= review_n:
            print(f"\nHit your review limit ({review_n}). Run again for more, "
                  f"or use --finalize to auto-accept everything still unlabelled.")
            break

        subject = str(df.at[i, "subject"])[:120]
        body = str(df.at[i, "body"])[:300]
        pred = df.at[i, "predicted_priority"]
        conf = df.at[i, "confidence"]

        print("-" * 70)
        print(f"Subject: {subject}")
        print(f"Body   : {body}{'...' if len(str(df.at[i, 'body'])) > 300 else ''}")
        choice = input(
            f"Model says: {pred} (confidence {conf:.2f}) - "
            f"[Enter]=accept, h/m/l=override, s=skip, q=quit: "
        ).strip().lower()

        if choice == "q":
            print("Saving and exiting.")
            break
        if choice == "s":
            continue  # leave unlabelled, revisit next run
        if choice == "":
            df.at[i, "priority"] = pred
        elif choice in LABEL_MAP:
            df.at[i, "priority"] = LABEL_MAP[choice]
        else:
            print("Not a valid option, skipping this one.")
            continue

        reviewed += 1
        df.to_csv(csv_path, index=False)  # save after every decision

    done = (df["priority"] != "").sum()
    print(f"\n{done}/{total} finalized so far ({reviewed} reviewed this run).")


def finalize_remaining(csv_path: str):
    df = pd.read_csv(csv_path)
    df["priority"] = df["priority"].fillna("")
    mask = df["priority"] == ""
    n = int(mask.sum())
    df.loc[mask, "priority"] = df.loc[mask, "predicted_priority"]
    df.to_csv(csv_path, index=False)
    print(f"Auto-accepted {n} remaining predictions. All {len(df)} rows now finalized.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="priority_auto_labelled.csv")
    parser.add_argument("--review_n", type=int, default=100)
    parser.add_argument("--finalize", action="store_true", help="Skip review, accept all remaining predictions")
    args = parser.parse_args()

    if args.finalize:
        finalize_remaining(args.csv)
    else:
        review_loop(args.csv, args.review_n)