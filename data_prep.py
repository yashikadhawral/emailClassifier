"""
data_prep.py

Prepares two things:
1. A combined "base" priority dataset:
     - every row from the spam dataset is auto-labelled priority = "Low"
       (spam is never high priority, and this saves you from labelling it)
2. A sample of Enron (ham/internal) emails with an EMPTY `priority` column,
   for YOU to manually label as High / Medium / Low.

Datasets expected (download separately, not fetched by this script):
  - Enron email dataset (Kaggle: "wcukierski/enron-email-dataset")
      -> CSV with columns: file, message   (message = raw RFC822 email text)
  - Spam email dataset (any Kaggle spam/ham CSV, e.g. "Spam Email Classification")
      -> CSV with a text column and a label column (spam/ham or 1/0)

Usage:
    python data_prep.py \
        --enron_csv path/to/enron_emails.csv \
        --spam_csv path/to/spam_emails.csv \
        --sample_size 400
"""

import argparse
import email
import re
import pandas as pd


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def parse_enron_message(raw_message: str) -> dict:
    """Extract subject + body from a raw Enron email string."""
    try:
        msg = email.message_from_string(raw_message)
        subject = msg.get("Subject", "") or ""
        if msg.is_multipart():
            body = ""
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body += payload.decode(errors="ignore")
        else:
            payload = msg.get_payload(decode=True)
            body = payload.decode(errors="ignore") if payload else str(msg.get_payload())
    except Exception:
        subject, body = "", raw_message

    return {"subject": _clean_text(subject), "body": _clean_text(body)}


def load_enron_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "message" not in df.columns:
        raise ValueError(
            f"Expected a 'message' column in {csv_path}, found: {list(df.columns)}"
        )
    parsed = df["message"].apply(parse_enron_message).apply(pd.Series)
    parsed["id"] = range(len(parsed))
    # drop empty/near-empty bodies (attachments-only, forwarded headers, etc.)
    parsed = parsed[parsed["body"].str.len() > 20].reset_index(drop=True)
    return parsed[["id", "subject", "body"]]


def load_spam_dataset(csv_path: str, text_col: str = None, label_col: str = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="latin-1")

    # case-insensitive lookup: map lowercased column name -> actual column name
    col_lookup = {c.lower(): c for c in df.columns}

    # try to auto-detect common column layouts if not given
    if text_col is None:
        for candidate in ["text", "message", "v2", "email", "body", "email_text", "content"]:
            if candidate in col_lookup:
                text_col = col_lookup[candidate]
                break
    if label_col is None:
        for candidate in ["label", "category", "v1", "spam", "class", "is_spam"]:
            if candidate in col_lookup:
                label_col = col_lookup[candidate]
                break
    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not auto-detect text/label columns in {csv_path}. "
            f"Columns found: {list(df.columns)}. Pass text_col=/label_col= explicitly."
        )

    print(f"  spam dataset: using text column '{text_col}', label column '{label_col}'")
    print(f"  label value counts:\n{df[label_col].value_counts().head(10)}")

    out = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    out["text"] = out["text"].astype(str).apply(_clean_text)

    # normalize label to boolean is_spam - handles "spam"/"ham" strings, "1"/"0" strings,
    # AND numeric 1/0 (including 1.0 floats, which happens when the column has NaNs)
    label_str = out["label"].astype(str).str.strip().str.lower()
    label_num = pd.to_numeric(out["label"], errors="coerce")
    out["is_spam"] = label_str.isin(["spam", "1", "true", "yes"]) | (label_num == 1)

    n_spam = int(out["is_spam"].sum())
    if n_spam == 0:
        print(
            f"  WARNING: 0 spam rows detected. Check the label values printed above - "
            f"if spam is marked some other way (e.g. a different number or word), "
            f"pass label_col= explicitly or tell me the actual values."
        )
    return out[out["text"].str.len() > 5][["text", "is_spam"]].reset_index(drop=True)


def build_priority_datasets(enron_csv: str, spam_csv: str, sample_size: int, out_dir: str = "."):
    print("Loading Enron dataset...")
    enron_df = load_enron_dataset(enron_csv)
    print(f"  {len(enron_df)} usable Enron emails parsed")

    print("Loading spam dataset...")
    spam_df = load_spam_dataset(spam_csv)
    spam_only = spam_df[spam_df["is_spam"]].copy()
    print(f"  {len(spam_only)} spam emails found -> auto-labelled 'Low'")

    # --- Base (auto-labelled) training data: spam = Low priority ---
    base = pd.DataFrame({
        "text": spam_only["text"],
        "priority": "Low",
    })
    base_path = f"{out_dir}/priority_base_labelled.csv"
    base.to_csv(base_path, index=False)
    print(f"Saved auto-labelled base set -> {base_path} ({len(base)} rows)")

    # --- Sample for manual labelling: Enron (ham) emails ---
    sample_n = min(sample_size, len(enron_df))
    to_label = enron_df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    to_label["priority"] = ""  # you fill this in: High / Medium / Low
    label_path = f"{out_dir}/priority_labelling_subset.csv"
    to_label.to_csv(label_path, index=False)
    print(f"Saved subset for manual labelling -> {label_path} ({len(to_label)} rows)")
    print("\nNext step: run `python auto_label_priority.py` to pre-label this subset, "
          "then `python label_priority.py` to spot-check the low-confidence ones.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enron_csv", required=True)
    parser.add_argument("--spam_csv", required=True)
    parser.add_argument("--sample_size", type=int, default=400)
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    build_priority_datasets(args.enron_csv, args.spam_csv, args.sample_size, args.out_dir)