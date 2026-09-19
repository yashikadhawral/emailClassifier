"""
training/train_lstm_priority.py

CLI entry point for Phase 2 (DL Module 5 / NLP Module 5). Trains the
from-scratch LSTM/GRU priority classifier on the SAME data + split
train_priority_classifier.py (DistilBERT) uses, so the two are directly
comparable. Mirrors train_priority_classifier.py's structure on purpose.

Run from the repo root, e.g. on Colab with a GPU runtime:

    # if you already have a trained word2vec model:
    python training/train_lstm_priority.py \
        --base_csv data/processed/priority_base_labelled.csv \
        --manual_csv data/processed/priority_auto_labelled.csv \
        --word2vec_model saved_models/word2vec_enron.model \
        --baseline_json baseline_metrics.json

    # OR, to train word2vec inline first (no separate Phase-1 run needed,
    # nothing gets saved to disk except the final artifacts unless you
    # also pass --word2vec_out):
    python training/train_lstm_priority.py \
        --base_csv data/processed/priority_base_labelled.csv \
        --manual_csv data/processed/priority_auto_labelled.csv \
        --enron_csv path/to/emails.csv

Explicit gradient clipping is applied every step (DL Module 5 LO).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# repo root on sys.path -> lstm_priority_classifier.py / word2vec_embeddings.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lstm_priority_classifier import (  # noqa: E402
    LABEL2ID,
    ID2LABEL,
    build_vocab,
    PriorityLSTMDataset,
    collate_batch,
    LSTMPriorityClassifier,
)
from word2vec_embeddings import train_word2vec, get_embedding_matrix  # noqa: E402


def load_combined_data(base_csv: str, manual_csv: str, max_low_from_spam: int):
    """Same logic as training/train_priority_classifier.py's load_combined_data --
    duplicated (not imported) so this script doesn't pull in the transformers/
    DistilBERT dependency chain just to reuse a ~15-line data-loading function.
    Keep these two in sync if the labelling/capping logic ever changes."""
    import pandas as pd

    base = pd.read_csv(base_csv)  # columns: text, priority
    if len(base) > max_low_from_spam:
        print(f"Capping spam-derived Low rows: {len(base)} -> {max_low_from_spam}")
        base = base.sample(n=max_low_from_spam, random_state=42).reset_index(drop=True)

    manual = pd.read_csv(manual_csv)  # columns: id, subject, body, priority (+ predicted_priority, confidence)
    manual = manual[manual["priority"].notna() & (manual["priority"] != "")]
    manual["text"] = manual["subject"].fillna("") + ". " + manual["body"].fillna("")
    manual = manual[["text", "priority"]]

    if len(manual) < 30:
        print(
            f"WARNING: only {len(manual)} finalized rows found in {manual_csv}. "
            "Run auto_label_priority.py then label_priority.py --finalize "
            "to fill in the rest before training."
        )

    combined = pd.concat([base, manual], ignore_index=True)
    combined = combined[combined["priority"].isin(LABEL2ID.keys())]
    combined = combined.dropna(subset=["text"])
    print("Final class distribution used for training:\n", combined["priority"].value_counts())
    return combined


def get_or_train_word2vec(args) -> str:
    if args.word2vec_model:
        print(f"using existing word2vec model -> {args.word2vec_model}")
        return args.word2vec_model

    if not args.enron_csv:
        raise ValueError(
            "Pass either --word2vec_model (path to an already-trained model) "
            "or --enron_csv (to train one inline before the LSTM)."
        )

    print("no --word2vec_model given -- training word2vec inline first...")
    out_path = args.word2vec_out or "saved_models/word2vec_enron.model"
    train_word2vec(
        args.enron_csv,
        sg=args.word2vec_sg,
        out_path=out_path,
        epochs=args.word2vec_epochs,
        workers=args.workers,
    )
    return out_path


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_time = 0.0
    n_examples = 0

    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(device)
        lengths_cpu = lengths  # pack_padded_sequence wants lengths on CPU
        start = time.perf_counter()
        logits = model(input_ids, lengths_cpu)
        total_time += time.perf_counter() - start
        n_examples += input_ids.size(0)

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    ms_per_example = (total_time / max(n_examples, 1)) * 1000
    return acc, f1_macro, ms_per_example


def main(args):
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device}")

    df = load_combined_data(args.base_csv, args.manual_csv, args.max_low_from_spam)
    df["label_id"] = df["priority"].map(LABEL2ID)

    # SAME split as train_priority_classifier.py (same random_state/test_size on
    # the same combined df) -- this is what makes the comparison table fair.
    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label_id"]
    )

    vocab = build_vocab(train_df["text"], min_freq=args.min_freq, max_vocab_size=args.max_vocab_size)

    word2vec_path = get_or_train_word2vec(args)
    embedding_matrix = get_embedding_matrix(vocab, word2vec_path)

    train_dataset = PriorityLSTMDataset(train_df["text"], train_df["label_id"].tolist(), vocab, args.max_len)
    val_dataset = PriorityLSTMDataset(val_df["text"], val_df["label_id"].tolist(), vocab, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1, 2]), y=train_df["label_id"].values
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"class weights (Low/Medium/High): {class_weights.tolist()}")

    model = LSTMPriorityClassifier(
        embedding_matrix,
        hidden_dim=args.hidden_dim,
        num_classes=3,
        num_layers=args.num_layers,
        bidirectional=not args.no_bidirectional,
        dropout=args.dropout,
        rnn_type=args.rnn_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_f1 = -1.0
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0

        for input_ids, lengths, labels in train_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()

            # explicit gradient clipping (DL-5 LO)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc, val_f1, _ = evaluate(model, val_loader, device)
        elapsed = time.time() - epoch_start
        print(
            f"epoch {epoch}: loss={avg_loss:.4f} val_acc={val_acc:.4f} "
            f"val_f1_macro={val_f1:.4f} ({elapsed:.1f}s)"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

    # final eval + inference timing on the best-saved checkpoint
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))
    final_acc, final_f1, ms_per_example = evaluate(model, val_loader, device)

    os.makedirs(args.output_dir, exist_ok=True)
    import pickle
    with open(os.path.join(args.output_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    lstm_metrics = {
        "model": f"{args.rnn_type}_priority",
        "accuracy": final_acc,
        "f1_macro": final_f1,
        "params": n_params,
        "inference_ms_per_example": ms_per_example,
    }
    print("\n--- LSTM final metrics ---")
    print(json.dumps(lstm_metrics, indent=2))

    with open(os.path.join(args.output_dir, "lstm_metrics.json"), "w") as f:
        json.dump(lstm_metrics, f, indent=2)

    if args.baseline_json and os.path.exists(args.baseline_json):
        with open(args.baseline_json) as f:
            baseline = json.load(f)
        print("\n--- comparison: DistilBERT (Phase 0) vs LSTM (Phase 2) ---")
        print(f"{'model':<20}{'accuracy':<12}{'f1_macro':<12}{'params':<15}{'ms/example':<12}")
        print(f"{baseline.get('model', 'distilbert'):<20}{baseline.get('accuracy', 0):<12.4f}"
              f"{baseline.get('f1_macro', 0):<12.4f}{baseline.get('params', 0):<15,}"
              f"{baseline.get('inference_ms_per_example', 0):<12.2f}")
        print(f"{lstm_metrics['model']:<20}{lstm_metrics['accuracy']:<12.4f}"
              f"{lstm_metrics['f1_macro']:<12.4f}{lstm_metrics['params']:<15,}"
              f"{lstm_metrics['inference_ms_per_example']:<12.2f}")
    else:
        print(
            "\nNo --baseline_json found -- run training/train_priority_classifier.py "
            "and save its eval metrics to baseline_metrics.json for the full comparison table."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_csv", default="data/processed/priority_base_labelled.csv")
    parser.add_argument("--manual_csv", default="data/processed/priority_auto_labelled.csv")
    parser.add_argument("--max_low_from_spam", type=int, default=500)

    parser.add_argument("--word2vec_model", default=None, help="path to an already-trained word2vec model")
    parser.add_argument("--enron_csv", default=None, help="train word2vec inline if --word2vec_model not given")
    parser.add_argument("--word2vec_out", default=None, help="where to save the inline-trained word2vec model")
    parser.add_argument("--word2vec_sg", type=int, default=1)
    parser.add_argument("--word2vec_epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_vocab_size", type=int, default=30000)
    parser.add_argument("--max_len", type=int, default=200)

    parser.add_argument("--rnn_type", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--no_bidirectional", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--device", default=None, help="cuda / cpu, defaults to auto-detect")

    parser.add_argument("--output_dir", default="saved_models/lstm_priority")
    parser.add_argument("--baseline_json", default="baseline_metrics.json")

    args = parser.parse_args()
    main(args)