"""
training/train_cnn_priority.py

CNN priority baseline (DL Module 5 ablation) -- model + training in one file.
Reuses vocab/dataset/embeddings from lstm_priority_classifier.py + word2vec_embeddings.py
so results are directly comparable to the LSTM run.

    python training/train_cnn_priority.py \
        --base_csv data/processed/priority_base_labelled.csv \
        --manual_csv data/processed/priority_auto_labelled.csv \
        --word2vec_model {ART}/word2vec_enron.model \
        --output_dir {ART}/cnn_priority
"""
import argparse
import json
import os
import sys
import time
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lstm_priority_classifier import (  # noqa: E402
    LABEL2ID, PAD_IDX, build_vocab, PriorityLSTMDataset, collate_batch,
)
from word2vec_embeddings import train_word2vec, get_embedding_matrix  # noqa: E402


# ---------------- model ----------------

class CNNPriorityClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_filters=100, filter_sizes=(2, 3, 4),
                 num_classes=3, dropout=0.3, freeze_embeddings=False):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, input_ids, lengths=None):  # lengths unused, kept for call-signature parity
        embedded = self.embedding(input_ids).permute(0, 2, 1)  # [batch, embed_dim, seq_len]
        pooled = []
        for conv in self.convs:
            if embedded.size(2) < conv.kernel_size[0]:
                embedded_in = F.pad(embedded, (0, conv.kernel_size[0] - embedded.size(2)))
            else:
                embedded_in = embedded
            conv_out = F.relu(conv(embedded_in))
            pooled.append(F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2))
        features = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(features)


# ---------------- data (identical to train_lstm_priority.py's loader) ----------------

def load_combined_data(base_csv, manual_csv, max_low_from_spam):
    base = pd.read_csv(base_csv)
    if len(base) > max_low_from_spam:
        print(f"Capping spam-derived Low rows: {len(base)} -> {max_low_from_spam}")
        base = base.sample(n=max_low_from_spam, random_state=42).reset_index(drop=True)

    manual = pd.read_csv(manual_csv)
    manual = manual[manual["priority"].notna() & (manual["priority"] != "")]
    manual["text"] = manual["subject"].fillna("") + ". " + manual["body"].fillna("")
    manual = manual[["text", "priority"]]

    combined = pd.concat([base, manual], ignore_index=True)
    combined = combined[combined["priority"].isin(LABEL2ID.keys())]
    combined = combined.dropna(subset=["text"])
    print("Final class distribution used for training:\n", combined["priority"].value_counts())
    return combined


def get_or_train_word2vec(args):
    if args.word2vec_model:
        print(f"using existing word2vec model -> {args.word2vec_model}")
        return args.word2vec_model
    if not args.enron_csv:
        raise ValueError("Pass either --word2vec_model or --enron_csv.")
    out_path = args.word2vec_out or "saved_models/word2vec_enron.model"
    train_word2vec(args.enron_csv, sg=args.word2vec_sg, out_path=out_path,
                    epochs=args.word2vec_epochs, workers=args.workers)
    return out_path


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_time, n_examples = 0.0, 0
    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(device)
        start = time.perf_counter()
        logits = model(input_ids, lengths)
        total_time += time.perf_counter() - start
        n_examples += input_ids.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    ms_per_example = (total_time / max(n_examples, 1)) * 1000
    return acc, f1_macro, ms_per_example


# ---------------- main ----------------

def main(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    df = load_combined_data(args.base_csv, args.manual_csv, args.max_low_from_spam)
    df["label_id"] = df["priority"].map(LABEL2ID)

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label_id"])

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

    filter_sizes = tuple(int(x) for x in args.filter_sizes.split(","))
    model = CNNPriorityClassifier(
        embedding_matrix, num_filters=args.num_filters, filter_sizes=filter_sizes,
        num_classes=3, dropout=args.dropout,
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc, val_f1, _ = evaluate(model, val_loader, device)
        elapsed = time.time() - epoch_start
        print(f"epoch {epoch}: loss={avg_loss:.4f} val_acc={val_acc:.4f} "
              f"val_f1_macro={val_f1:.4f} ({elapsed:.1f}s)")

        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))
    final_acc, final_f1, ms_per_example = evaluate(model, val_loader, device)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    cnn_metrics = {
        "model": "cnn_priority", "accuracy": final_acc, "f1_macro": final_f1,
        "params": n_params, "inference_ms_per_example": ms_per_example,
    }
    print("\n--- CNN final metrics ---")
    print(json.dumps(cnn_metrics, indent=2))
    with open(os.path.join(args.output_dir, "cnn_metrics.json"), "w") as f:
        json.dump(cnn_metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_csv", default="data/processed/priority_base_labelled.csv")
    parser.add_argument("--manual_csv", default="data/processed/priority_auto_labelled.csv")
    parser.add_argument("--max_low_from_spam", type=int, default=500)
    parser.add_argument("--word2vec_model", default=None)
    parser.add_argument("--enron_csv", default=None)
    parser.add_argument("--word2vec_out", default=None)
    parser.add_argument("--word2vec_sg", type=int, default=1)
    parser.add_argument("--word2vec_epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_vocab_size", type=int, default=30000)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--num_filters", type=int, default=100)
    parser.add_argument("--filter_sizes", default="2,3,4")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default="saved_models/cnn_priority")

    args = parser.parse_args()
    main(args)