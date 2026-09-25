"""
training/train_autoencoder.py

Trains EmailAutoencoder on Word2Vec-averaged email vectors. Same data/split
as train_lstm_priority.py (duplicated helpers, same convention as
train_cnn_priority.py).

    python training/train_autoencoder.py \
        --base_csv data/processed/priority_base_labelled.csv \
        --manual_csv data/processed/priority_auto_labelled.csv \
        --word2vec_model {ART}/word2vec_enron.model \
        --output_dir {ART}/autoencoder
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoencoder import EmailAutoencoder, average_embedding_vectors  # noqa: E402
from lstm_priority_classifier import LABEL2ID, ID2LABEL, build_vocab  # noqa: E402
from word2vec_embeddings import train_word2vec, get_embedding_matrix  # noqa: E402


def load_combined_data(base_csv, manual_csv, max_low_from_spam):
    base = pd.read_csv(base_csv)
    if len(base) > max_low_from_spam:
        base = base.sample(n=max_low_from_spam, random_state=42).reset_index(drop=True)
    manual = pd.read_csv(manual_csv)
    manual = manual[manual["priority"].notna() & (manual["priority"] != "")]
    manual["text"] = manual["subject"].fillna("") + ". " + manual["body"].fillna("")
    manual = manual[["text", "priority"]]
    combined = pd.concat([base, manual], ignore_index=True)
    combined = combined[combined["priority"].isin(LABEL2ID.keys())]
    return combined.dropna(subset=["text"])


def get_or_train_word2vec(args):
    if args.word2vec_model:
        return args.word2vec_model
    out_path = args.word2vec_out or "saved_models/word2vec_enron.model"
    train_word2vec(args.enron_csv, sg=args.word2vec_sg, out_path=out_path,
                    epochs=args.word2vec_epochs, workers=args.workers)
    return out_path


def main(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    df = load_combined_data(args.base_csv, args.manual_csv, args.max_low_from_spam)
    df["label_id"] = df["priority"].map(LABEL2ID)
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label_id"])

    vocab = build_vocab(train_df["text"], min_freq=args.min_freq, max_vocab_size=args.max_vocab_size)
    embedding_matrix = get_embedding_matrix(vocab, get_or_train_word2vec(args))

    x_train = average_embedding_vectors(train_df["text"].tolist(), vocab, embedding_matrix, args.max_len)
    x_val = average_embedding_vectors(val_df["text"].tolist(), vocab, embedding_matrix, args.max_len)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val)), batch_size=args.batch_size)

    model = EmailAutoencoder(x_train.shape[1], args.hidden_dim, args.latent_dim, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt = os.path.join(args.output_dir, "model.pt")
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, _ = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0); n += x.size(0)
        train_loss = total / n

        model.eval()
        with torch.no_grad():
            vt, vn = 0.0, 0
            for (x,) in val_loader:
                x = x.to(device)
                recon, _ = model(x)
                vt += ((recon - x) ** 2).mean(dim=1).sum().item(); vn += x.size(0)
            val_loss = vt / vn
        print(f"epoch {epoch}: train_mse={train_loss:.6f} val_mse={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    errors = model.reconstruction_error(torch.from_numpy(x_val).to(device)).cpu().numpy()
    per_class = {}
    for lid, name in ID2LABEL.items():
        mask = val_df["label_id"].values == lid
        if mask.any():
            per_class[name] = float(errors[mask].mean())

    metrics = {"model": "email_autoencoder", "final_val_mse": best_val, "params": n_params,
               "latent_dim": args.latent_dim, "per_class_recon_error": per_class}
    with open(os.path.join(args.output_dir, "reconstruction_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


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
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default="saved_models/autoencoder")
    main(parser.parse_args())