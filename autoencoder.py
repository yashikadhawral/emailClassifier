"""
autoencoder.py

Small fully-connected autoencoder over email embeddings (DL Module 3).
Input = mean of each email's Word2Vec token vectors. Unsupervised --
reconstruction error is used afterward as an anomaly-style signal.
"""

import numpy as np
import torch
import torch.nn as nn

from lstm_priority_classifier import PAD_IDX, UNK_IDX, encode


def average_embedding_vectors(texts, vocab: dict, embedding_matrix: np.ndarray, max_len: int = 200) -> np.ndarray:
    embed_dim = embedding_matrix.shape[1]
    out = np.zeros((len(texts), embed_dim), dtype=np.float32)
    for i, text in enumerate(texts):
        ids = [t for t in encode(text, vocab, max_len) if t not in (PAD_IDX, UNK_IDX)]
        if ids:
            out[i] = embedding_matrix[ids].mean(axis=0)
    return out


class EmailAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 100, hidden_dim: int = 64, latent_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    @torch.no_grad()
    def reconstruction_error(self, x):
        recon, _ = self.forward(x)
        return ((recon - x) ** 2).mean(dim=1)