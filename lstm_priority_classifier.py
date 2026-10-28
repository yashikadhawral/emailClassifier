"""
lstm_priority_classifier.py

From-scratch LSTM/GRU priority classifier (Low/Medium/High), embedding layer
initialized from Phase 1's Word2Vec vectors. This is the DL Module 5 / NLP
Module 5 deliverable -- compared against the existing DistilBERT classifier
on the same data/split for the ablation table.

Uses the SAME 3-class scheme as priority_classifier.py / train_priority_classifier.py
(Low/Medium/High) -- not a separate 4-class scheme -- so the comparison table
in Phase 2's training script is apples-to-apples.

Stays a plain importable module (like word2vec_embeddings.py) so pipeline.py
can eventually import LSTMPriorityClassifier directly without pulling in the
DistilBERT training script's heavier dependencies. Training is driven from
training/train_lstm_priority.py.

Includes an optional attention-pooling head (use_attention=True) as an
ablation against the default last-hidden-state pooling.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from word2vec_embeddings import tokenize

LABEL2ID = {"Low": 0, "Medium": 1, "High": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


def build_vocab(texts, min_freq: int = 2, max_vocab_size: int = None) -> dict:
    """Builds word->index vocab from training texts only (never touch val/test
    text here -- avoids vocab leakage). Index 0/1 reserved for pad/unk."""
    from collections import Counter

    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    items = [(w, c) for w, c in counter.items() if c >= min_freq]
    items.sort(key=lambda x: -x[1])  # most frequent first
    if max_vocab_size:
        items = items[: max_vocab_size - 2]

    for word, _ in items:
        vocab[word] = len(vocab)

    print(f"vocab size: {len(vocab)} (min_freq={min_freq})")
    return vocab


def encode(text: str, vocab: dict, max_len: int) -> list[int]:
    tokens = tokenize(text)[:max_len]
    ids = [vocab.get(t, UNK_IDX) for t in tokens]
    return ids


class PriorityLSTMDataset(Dataset):
    def __init__(self, texts, labels, vocab: dict, max_len: int = 200):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = encode(self.texts[idx], self.vocab, self.max_len)
        if not ids:
            ids = [UNK_IDX]
        return torch.tensor(ids, dtype=torch.long), self.labels[idx]


def collate_batch(batch):
    """Pads a batch of variable-length sequences. Returns (padded_ids, lengths, labels)
    -- lengths are needed for pack_padded_sequence so the LSTM ignores pad positions."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


class AttentionPool(nn.Module):
    """Scores each timestep's RNN output, softmaxes over the non-padded
    positions only, and returns the weighted sum -- an alternative to
    taking just the last hidden state."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.score = nn.Linear(in_dim, 1)

    def forward(self, outputs, lengths):
        # outputs: [batch, seq_len, in_dim], lengths: [batch]
        scores = self.score(outputs).squeeze(-1)
        positions = torch.arange(outputs.size(1), device=outputs.device).unsqueeze(0)
        mask = positions < lengths.to(outputs.device).unsqueeze(1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)
        return pooled


class LSTMPriorityClassifier(nn.Module):
    def __init__(
        self,
        embedding_matrix,  # numpy array [vocab_size, embed_dim] from get_embedding_matrix()
        hidden_dim: int = 128,
        num_classes: int = 3,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        rnn_type: str = "lstm",  # "lstm" or "gru"
        freeze_embeddings: bool = False,
        use_attention: bool = False,  # False = last hidden state (original behaviour)
    ):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = not freeze_embeddings

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.use_attention = use_attention
        self.attention = AttentionPool(rnn_out_dim) if use_attention else None
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_out_dim, num_classes)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)  # [batch, seq_len, embed_dim]

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        if isinstance(self.rnn, nn.LSTM):
            packed_out, (h_n, _) = self.rnn(packed)
        else:
            packed_out, h_n = self.rnn(packed)

        if self.use_attention:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=input_ids.size(1)
            )
            final = self.attention(outputs, lengths)
        elif self.rnn.bidirectional:
            # h_n: [num_layers * num_directions, batch, hidden_dim] -- last layer.
            final = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            final = h_n[-1]

        final = self.dropout(final)
        return self.fc(final)  # logits [batch, num_classes]