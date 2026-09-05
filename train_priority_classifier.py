"""
train_priority_classifier.py

Fine-tunes DistilBERT as a 3-class priority classifier (Low / Medium / High)
on:
  - priority_base_labelled.csv        (spam -> Low, from data_prep.py)
  - priority_auto_labelled.csv        (Enron subset: zero-shot labelled,
                                        spot-checked/corrected by you via
                                        auto_label_priority.py + label_priority.py)

The spam file usually has WAY more rows than the Enron subset (tens of
thousands vs a few hundred), which would make "Low" ~99% of the training
data and produce a classifier that just predicts Low for everything. Two
fixes applied here:
  1. --max_low_from_spam caps how many spam rows get used, so Low doesn't
     swamp the other two classes.
  2. Class-weighted loss (WeightedTrainer below) handles whatever
     imbalance remains, especially the likely-small High class.

Run this on Colab / a GPU machine, not on a laptop CPU.

Usage:
    python train_priority_classifier.py \
        --base_csv priority_base_labelled.csv \
        --manual_csv priority_auto_labelled.csv \
        --max_low_from_spam 500 \
        --output_dir saved_models/priority_classifier
"""
import os
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

LABEL2ID = {"Low": 0, "Medium": 1, "High": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class PriorityDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy, to counter whatever
    class imbalance remains after capping the spam-derived Low rows."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def load_combined_data(base_csv: str, manual_csv: str, max_low_from_spam: int) -> pd.DataFrame:
    base = pd.read_csv(base_csv)  # columns: text, priority
    if len(base) > max_low_from_spam:
        print(f"Capping spam-derived Low rows: {len(base)} -> {max_low_from_spam}")
        base = base.sample(n=max_low_from_spam, random_state=42).reset_index(drop=True)

    manual = pd.read_csv(manual_csv)  # columns: id, subject, body, priority (+ predicted_priority, confidence)
    manual = manual[manual["priority"].notna() & (manual["priority"] != "")]
    manual["text"] = (manual["subject"].fillna("") + ". " + manual["body"].fillna(""))
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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main(args):
    df = load_combined_data(args.base_csv, args.manual_csv, args.max_low_from_spam)
    df["label_id"] = df["priority"].map(LABEL2ID)

    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label_id"]
    )

    # class weights from the TRAIN split only (avoid peeking at val distribution)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=train_df["label_id"].values,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"Class weights (Low/Medium/High): {class_weights.tolist()}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(texts):
        return tokenizer(
            list(texts), truncation=True, padding=True, max_length=256
        )

    train_encodings = tokenize(train_df["text"])
    val_encodings = tokenize(val_df["text"])

    train_dataset = PriorityDataset(train_encodings, train_df["label_id"].tolist())
    val_dataset = PriorityDataset(val_encodings, val_df["label_id"].tolist())

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir + "_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=20,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    print("Final eval:", trainer.evaluate())

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved priority classifier -> {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_csv", default="priority_base_labelled.csv")
    parser.add_argument("--manual_csv", default="priority_auto_labelled.csv")
    parser.add_argument("--max_low_from_spam", type=int, default=500)
    parser.add_argument("--output_dir", default="saved_models/priority_classifier")
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    main(args)