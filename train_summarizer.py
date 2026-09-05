"""
train_summarizer.py

Fine-tunes a small seq2seq model (T5-small by default) on AESLC
(Annotated Enron Subject Line Corpus) for email summarization.

NOTE / honest caveat: AESLC pairs an email body with its ORIGINAL SUBJECT
LINE, so the model learns to produce short, headline-style summaries
(a handful of words) rather than multi-sentence paragraph summaries.
That's genuinely useful for a "quick gist" feature in your app, but if you
want longer abstractive summaries, either:
  (a) blend in a general-purpose summarization dataset (e.g. CNN/DailyMail
      or SAMSum) alongside AESLC, or
  (b) skip fine-tuning for this component and use a pretrained
      "facebook/bart-large-cnn" model as-is.
The code below does the AESLC fine-tune as planned; swap the dataset name
in main() if you want to try option (a).

Usage:
    python train_summarizer.py --output_dir saved_models/summarizer
"""

import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from rouge_score import rouge_scorer

_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def preprocess_batch(examples, tokenizer, max_input_len=384, max_target_len=32):
    inputs = ["summarize: " + doc for doc in examples["email_body"]]
    model_inputs = tokenizer(inputs, max_length=max_input_len, truncation=True)

    labels = tokenizer(
        text_target=examples["subject_line"], max_length=max_target_len, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def make_compute_metrics(tokenizer):
    """Closure so compute_metrics has access to tokenizer without globals,
    and doesn't depend on the HF `evaluate` Hub-script loader (which was
    failing on network-flaky machines)."""

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(decoded_preds, decoded_labels):
            result = _SCORER.score(ref, pred)
            for k in scores:
                scores[k].append(result[k].fmeasure)

        return {k: round(float(np.mean(v)), 4) for k, v in scores.items()}

    return compute_metrics


def main(args):
    print("Loading AESLC dataset (Enron subject-line corpus)...")
    dataset = load_dataset("Yale-LILY/aeslc")

    # sanity check on expected column names; datasets occasionally rename these
    cols = dataset["train"].column_names
    assert "email_body" in cols and "subject_line" in cols, (
        f"Expected 'email_body' / 'subject_line' columns, found {cols}. "
        "Check the current AESLC schema and adjust preprocess_batch()."
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    tokenized = dataset.map(
        lambda ex: preprocess_batch(ex, tokenizer),
        batched=True,
        remove_columns=cols,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    compute_metrics = make_compute_metrics(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir + "_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-4 if "t5" in args.base_model else 2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        logging_steps=50,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Final eval:", trainer.evaluate())

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved summarizer -> {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="t5-small")
    parser.add_argument("--output_dir", default="saved_models/summarizer")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    main(args)