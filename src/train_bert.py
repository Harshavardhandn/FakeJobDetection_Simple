import argparse
import json
from pathlib import Path

import numpy as np
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)

from src.data_utils import load_csv

ROOT = Path(__file__).resolve().parents[1]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=str(ROOT / "data/raw/fake_job_postings.csv")
    )

    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased"
    )

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--sample-size", type=int, default=0)

    a = parser.parse_args()

    set_seed(42)

    # 1. Load dataset
    print("\nLoading dataset...")

    df = load_csv(a.data, True)

    print("Total records:", len(df))

    # 2. Quick-run sampling
    if a.sample_size and len(df) > a.sample_size:

        print(f"Using {a.sample_size} records for quick training.")

        df, _ = train_test_split(
            df,
            train_size=a.sample_size,
            random_state=42,
            stratify=df.fraudulent
        )

    # 3. Train-test split
    tr, te = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df.fraudulent
    )

    print("Training records:", len(tr))
    print("Testing records:", len(te))

    # 4. Load tokenizer and model
    print("\nLoading DistilBERT...")

    tok = AutoTokenizer.from_pretrained(a.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        a.model_name,
        num_labels=2
    )

    # 5. Tokenization
    def make(frame):

        ds = Dataset.from_pandas(
            frame[["text", "fraudulent"]].rename(
                columns={"fraudulent": "labels"}
            ),
            preserve_index=False
        )

        return ds.map(
            lambda batch: tok(
                batch["text"],
                truncation=True,
                max_length=a.max_length
            ),
            batched=True
        )

    print("\nTokenizing datasets...")

    train_ds = make(tr)
    test_ds = make(te)

    # 6. Evaluation metrics
    def calc(eval_pred):

        logits, labels = eval_pred

        pred = np.argmax(logits, axis=1)

        pr, re, f1, _ = precision_recall_fscore_support(
            labels,
            pred,
            average="binary",
            zero_division=0
        )

        return {
            "accuracy": accuracy_score(labels, pred),
            "precision": pr,
            "recall": re,
            "f1": f1
        }

    # 7. Training configuration
    args = TrainingArguments(
        output_dir=str(ROOT / "results/bert_checkpoints"),

        num_train_epochs=a.epochs,

        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,

        eval_strategy="epoch",
        save_strategy="no",

        learning_rate=2e-5,
        weight_decay=0.01,

        logging_steps=50,
        report_to="none",

        fp16=torch.cuda.is_available()
    )

    # 8. Data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tok
    )

    # 9. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,

        # Correct variable name: tok
        processing_class=tok,

        data_collator=data_collator,
        compute_metrics=calc
    )

    # 10. Train model
    print("\nStarting DistilBERT training...\n")

    trainer.train()

    # 11. Evaluate model
    print("\nEvaluating DistilBERT...")

    result = trainer.predict(test_ds)

    pred = np.argmax(result.predictions, axis=1)

    y = np.array(te.fraudulent)

    pr, re, f1, _ = precision_recall_fscore_support(
        y,
        pred,
        average="binary",
        zero_division=0
    )

    report = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(pr),
        "recall": float(re),
        "f1": float(f1),

        "confusion_matrix": confusion_matrix(
            y,
            pred
        ).tolist(),

        "classification_report": classification_report(
            y,
            pred,
            output_dict=True,
            zero_division=0
        ),

        "model_name": a.model_name,
        "epochs": a.epochs,
        "test_rows": len(te),
        "train_rows": len(tr)
    }

    # 12. Save model
    out = ROOT / "models/distilbert"

    out.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(out))
    tok.save_pretrained(str(out))

    # 13. Save evaluation results
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "bert_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\n================================")
    print("DISTILBERT TRAINING COMPLETED")
    print("================================")

    print("Accuracy :", report["accuracy"])
    print("Precision:", report["precision"])
    print("Recall   :", report["recall"])
    print("F1 Score :", report["f1"])

    print("\nModel saved at:", out)
    print("Metrics saved at:", results_dir / "bert_metrics.json")


if __name__ == "__main__":
    main()
