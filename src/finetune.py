"""Fine-tune a transformer on the project splits.

Run on a GPU (Colab/Kaggle):
    python -m src.finetune --model muril
    python -m src.finetune --model xlmr
    python -m src.finetune --model hingbert
Outputs: models/<name>/ (weights) and reports/metrics_<name>.json
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback, Trainer,
                          TrainingArguments, set_seed)

from src.config import ID2LABEL, LABEL2ID, MODELS, PROCESSED, SEED, TRANSFORMERS
from src.metrics import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=TRANSFORMERS, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    set_seed(SEED)
    ckpt = TRANSFORMERS[args.model]
    tok = AutoTokenizer.from_pretrained(ckpt)

    def load(split):
        df = pd.read_csv(PROCESSED / f"{split}.csv")
        ds = Dataset.from_dict({"text": df["text"].tolist(),
                                "labels": df["label"].map(LABEL2ID).tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                      batched=True), df

    train_ds, _ = load("train")
    val_ds, _ = load("validation")
    test_ds, test_df = load("test")

    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt, num_labels=2, label2id=LABEL2ID, id2label=ID2LABEL)

    def compute(p):
        preds = np.argmax(p.predictions, axis=-1)
        return {"accuracy": accuracy_score(p.label_ids, preds),
                "macro_f1": f1_score(p.label_ids, preds, average="macro")}

    out = MODELS / args.model
    steps_per_epoch = -(-len(train_ds) // args.batch)
    warmup_steps = int(0.1 * steps_per_epoch * args.epochs)  # same effect as warmup_ratio=0.1
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=args.epochs,
        learning_rate=args.lr, per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=64, weight_decay=0.01, warmup_steps=warmup_steps,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", save_total_limit=1, seed=SEED,
        fp16=torch.cuda.is_available(), report_to="none")
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])

    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60

    logits = trainer.predict(test_ds).predictions
    preds = [ID2LABEL[int(i)] for i in np.argmax(logits, axis=-1)]
    res = evaluate(test_df["label"], preds, args.model)
    print(f"train minutes: {minutes:.1f}  checkpoint: {ckpt}")

    trainer.save_model(str(out / "final"))
    tok.save_pretrained(str(out / "final"))

    errs = test_df[["raw_text", "text", "label"]].copy()
    errs["prediction"] = preds
    errs[errs.label != errs.prediction].to_csv(
        MODELS.parent / "reports" / f"errors_{args.model}.csv", index=False)
    return res


if __name__ == "__main__":
    main()
