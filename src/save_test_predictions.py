"""Save each model's test-set predictions to reports/test_predictions.csv.

Lets the report notebook show full metrics without needing the large model weights.
Run locally after training:  python -m src.save_test_predictions
"""
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import ID2LABEL, MODELS, PROCESSED, REPORTS


@torch.no_grad()
def predict_hf(name, texts, bs=64):
    path = MODELS / name / "final"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=128)
        out.append(model(**enc).logits.argmax(-1).numpy())
    return [ID2LABEL[int(i)] for i in np.concatenate(out)]


def main():
    test = pd.read_csv(PROCESSED / "test.csv")
    out = test[["raw_text", "text", "label"]].copy()
    out["tfidf_lr"] = joblib.load(MODELS / "tfidf_lr.joblib").predict(test["text"])
    for name in ("muril", "xlmr", "hingbert"):
        if (MODELS / name / "final").exists():
            out[name] = predict_hf(name, test["text"].tolist())
    out.to_csv(REPORTS / "test_predictions.csv", index=False)
    print("saved", REPORTS / "test_predictions.csv", list(out.columns))


if __name__ == "__main__":
    main()
