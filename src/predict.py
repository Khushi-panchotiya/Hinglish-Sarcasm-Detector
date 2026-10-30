"""Unified predictor: TF-IDF baseline or a fine-tuned transformer folder."""
from pathlib import Path
from typing import List

import joblib
import numpy as np

from src.config import LABELS, MODELS
from src.preprocess import normalize


class Predictor:
    """predict_proba(list[str]) -> array (n, 2) ordered as config.LABELS."""

    def __init__(self, name: str = "tfidf_lr"):
        self.name = name
        self._pipe = None
        self._tok = self._model = None
        if name == "tfidf_lr":
            self._pipe = joblib.load(MODELS / "tfidf_lr.joblib")
        else:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            path = MODELS / name / "final"
            self._tok = AutoTokenizer.from_pretrained(path)
            self._model = AutoModelForSequenceClassification.from_pretrained(path).eval()
            self._torch = torch

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        texts = [normalize(t) for t in texts]
        if self._pipe is not None:
            probs = self._pipe.predict_proba(texts)
            order = [list(self._pipe.classes_).index(l) for l in LABELS]
            return probs[:, order]
        out = []
        with self._torch.no_grad():
            for i in range(0, len(texts), 32):
                enc = self._tok(texts[i:i + 32], return_tensors="pt", padding=True,
                                truncation=True, max_length=128)
                out.append(self._torch.softmax(self._model(**enc).logits, -1).numpy())
        return np.vstack(out)

    def predict(self, text: str) -> dict:
        p = self.predict_proba([text])[0]
        i = int(p.argmax())
        return {"label": LABELS[i], "confidence": float(p[i]),
                "probs": dict(zip(LABELS, map(float, p)))}


def available_models() -> List[str]:
    names = ["tfidf_lr"] if (MODELS / "tfidf_lr.joblib").exists() else []
    names += sorted(p.parent.name for p in Path(MODELS).glob("*/final") if p.is_dir())
    return names
