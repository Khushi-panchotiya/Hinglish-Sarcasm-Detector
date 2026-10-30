import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)

from src.config import LABELS, REPORTS


def evaluate(y_true, y_pred, name: str) -> dict:
    """Print metrics, save confusion matrix + json under reports/, return summary."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(values_format="d")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(REPORTS / f"confusion_{name}.png", dpi=150)
    plt.close()
    res = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro")),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    (REPORTS / f"metrics_{name}.json").write_text(json.dumps(res, indent=2))
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
    return res
