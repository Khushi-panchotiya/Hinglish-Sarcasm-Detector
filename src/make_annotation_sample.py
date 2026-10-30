"""Create the 200-row double-annotation sheet and compute Cohen's kappa once filled.

    python -m src.make_annotation_sample          # create template
    python -m src.make_annotation_sample --kappa  # after both annotators finish
"""
import argparse

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.config import LABELS, PROCESSED, REPORTS, SEED

TEMPLATE = REPORTS / "double_annotation_template.csv"
DONE = REPORTS / "double_annotation_completed.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    if not a.kappa:
        df = pd.read_csv(PROCESSED / "cleaned.csv").sample(n=a.n, random_state=SEED)
        out = df[["raw_text"]].rename(columns={"raw_text": "comment"})
        out.insert(0, "sample_id", [f"A{i:04d}" for i in range(1, len(out) + 1)])
        out["khushi"] = ""
        out["grishma"] = ""
        out.to_csv(TEMPLATE, index=False)
        print("wrote", TEMPLATE, "- each person fills their own column with Sarcastic/Sincere")
        return
    d = pd.read_csv(DONE).dropna(subset=["khushi", "grishma"])
    d = d[d.khushi.isin(LABELS) & d.grishma.isin(LABELS)]
    print("items:", len(d), "| agreement:", round((d.khushi == d.grishma).mean(), 3),
          "| Cohen's kappa:", round(cohen_kappa_score(d.khushi, d.grishma, labels=LABELS), 3))
    d[d.khushi != d.grishma].to_csv(REPORTS / "annotation_disagreements.csv", index=False)


if __name__ == "__main__":
    main()
