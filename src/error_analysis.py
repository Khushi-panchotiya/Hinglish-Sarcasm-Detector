"""Sample ~30 wrong predictions for manual categorisation.

    python -m src.error_analysis --model tfidf_lr
Fill the 'error_type' column by hand, e.g. no-context, mixed-language, hate/abuse,
emoji-only, label-noise, subtle-irony.
"""
import argparse

import pandas as pd

from src.config import REPORTS, SEED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tfidf_lr")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    errs = pd.read_csv(REPORTS / f"errors_{a.model}.csv")
    sample = errs.sample(n=min(a.n, len(errs)), random_state=SEED)
    sample["error_type"] = ""
    sample["notes"] = ""
    out = REPORTS / f"error_analysis_{a.model}.csv"
    sample.to_csv(out, index=False)
    print(f"{len(errs)} errors total; wrote {len(sample)} to {out}")


if __name__ == "__main__":
    main()
