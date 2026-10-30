"""Clean the raw CSV, remove duplicates/conflicts, and write stratified splits."""
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import LABEL_MAP, PROCESSED, RAW_CSV, REPORTS, SEED
from src.preprocess import dedup_key, normalize


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    df = pd.DataFrame({"raw_text": raw["comment"].fillna("").astype(str)})
    df["label"] = raw["human_labelled"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    df["emoji_present"] = raw["emoji_present"]
    df["hindi_pct"] = raw["hindi_pct"]
    df["key"] = df["raw_text"].map(dedup_key)

    n_raw = len(df)
    df = df[df["label"].notna() & df["key"].ne("")]
    n_valid = len(df)

    nl = df.groupby("key")["label"].nunique()
    conflicts = nl[nl > 1].index
    df = df[~df["key"].isin(conflicts)].drop_duplicates("key").reset_index(drop=True)
    df["text"] = df["raw_text"].map(normalize)

    train, tmp = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
    val, test = train_test_split(tmp, test_size=0.5, random_state=SEED, stratify=tmp["label"])
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        part.to_csv(PROCESSED / f"{name}.csv", index=False)
    df.to_csv(PROCESSED / "cleaned.csv", index=False)

    audit = {
        "raw_rows": n_raw,
        "invalid_label_rows": n_raw - n_valid,
        "conflicting_duplicate_texts": int(len(conflicts)),
        "final_rows": len(df),
        "class_counts": df["label"].value_counts().to_dict(),
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
    }
    (REPORTS / "data_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
