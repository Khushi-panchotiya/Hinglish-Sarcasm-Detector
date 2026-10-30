"""TF-IDF (word + char) + Logistic Regression baseline, plus majority-class reference."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.config import MODELS, PROCESSED, REPORTS, SEED
from src.metrics import evaluate


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                     sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                     sublinear_tf=True)),
        ])),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced",
                                          C=2.0, random_state=SEED)),
    ])


def main():
    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "validation.csv")
    test = pd.read_csv(PROCESSED / "test.csv")

    maj = train["label"].mode()[0]
    evaluate(test["label"], [maj] * len(test), "majority_baseline")

    pipe = build_pipeline().fit(train["text"], train["label"])
    evaluate(val["label"], pipe.predict(val["text"]), "tfidf_lr_validation")
    res = evaluate(test["label"], pipe.predict(test["text"]), "tfidf_lr")

    MODELS.mkdir(exist_ok=True)
    joblib.dump(pipe, MODELS / "tfidf_lr.joblib")

    errs = test[["raw_text", "text", "label"]].copy()
    errs["prediction"] = pipe.predict(test["text"])
    errs["confidence"] = pipe.predict_proba(test["text"]).max(axis=1)
    errs[errs.label != errs.prediction].to_csv(REPORTS / "errors_tfidf_lr.csv", index=False)
    return res


if __name__ == "__main__":
    main()
