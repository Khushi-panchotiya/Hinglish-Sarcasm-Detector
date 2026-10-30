"""LIME word-level explanations for any Predictor."""
from lime.lime_text import LimeTextExplainer

from src.config import LABELS
from src.predict import Predictor

_explainer = LimeTextExplainer(class_names=LABELS, random_state=42)


def explain(predictor: Predictor, text: str, num_features: int = 8, num_samples: int = 500):
    """Return [(word, weight)] where positive weight pushes toward Sarcastic."""
    exp = _explainer.explain_instance(text, predictor.predict_proba, labels=[1],
                                      num_features=num_features, num_samples=num_samples)
    return [(str(w), float(s)) for w, s in exp.as_list(label=1)]


if __name__ == "__main__":
    import sys
    p = Predictor("tfidf_lr")
    t = " ".join(sys.argv[1:]) or "wah kya baat hai, bilkul time pe aaye"
    print(p.predict(t))
    print(explain(p, t))
