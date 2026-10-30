from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "human_labelled_features.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

SEED = 42
LABELS = ["Sincere", "Sarcastic"]
LABEL2ID = {"Sincere": 0, "Sarcastic": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

LABEL_MAP = {
    "sarcastic": "Sarcastic",
    "non-sarcastic": "Sincere",
    "non_sarcastic": "Sincere",
    "non sarcastic": "Sincere",
}

# Hugging Face checkpoints compared in the project.
TRANSFORMERS = {
    "muril": "google/muril-base-cased",
    "xlmr": "xlm-roberta-base",
    "hingbert": "l3cube-pune/hing-bert",
}
