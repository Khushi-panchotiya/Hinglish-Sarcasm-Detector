"""Minimal web app: type a Hinglish line, get Sarcastic / Sincere.

    python web/server.py            # http://localhost:5000
    python web/server.py --model muril --port 8000
"""
import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from flask import Flask, jsonify, request, send_from_directory

from src.explain import explain
from src.predict import Predictor, available_models

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))
predictor = None
MAX_CHARS = 500


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify(error="Please enter some text."), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Please keep it under {MAX_CHARS} characters."), 400
    result = predictor.predict(text)
    if data.get("explain"):
        result["words"] = explain(predictor, text)
    result["model"] = predictor.name
    return jsonify(result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="tfidf_lr (default) or a fine-tuned folder name")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    models = available_models()
    if not models:
        sys.exit("No trained model in models/. Run: python -m src.train_baseline")
    predictor = Predictor(a.model or models[-1])
    print(f"Using model: {predictor.name}")
    app.run(host="127.0.0.1", port=a.port)
