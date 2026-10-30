import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.explain import explain
from src.predict import Predictor, available_models

FEEDBACK = ROOT / "data" / "feedback.csv"
EXAMPLES = [
    "wah kya baat hai, bilkul time pe aaye",
    "thanks yaar, time pe aaye, maza aa gaya",
    "great, another Monday meeting, just what I wanted",
]

st.set_page_config(page_title="Hinglish Sarcasm Detector", page_icon="🎭")
st.title("🎭 Hinglish Sarcasm Detector")
st.caption("NLP mini project · PDEU · Khushi Panchotiya (23BIT112) & Grishma Makwana (23BIT108)")


@st.cache_resource
def load(name):
    return Predictor(name)


models = available_models()
if not models:
    st.error("No trained model found in models/. Run `python -m src.train_baseline` first.")
    st.stop()

model_name = st.sidebar.selectbox("Model", models, index=len(models) - 1)
show_lime = st.sidebar.checkbox("Explain with LIME (slower)", value=True)
st.sidebar.write("Try an example:")
for ex in EXAMPLES:
    if st.sidebar.button(ex):
        st.session_state["text"] = ex

tab_detect, tab_fool = st.tabs(["Detect", "Try to fool the model"])


def save_feedback(text, predicted, corrected):
    FEEDBACK.parent.mkdir(exist_ok=True)
    new = not FEEDBACK.exists()
    with FEEDBACK.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time_utc", "text", "model", "predicted", "corrected"])
        w.writerow([datetime.now(timezone.utc).isoformat(), text, model_name,
                    predicted, corrected])


def run(text, key):
    pred = load(model_name).predict(text)
    icon = "😏" if pred["label"] == "Sarcastic" else "🙂"
    st.subheader(f"{icon} {pred['label']}")
    st.progress(pred["confidence"], text=f"Confidence: {pred['confidence']:.0%}")
    if show_lime:
        with st.spinner("Computing explanation..."):
            words = explain(load(model_name), text)
        st.write("**Words that pushed the prediction** (red = Sarcastic, blue = Sincere)")
        chips = []
        for tok, wt in words:
            rgb = "220,60,60" if wt > 0 else "60,100,220"
            alpha = min(abs(wt) * 4 + 0.15, 0.9)
            chips.append(f"<span style='background:rgba({rgb},{alpha:.2f});padding:2px 6px;"
                         f"border-radius:4px'>{tok} ({wt:+.2f})</span>")
        st.markdown(" ".join(chips), unsafe_allow_html=True)
    st.write("Was this wrong? Tell us the right label:")
    cols = st.columns(2)
    for col, lab in zip(cols, ("Sarcastic", "Sincere")):
        if col.button(f"It is {lab}", key=f"{key}_{lab}"):
            save_feedback(text, pred["label"], lab)
            st.success("Saved for future retraining. Thanks!")


with tab_detect:
    text = st.text_area("Enter a Hinglish comment", key="text", height=100)
    if st.button("Detect", type="primary") and text.strip():
        run(text, "d")

with tab_fool:
    st.write("Try to write a comment the model gets wrong.")
    goal = st.radio("Your goal", ["Sarcastic comment that looks Sincere",
                                  "Sincere comment that looks Sarcastic"])
    t2 = st.text_area("Your attempt", key="fool", height=100)
    if st.button("Test it") and t2.strip():
        pred = load(model_name).predict(t2)
        want = "Sincere" if goal.startswith("Sarcastic") else "Sarcastic"
        if pred["label"] == want:
            st.success(f"You fooled it! Model says {pred['label']} ({pred['confidence']:.0%}).")
        else:
            st.warning(f"Not fooled. Model says {pred['label']} ({pred['confidence']:.0%}).")
        run(t2, "f")
