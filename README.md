# Hinglish Sarcasm Detector

Detects sarcasm in code-mixed Hindi-English (Hinglish) comments. NLP mini project, ICT Semester 7, Pandit Deendayal Energy University.

**Team:** Khushi Panchotiya (23BIT112), Grishma Makwana (23BIT108)

## Data
Public dataset [`mananjen/hinglish_sarcasm_dataset`](https://github.com/mananjen/hinglish_sarcasm_dataset) (`human_labelled_features.csv`, Jain & Nagulapalli, 2025). The target is the `human_labelled` column. **License: no license file was found in the repository on 2026-10-26; confirm permitted use with the authors before publishing the data.** Comments may contain offensive content.

After cleaning (3 invalid labels, 24 conflicting duplicate texts, exact duplicates removed): 8,464 comments (4,819 Sincere / 3,645 Sarcastic), stratified 80/10/10 split (6,771 / 846 / 847), seed 42.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.prepare_data      # clean + split -> data/processed/
python -m src.train_baseline    # TF-IDF + Logistic Regression -> models/, reports/
streamlit run app/app.py       # Streamlit app (LIME, fool-the-model, feedback)
python web/server.py           # simple web app at http://localhost:5000
```

## Fine-tuning transformers (GPU needed)
On Colab/Kaggle: `pip install torch transformers datasets accelerate`, copy the project, then
```bash
python -m src.finetune --model muril      # google/muril-base-cased
python -m src.finetune --model xlmr       # xlm-roberta-base
python -m src.finetune --model hingbert   # l3cube-pune/hing-bert
```
Download `models/<name>/final/` back into this project; the app lists it automatically.

## Other tools
- `python -m src.explain "text"` LIME explanation
- `python -m src.error_analysis --model tfidf_lr` samples 30 errors to categorise by hand

## Results (test set, 847 comments)
| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority class | 0.569 | 0.363 |
| TF-IDF + Logistic Regression | 0.661 | 0.659 |
| MuRIL | 0.667 | 0.665 |
| XLM-RoBERTa | 0.647 | 0.646 |
| HingBERT | 0.694 | 0.693 |

## Layout
`src/` pipeline code · `app/` Streamlit app · `notebooks/` original exploration notebook · `docs/` plan and guideline · `reports/` metrics and error files · `data/` raw and processed data · `models/` trained models

## Limitations
Standalone comments lack context; spelling and code-mixing vary; labels are subjective and come from one annotator in the source data, and no inter-annotator agreement (Cohen's kappa) was measured by us; results may not transfer to other platforms.
# Hinglish-Sarcasm-Detector
