# AI-Based Fake Job Posting Detection — Complete Project

## Included
1. EMSCAD and Naukri CSV loading, column normalization, cleaning and text combination.
2. TF-IDF + Chi-Square + Logistic Regression, Random Forest and calibrated SVM.
3. DistilBERT fine-tuning using Hugging Face Transformers.
4. Accuracy, precision, recall, F1-score and confusion matrices.
5. Naukri external predictions; external metrics only when independently verified labels exist.
6. Streamlit dashboard, model training controls, job prediction, confidence and simple text indicators.

## Datasets
Download EMSCAD from:
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

Place it at `data/raw/fake_job_postings.csv`. Expected target: `fraudulent` (0=genuine, 1=fraudulent).

Place your permitted Naukri CSV at `data/raw/naukri_jobs.csv`. Recognized text columns include title, company profile, description, requirements, benefits and location.

The original datasets are not bundled. Respect their licences and keep track of source, version and download date.

## Windows / VS Code setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```
For ML + Streamlit only (without BERT):
```powershell
pip install -r requirements-ml-only.txt
```

## Run
```powershell
streamlit run app.py
```

## Train traditional ML
```powershell
python -m src.train_ml
```
Optional:
```powershell
python -m src.train_ml --features 5000
```
Uses a stratified 80/20 EMSCAD split (random_state=42). Each sklearn pipeline fits TF-IDF and Chi-Square only on training data, avoiding test leakage. Models are saved in `models/`; metrics in `results/ml_metrics.json`.

## Fine-tune DistilBERT
```powershell
python -m src.train_bert --epochs 2 --batch-size 8
```
Quick smoke run:
```powershell
python -m src.train_bert --epochs 1 --sample-size 1500 --batch-size 4
```
The first run downloads `distilbert-base-uncased`; internet is required. A GPU is recommended. Do not compare a sampled quick run against full-data ML as a final apples-to-apples comparison. For final reporting, use the same labelled source data and matching fixed train/test rows. The model is saved to `models/distilbert/`, metrics to `results/bert_metrics.json`.

## Naukri external validation
```powershell
python -m src.external_validation
```
Saves `results/naukri_predictions.csv` and `results/naukri_predictions.json`. If no verified `fraudulent` labels exist, it outputs predictions only. It cannot calculate external accuracy or F1 without ground truth.

### Important: how to label a Naukri sample
Do not use model predictions as labels. Independently review a documented sample using employer career pages, verified company contact information and credible evidence. Add `fraudulent` with 0 or 1, exclude unresolved cases, and ideally use a second reviewer for a subset. See `LABELING_GUIDELINES.md`. This is necessary for defensible external metrics.

## Explanation and confidence
Traditional ML shows high TF-IDF terms present in the posting as simple indicators, not causal explanations. DistilBERT returns its predicted class and confidence; SHAP/LIME token attribution is not included in this version. Confidence is not proof that a job is fake or genuine.

## Research cautions
- Report precision and recall, not accuracy alone, because fraudulent jobs may be rare.
- Deduplicate near-identical postings before splitting where possible.
- A random split is a baseline; time-based/source-based evaluation can better test generalization.
- Keep Naukri external data separate from training for a genuine external validation.
- Record dataset versions, label protocol, source URLs and dates.
