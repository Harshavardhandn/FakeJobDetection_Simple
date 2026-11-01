from pathlib import Path
import joblib,numpy as np,torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
ROOT=Path(__file__).resolve().parents[1]
ML_FILES={"Logistic Regression":ROOT/"models/logistic_regression.joblib","Random Forest":ROOT/"models/random_forest.joblib","SVM":ROOT/"models/svm.joblib"}
def available_ml_models(): return [n for n,p in ML_FILES.items() if p.exists()]
def explain(model,text):
 try:
  v=model.named_steps["tfidf"]; s=model.named_steps["select"]; row=s.transform(v.transform([text])).toarray()[0]
  terms=v.get_feature_names_out()[s.get_support()]
  return [str(terms[i]) for i in np.argsort(row)[::-1] if row[i]>0][:7] or ["No strong text indicators extracted."]
 except Exception: return ["Simple term explanation unavailable."]
def predict_ml(text,name):
 m=joblib.load(ML_FILES[name]); y=int(m.predict([text])[0]); probs=m.predict_proba([text])[0]
 return {"class":y,"label":"Fraudulent" if y else "Genuine","confidence":float(probs[y]),"explanation":explain(m,text)}
def predict_bert(text):
 d=ROOT/"models/distilbert"
 if not (d/"config.json").exists(): raise FileNotFoundError("Train DistilBERT first.")
 tok=AutoTokenizer.from_pretrained(str(d)); m=AutoModelForSequenceClassification.from_pretrained(str(d)); m.eval()
 with torch.no_grad(): probs=torch.softmax(m(**tok(text,return_tensors="pt",truncation=True,max_length=256)).logits,dim=1)[0]
 y=int(torch.argmax(probs))
 return {"class":y,"label":"Fraudulent" if y else "Genuine","confidence":float(probs[y]),
 "explanation":["DistilBERT uses contextual language representations.","This is a prediction, not proof of fraud."]}
