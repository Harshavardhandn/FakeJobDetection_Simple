import argparse,json
from pathlib import Path
import joblib,pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from src.data_utils import prepare_dataframe
from src.predict import ML_FILES,predict_bert,ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--data",default=str(ROOT/"data/raw/naukri_jobs.csv"))
 p.add_argument("--output",default=str(ROOT/"results/naukri_predictions.csv")); a=p.parse_args()
 df=prepare_dataframe(pd.read_csv(a.data),require_labels=False); out=df.copy()
 labelled="fraudulent" in df.columns
 report={"rows":len(df),"labels_available":labelled,"models":{}}
 for name,path in ML_FILES.items():
  if path.exists():
   model=joblib.load(path); pred=model.predict(df.text).astype(int)
   out[name.lower().replace(" ","_")+"_prediction"]=pred
   if labelled:
    y=df.fraudulent
    report["models"][name]={"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),
     "recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),
     "confusion_matrix":confusion_matrix(y,pred).tolist()}
 if (ROOT/"models/distilbert/config.json").exists():
  rs=[predict_bert(t) for t in df.text]; pred=[r["class"] for r in rs]
  out["distilbert_prediction"]=pred; out["distilbert_confidence"]=[r["confidence"] for r in rs]
  if labelled:
   y=df.fraudulent
   report["models"]["DistilBERT"]={"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),
    "recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),
    "confusion_matrix":confusion_matrix(y,pred).tolist()}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False)
 Path(a.output).with_suffix(".json").write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
 if not labelled: print("Predictions only: no verified labels, so no external metrics were calculated.")
if __name__=="__main__": main()
