import argparse,json
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from src.data_utils import load_csv
ROOT=Path(__file__).resolve().parents[1]
def metrics(y,p):
 return {"accuracy":float(accuracy_score(y,p)),"precision":float(precision_score(y,p,zero_division=0)),
 "recall":float(recall_score(y,p,zero_division=0)),"f1":float(f1_score(y,p,zero_division=0)),
 "confusion_matrix":confusion_matrix(y,p).tolist(),"report":classification_report(y,p,output_dict=True,zero_division=0)}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--data",default=str(ROOT/"data/raw/fake_job_postings.csv"))
 ap.add_argument("--test-size",type=float,default=.2); ap.add_argument("--features",type=int,default=5000); a=ap.parse_args()
 df=load_csv(a.data,True)
 if df.fraudulent.nunique()!=2: raise ValueError("Both classes 0 and 1 are required.")
 Xtr,Xte,ytr,yte=train_test_split(df.text,df.fraudulent,test_size=a.test_size,random_state=42,stratify=df.fraudulent)
 def base_steps():
  return [("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=50000,sublinear_tf=True,stop_words="english")),
          ("select",SelectKBest(chi2,k=a.features))]
 models={
 "Logistic Regression":Pipeline(base_steps()+[("classifier",LogisticRegression(max_iter=1500,class_weight="balanced"))]),
 "Random Forest":Pipeline(base_steps()+[("classifier",RandomForestClassifier(n_estimators=250,class_weight="balanced_subsample",random_state=42,n_jobs=-1,min_samples_leaf=2))]),
 "SVM":Pipeline(base_steps()+[("classifier",CalibratedClassifierCV(LinearSVC(class_weight="balanced",random_state=42),cv=3))])}
 out={}
 (ROOT/"models").mkdir(exist_ok=True); (ROOT/"results").mkdir(exist_ok=True)
 for name,m in models.items():
  print("Training",name); m.fit(Xtr,ytr); p=m.predict(Xte); out[name]=metrics(yte,p)
  joblib.dump(m,ROOT/"models"/(name.lower().replace(" ","_")+".joblib"))
 payload={"rows":len(df),"train_rows":len(Xtr),"test_rows":len(Xte),"models":out}
 (ROOT/"results/ml_metrics.json").write_text(json.dumps(payload,indent=2))
 print("Saved ML models and results/ml_metrics.json")
if __name__=="__main__": main()
