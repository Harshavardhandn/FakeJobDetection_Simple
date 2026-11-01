from pathlib import Path
import json,subprocess,sys
import pandas as pd,streamlit as st,plotly.express as px
from src.data_utils import prepare_dataframe
from src.predict import available_ml_models,predict_ml,predict_bert,ROOT
st.set_page_config(page_title="Fake Job Detection",page_icon="🛡️",layout="wide")
st.markdown("""<style>.main{background:#f7f9fc}.block-container{padding-top:1.5rem;max-width:1300px}
[data-testid="stMetric"]{padding:14px;border:1px solid #e7ebf2;border-radius:12px}</style>""",unsafe_allow_html=True)
R=ROOT/"results"
def readj(p): return json.loads(p.read_text()) if p.exists() else None
page=st.sidebar.radio("Navigation",["Dashboard","Dataset Explorer","Train Models","Predict a Job","Naukri External Validation","About"])
st.sidebar.caption("Research prototype • Prediction is not proof of fraud.")
if page=="Dashboard":
 st.title("🛡️ AI-Based Fake Job Posting Detection")
 st.write("Traditional ML + DistilBERT + independent external validation")
 ml=readj(R/"ml_metrics.json"); bert=readj(R/"bert_metrics.json")
 a,b,c=st.columns(3); a.metric("ML models ready",len(available_ml_models())); b.metric("DistilBERT","Ready" if bert else "Not trained")
 c.metric("Naukri validation","Done" if (R/"naukri_predictions.json").exists() else "Not run")
 rows=[]
 if ml:
  for n,m in ml["models"].items(): rows.append({"Model":n,**{k:m[k] for k in ["accuracy","precision","recall","f1"]}})
 if bert: rows.append({"Model":"DistilBERT",**{k:bert[k] for k in ["accuracy","precision","recall","f1"]}})
 if rows:
  d=pd.DataFrame(rows); st.dataframe(d,use_container_width=True)
  st.plotly_chart(px.bar(d.melt("Model",var_name="Metric",value_name="Score"),x="Model",y="Score",color="Metric",barmode="group",range_y=[0,1]),use_container_width=True)
 else: st.info("Add EMSCAD and train models to populate this dashboard.")
 st.warning("Do not use model output as proof that an employer is fraudulent.")
elif page=="Dataset Explorer":
 st.title("Dataset Explorer"); f=st.file_uploader("Upload EMSCAD or Naukri CSV",type=["csv"])
 if f:
  try:
   raw=pd.read_csv(f); st.write("Original rows/columns:",raw.shape); st.dataframe(raw.head(8),use_container_width=True)
   d=prepare_dataframe(raw,require_labels=False); st.success(f"Prepared {len(d)} usable postings.")
   st.dataframe(d[["text"]+(["fraudulent"] if "fraudulent" in d else [])].head(10),use_container_width=True)
   if "fraudulent" not in d: st.info("No verified labels found: prediction is possible, metric evaluation is not.")
   st.download_button("Download prepared CSV",d.to_csv(index=False),"prepared_jobs.csv")
  except Exception as e: st.error(str(e))
elif page=="Train Models":
 st.title("Train ML and DistilBERT")
 data=ROOT/"data/raw/fake_job_postings.csv"
 st.write("EMSCAD expected at `data/raw/fake_job_postings.csv`.")
 if data.exists(): st.success(f"Dataset found: {len(pd.read_csv(data)):,} rows")
 else: st.error("Place the EMSCAD CSV in data/raw/ before training.")
 k=st.number_input("Chi-Square selected features",100,30000,5000,500)
 if st.button("Train three traditional ML models",disabled=not data.exists()):
  with st.spinner("Training ML models..."):
   r=subprocess.run([sys.executable,"-m","src.train_ml","--features",str(int(k))],cwd=ROOT,capture_output=True,text=True)
  st.code(r.stdout)
  if r.returncode==0: st.success("ML training completed.")
  else: st.error(r.stderr[-4000:])
 st.divider(); st.caption("DistilBERT downloads pretrained weights on first run. GPU recommended; CPU can be slow.")
 ep=st.number_input("DistilBERT epochs",1,5,2); sample=st.number_input("Quick-run sample size (0 = full data)",0,100000,0,500)
 if st.button("Fine-tune DistilBERT",disabled=not data.exists()):
  cmd=[sys.executable,"-m","src.train_bert","--epochs",str(int(ep))]
  if sample: cmd += ["--sample-size",str(int(sample))]
  with st.spinner("Fine-tuning DistilBERT..."):
   r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
  st.code(r.stdout[-4000:])
  if r.returncode==0: st.success("DistilBERT training completed.")
  else: st.error(r.stderr[-4000:])
elif page=="Predict a Job":
 st.title("Predict a job posting")
 title=st.text_input("Job title"); company=st.text_input("Company profile (optional)")
 desc=st.text_area("Job description",height=180); req=st.text_area("Requirements (optional)",height=100)
 opts=available_ml_models()
 if (ROOT/"models/distilbert/config.json").exists(): opts.append("DistilBERT")
 if not opts: st.info("Train at least one model first.")
 else:
  name=st.selectbox("Model",opts)
  if st.button("Analyze job"):
   text=" ".join([title,company,desc,req]).strip().lower()
   if not text: st.error("Enter a job title or description.")
   else:
    try:
     r=predict_bert(text) if name=="DistilBERT" else predict_ml(text,name)
     st.subheader(("🔴 " if r["class"] else "🟢 ")+r["label"])
     st.metric("Model confidence",f"{r['confidence']:.1%}"); st.progress(r["confidence"])
     st.write("Text indicators / model notes:")
     for x in r["explanation"]: st.write("• "+x)
     st.caption("Indicators are not causal proof. Verify employers independently.")
    except Exception as e: st.error(str(e))
elif page=="Naukri External Validation":
 st.title("External validation on Naukri")
 st.info("No verified fraudulent labels means predictions only. Accuracy, precision, recall and F1 require independent ground-truth labels.")
 path=ROOT/"data/raw/naukri_jobs.csv"
 if path.exists():
  st.success(f"Naukri data found: {len(pd.read_csv(path)):,} rows")
  if st.button("Run external predictions"):
   with st.spinner("Processing Naukri postings..."):
    r=subprocess.run([sys.executable,"-m","src.external_validation"],cwd=ROOT,capture_output=True,text=True)
   st.code(r.stdout[-5000:])
   if r.returncode==0: st.success("External processing complete.")
   else: st.error(r.stderr[-4000:])
  out=R/"naukri_predictions.csv"
  if out.exists():
   st.dataframe(pd.read_csv(out).head(30),use_container_width=True)
   st.download_button("Download all predictions",out.read_bytes(),"naukri_predictions.csv")
   report=readj(R/"naukri_predictions.json")
   if report and report["labels_available"]: st.json(report["models"])
 else: st.warning("Place Naukri CSV at data/raw/naukri_jobs.csv.")
else:
 st.title("About")
 st.markdown("""
**Objective:** identify potentially fraudulent job advertisements using text classification.

**Pipeline:** EMSCAD preparation → TF-IDF → Chi-Square → Logistic Regression / Random Forest / SVM → DistilBERT fine-tuning → common metrics → Naukri external validation → Streamlit prediction.

**Labels:** 0 = genuine, 1 = fraudulent.

A model prediction is a research signal, not proof. Data quality, class imbalance, duplicate postings and source differences can affect results.
""")
