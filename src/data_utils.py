import re
from pathlib import Path
import pandas as pd

FIELDS = ["title","company_profile","description","requirements","benefits","location",
          "employment_type","required_experience","required_education","industry","function"]
ALIASES = {
 "fraudulent":["fraudulent","is_fraudulent","label","target","class"],
 "title":["title","job_title","position"],
 "company_profile":["company_profile","company","company_description"],
 "description":["description","job_description","details"],
 "requirements":["requirements","qualification","qualifications"],
 "benefits":["benefits","perks"], "location":["location","city","job_location"]
}
def normalize_columns(df):
    df=df.copy()
    df.columns=[re.sub(r"[^a-z0-9]+","_",str(c).strip().lower()).strip("_") for c in df.columns]
    rename={}
    for target, aliases in ALIASES.items():
        for a in aliases:
            if a in df.columns:
                rename[a]=target; break
    return df.rename(columns=rename)
def clean_text(v):
    if pd.isna(v): return ""
    s=str(v).lower()
    s=re.sub(r"<[^>]+>"," ",s)
    s=re.sub(r"http\S+|www\.\S+"," URL ",s)
    s=re.sub(r"[\r\n\t]+"," ",s)
    s=re.sub(r"[^a-z0-9₹$%.,!?'-]+"," ",s)
    return re.sub(r"\s+"," ",s).strip()
def prepare_dataframe(df, require_labels=True):
    df=normalize_columns(df)
    cols=[c for c in FIELDS if c in df.columns]
    if not cols: raise ValueError("No recognized job-text columns. Include title or description.")
    for c in cols: df[c]=df[c].fillna("").astype(str)
    df["text"]=df[cols].agg(" ".join,axis=1).map(clean_text)
    df=df[df.text.str.len()>0].copy()
    if "fraudulent" in df:
        df["fraudulent"]=pd.to_numeric(df["fraudulent"],errors="coerce")
        df=df[df.fraudulent.isin([0,1])].copy()
        df["fraudulent"]=df.fraudulent.astype(int)
    elif require_labels:
        raise ValueError("A verified fraudulent column is required: 0=genuine, 1=fraudulent.")
    return df.reset_index(drop=True)
def load_csv(path, require_labels=True):
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(path)
    return prepare_dataframe(pd.read_csv(path),require_labels)
