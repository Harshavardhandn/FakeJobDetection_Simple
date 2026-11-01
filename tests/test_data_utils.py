import pandas as pd
from src.data_utils import prepare_dataframe, clean_text
def test_prepare_dataframe():
 d=pd.DataFrame({"Job Title":["Developer","Fake job"],"Job Description":["Build software","Pay fee"],"Fraudulent":[0,1]})
 out=prepare_dataframe(d)
 assert list(out.fraudulent)==[0,1]
 assert "developer" in out.iloc[0].text
def test_clean_text():
 assert "url" in clean_text("Apply at https://example.com")
